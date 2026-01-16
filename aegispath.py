import streamlit as st
import hcl2
import io
import networkx as nx

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet


# =====================================================
# Terraform Parser
# =====================================================
def load_terraform_files(uploaded_files):
    parsed = []
    for file in uploaded_files:
        try:
            content = io.StringIO(file.getvalue().decode("utf-8"))
            parsed.append(hcl2.load(content))
        except Exception:
            continue
    return parsed


# =====================================================
# Policy Engine (Allow / Deny aware)
# =====================================================
class PolicyEngine:
    def __init__(self, policy):
        self.statements = policy.get("Statement", [])

    def allows(self, action):
        allowed = False

        for stmt in self.statements:
            effect = stmt.get("Effect", "Deny")
            actions = stmt.get("Action", [])

            if isinstance(actions, str):
                actions = [actions]

            match = (
                action in actions or
                "*" in actions or
                action.split(":")[0] + ":*" in actions
            )

            if match:
                if effect == "Deny":
                    return False
                if effect == "Allow":
                    allowed = True

        return allowed


# =====================================================
# Role Classification
# =====================================================
def classify_role(engine):
    if engine.allows("*") or engine.allows("iam:*"):
        return "Tier-1"
    return "Tier-2"


# =====================================================
# Attack Graph
# =====================================================
class AttackGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_edge(self, src, dst, reason):
        self.graph.add_edge(src, dst, reason=reason)

    def find_paths(self, sources, targets, max_depth=4):
        paths = []
        for s in sources:
            for t in targets:
                try:
                    for p in nx.all_simple_paths(self.graph, s, t, cutoff=max_depth):
                        paths.append(p)
                except Exception:
                    pass
        return paths


# =====================================================
# Risk Scoring
# =====================================================
def score_path(path, target_tier):
    score = len(path) * 20
    if target_tier == "Tier-1":
        score += 30
    return min(score, 100)


# =====================================================
# PDF Report Generator (Sanitized)
# =====================================================
def sanitize(text):
    return text.replace("<", "").replace(">", "")

def generate_report(paths, scores, filename="AegisPath_Executive_Report.pdf"):
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph(
        "<b>AegisPath – Executive Cloud IAM Security Report</b>",
        styles["Title"]
    ))

    for path, score in zip(paths, scores):
        content.append(Paragraph(
            f"<b>Risk Score:</b> {score}/100<br/>"
            f"<b>Attack Path:</b> {' → '.join(map(sanitize, path))}",
            styles["Normal"]
        ))

    doc.build(content)


# =====================================================
# Streamlit UI
# =====================================================
st.set_page_config(layout="wide")
st.title("🛡️ AegisPath – Cloud IAM Attack Path Analyzer")

uploaded_files = st.file_uploader(
    "Upload Terraform (.tf) files",
    type=["tf"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Upload Terraform files to begin analysis.")
    st.stop()


# =====================================================
# Analysis Pipeline
# =====================================================
tf_data = load_terraform_files(uploaded_files)

roles = {}
policies = {}

for block in tf_data:
    for r_type, resources in block.get("resource", {}).items():

        if r_type == "aws_iam_role":
            for name, body in resources.items():
                roles[name] = body

        if r_type == "aws_iam_role_policy":
            for _, body in resources.items():
                role_name = body.get("role")
                if role_name:
                    policies.setdefault(role_name, []).append(body["policy"])


# Build Policy Engines
policy_engines = {}
for role, docs in policies.items():
    merged = {"Statement": []}
    for doc in docs:
        merged["Statement"].extend(doc.get("Statement", []))
    policy_engines[role] = PolicyEngine(merged)


# Classify Roles
tiers = {r: classify_role(e) for r, e in policy_engines.items()}
entry_roles = [r for r, t in tiers.items() if t != "Tier-1"]
admin_roles = [r for r, t in tiers.items() if t == "Tier-1"]


# Build Graph
graph = AttackGraph()

for src, engine in policy_engines.items():

    if engine.allows("iam:PassRole"):
        for target in roles:
            graph.add_edge(src, target, "iam:PassRole")

    if engine.allows("ec2:RunInstances"):
        for target in roles:
            graph.add_edge(src, target, "ec2:RunInstances")

    if engine.allows("lambda:CreateFunction"):
        for target in roles:
            graph.add_edge(src, target, "lambda:CreateFunction")


# Find Paths
paths = graph.find_paths(entry_roles, admin_roles)
results = []

for p in paths:
    results.append((p, score_path(p, "Tier-1")))

results.sort(key=lambda x: x[1], reverse=True)


# =====================================================
# UI Output
# =====================================================
st.subheader("🔥 Critical Privilege Escalation Paths")

if not results:
    st.success("No critical attack paths found.")
    st.stop()

for path, score in results[:5]:
    st.error(f"Risk Score: {score}/100")
    st.write(" → ".join(path))
    st.divider()


# =====================================================
# PDF Button
# =====================================================
if st.button("📄 Generate Executive PDF Report"):
    generate_report(
        [p for p, _ in results],
        [s for _, s in results]
    )
    st.success("PDF report generated successfully.")
