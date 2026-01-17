import streamlit as st
import json
import fnmatch
import requests
from dataclasses import dataclass
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# =========================================================
# Utilities
# =========================================================
def to_list(x):
    if x is None: return []
    return x if isinstance(x, list) else [x]

def wildcard_match(value, patterns):
    for p in to_list(patterns):
        if fnmatch.fnmatchcase(str(value), str(p)):
            return True
    return False

# =========================================================
# IAM Core
# =========================================================
@dataclass
class Statement:
    effect: str
    action: List[str]
    not_action: List[str]
    resource: List[str]
    not_resource: List[str]
    condition: Dict[str, Any]

class PolicyEngine:
    def __init__(self, policy):
        self.statements = self._parse(policy)

    def _parse(self, policy):
        stmts = policy.get("Statement", [])
        if isinstance(stmts, dict):
            stmts = [stmts]
        out = []
        for s in stmts:
            out.append(Statement(
                effect=s.get("Effect", "Allow"),
                action=to_list(s.get("Action")),
                not_action=to_list(s.get("NotAction")),
                resource=to_list(s.get("Resource")),
                not_resource=to_list(s.get("NotResource")),
                condition=s.get("Condition", {})
            ))
        return out

    def evaluate(self, action, resource="*"):
        decision = "ImplicitDeny"
        for s in self.statements:
            if s.action and not wildcard_match(action, s.action):
                continue
            if s.not_action and wildcard_match(action, s.not_action):
                continue
            if s.resource and not wildcard_match(resource, s.resource):
                continue
            if s.not_resource and wildcard_match(resource, s.not_resource):
                continue
            if s.effect == "Deny":
                return "ExplicitDeny"
            decision = "Allow"
        return decision

# =========================================================
# Base Risk Capabilities
# =========================================================
CAPABILITIES = [
    {"name": "Policy Privilege Manipulation", "actions": ["iam:CreatePolicyVersion"], "risk": 10},
    {"name": "Credential Minting", "actions": ["iam:CreateAccessKey"], "risk": 9},
    {"name": "User Persistence", "actions": ["iam:CreateLoginProfile"], "risk": 9},
    {"name": "Resource Takeover", "actions": ["sts:AssumeRole"], "risk": 10},
    {"name": "Role Pass to EC2", "actions": ["iam:PassRole", "ec2:RunInstances"], "risk": 8},
    {"name": "Lambda Privileged Execution", "actions": ["lambda:CreateFunction", "iam:PassRole"], "risk": 7},
    {"name": "S3 Data Leak", "actions": ["s3:GetBucketPolicy", "s3:ListBucket"], "risk": 8},
]

# =========================================================
# Logic-Aware Analyzer
# =========================================================
def analyze_policy(policy_json):
    engine = PolicyEngine(policy_json)
    findings = []

    HIGH_LEVEL_CAPS = CAPABILITIES + [
        {"name": "Full Admin Access (*)", "actions": ["*"], "risk": 10},
        {"name": "Full IAM Administrative", "actions": ["iam:*"], "risk": 10},
        {"name": "Full S3 Management", "actions": ["s3:*"], "risk": 9},
        {"name": "Dangerous Resource Star", "actions": ["iam:GetPolicy"], "risk": 7}
    ]

    for cap in HIGH_LEVEL_CAPS:
        if any(engine.evaluate(act) == "Allow" for act in cap["actions"]):
            findings.append(cap)

    return findings

# =========================================================
# Wayback Machine — Deep Search
# =========================================================
def fetch_json_urls(domain):
    try:
        api = (
            f"https://web.archive.org/cdx/search/cdx"
            f"?url=*.{domain}/*&output=json&collapse=urlkey"
            f"&filter=statuscode:200&limit=1000"
        )
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) SovereignHunter/3.0'}
        r = requests.get(api, headers=headers, timeout=30)
        if r.status_code != 200: return []

        data = r.json()
        urls = []

        targets = ['.json', '.env', '.config', '.txt', 'aws', 'iam', 'cred']

        for row in data[1:]:
            path = row[2].lower()
            if any(t in path for t in targets):
                ts = row[1]
                urls.append(f"https://web.archive.org/web/{ts}id_/{row[2]}")

        return list(set(urls))
    except:
        return []

def extract_policies(urls, workers=12):
    results = []

    def fetch(url):
        try:
            r = requests.get(url, timeout=8)
            if r.status_code != 200: return None
            data = r.json()
            if isinstance(data, dict) and "Statement" in data:
                return url, data
            if isinstance(data, list):
                for obj in data:
                    if isinstance(obj, dict) and "Statement" in obj:
                        return url, obj
        except:
            return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, u) for u in urls]
        for f in as_completed(futures):
            if f.result():
                results.append(f.result())
    return results

# =========================================================
# PDF Report
# =========================================================
def build_pdf(domain, findings):
    buf = BytesIO()
    doc = SimpleDocTemplate(buf)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Sovereign IAM Hunter Report", styles["Title"]))
    story.append(Spacer(1,12))
    story.append(Paragraph(f"Target Domain: <b>{domain}</b>", styles["Normal"]))
    story.append(Spacer(1,12))
    if not findings:
        story.append(Paragraph("No high-risk IAM policies found.", styles["Normal"]))
    else:
        for url, caps in findings.items():
            story.append(Spacer(1,8))
            story.append(Paragraph(f"<b>Source:</b> {url}", styles["Normal"]))
            for c in caps:
                story.append(
                    Paragraph(
                        f"• <b>{c['name']}</b> — Risk {c['risk']}/10<br/>Actions: {', '.join(c['actions'])}",
                        styles["Normal"]
                    )
                )
    doc.build(story)
    buf.seek(0)
    return buf

# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config("Sovereign IAM Hunter Pro Ultimate", layout="wide")
st.title("🛡️ Sovereign IAM Hunter Pro — Ultimate Deep Recon")

tab1, tab2 = st.tabs(["🧪 Manual Policy Analysis", "🌐 Automated Deep Recon"])

# ---------------- Manual ----------------
with tab1:
    policy_text = st.text_area("Paste IAM Policy JSON", height=320)
    if st.button("Analyze Policy"):
        try:
            policy = json.loads(policy_text)
            findings = analyze_policy(policy)
            if findings:
                st.error("High-Risk Capabilities Detected")
                for f in findings:
                    st.write(f"• {f['name']} (Risk {f['risk']}/10)")
            else:
                st.success("No critical IAM risks detected.")
        except Exception as e:
            st.error(f"Invalid JSON: {e}")

# ---------------- Automated ----------------
with tab2:
    domain = st.text_input("Target Domain (e.g., tesla.com)")
    if st.button("Start Deep Recon"):
        if not domain:
            st.error("Please enter a domain.")
        else:
            with st.spinner("Querying Wayback Machine deeply..."):
                urls = fetch_json_urls(domain)

            st.info(f"Discovered {len(urls)} potentially interesting files.")

            findings_map = {}
            policies = extract_policies(urls[:60])  # limit for speed

            for url, policy in policies:
                caps = analyze_policy(policy)
                if caps:
                    findings_map[url] = caps
                    st.warning(f"IAM Risk Found → {url}")
                    for c in caps:
                        st.write(f"• {c['name']} (Risk {c['risk']}/10)")

            if not findings_map:
                st.success("No risky IAM policies discovered.")

            pdf = build_pdf(domain, findings_map)
            st.download_button(
                "📄 Download PDF Report",
                pdf,
                file_name=f"{domain}_IAM_Report.pdf",
                mime="application/pdf"
            )

st.sidebar.markdown("""
### ✔ Deep Cloud Professional
• Subdomain scanning (*.domain.com)  
• Multi-file types & keywords (.json, .env, .config, .txt, aws, iam, cred)  
• Detects wildcard '*' and iam:* / s3:*  
• Multi-threaded extraction  
• Raw JSON fetch (id_)  
• Limit 60 files for performance  
""")
