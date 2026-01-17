import streamlit as st
import json
import fnmatch
import requests
from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime

# =========================================================
# Utilities
# =========================================================
def ensure_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]

def wildcard_match(value, patterns):
    for p in ensure_list(patterns):
        if fnmatch.fnmatchcase(str(value), str(p)):
            return True
    return False

# =========================================================
# IAM Models
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
    def __init__(self, policy: Dict[str, Any]):
        self.statements = self._parse(policy)

    def _parse(self, policy):
        stmts = policy.get("Statement", [])
        if isinstance(stmts, dict):
            stmts = [stmts]

        parsed = []
        for s in stmts:
            parsed.append(Statement(
                effect=s.get("Effect", "Allow"),
                action=ensure_list(s.get("Action")),
                not_action=ensure_list(s.get("NotAction")),
                resource=ensure_list(s.get("Resource")),
                not_resource=ensure_list(s.get("NotResource")),
                condition=s.get("Condition", {})
            ))
        return parsed

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
# Capabilities (Enhanced)
# =========================================================
CAPABILITIES = [
    {
        "name": "Policy Privilege Manipulation",
        "actions": ["iam:CreatePolicyVersion"],
        "risk": 10
    },
    {
        "name": "Credential Minting",
        "actions": ["iam:CreateAccessKey"],
        "risk": 9
    },
    {
        "name": "Role Pass to Compute",
        "actions": ["iam:PassRole", "ec2:RunInstances"],
        "risk": 8
    },
    {
        "name": "Lambda Privileged Execution",
        "actions": ["lambda:CreateFunction", "iam:PassRole"],
        "risk": 7
    },
    {
        "name": "S3 Data Leak",
        "actions": ["s3:GetBucketPolicy", "s3:ListBucket"],
        "risk": 8
    },
    {
        "name": "User Persistence",
        "actions": ["iam:CreateLoginProfile"],
        "risk": 9
    },
    {
        "name": "Resource Takeover",
        "actions": ["sts:AssumeRole"],
        "risk": 10
    }
]

def analyze_policy(policy_json):
    engine = PolicyEngine(policy_json)
    findings = []

    for cap in CAPABILITIES:
        allowed = []
        for act in cap["actions"]:
            if engine.evaluate(act) == "Allow":
                allowed.append(act)

        if len(allowed) == len(cap["actions"]):
            findings.append({
                "capability": cap["name"],
                "risk": cap["risk"],
                "actions": cap["actions"]
            })

    return findings

# =========================================================
# Automated Recon (Public Data Only)
# =========================================================
def fetch_json_urls(domain):
    """Fetch JSON URLs using Wayback Machine CDX API (no Go required)"""
    try:
        cdx_api = f"http://web.archive.org/cdx/search/cdx?url={domain}/*&output=json&filter=mimetype:application/json"
        resp = requests.get(cdx_api, timeout=10)
        urls = []
        data = resp.json()
        for entry in data[1:]:
            urls.append("http://web.archive.org/web/" + entry[1] + "/" + entry[2])
        return urls
    except:
        return []

def extract_iam_policy(url):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        data = r.json()
        if isinstance(data, dict) and "Statement" in data:
            return data
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "Statement" in item:
                    return item
    except:
        pass
    return None

# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(page_title="Sovereign IAM Hunter Ultimate", layout="wide")
st.title("🛡️ Sovereign IAM Hunter — Ultimate Edition")

tab_manual, tab_auto = st.tabs(["🧪 Manual Analysis", "🌐 Automated Hunting"])

# ----------------- Manual Analysis ----------------------
with tab_manual:
    st.subheader("Manual IAM Policy Analysis")
    policy_text = st.text_area("Paste IAM Policy JSON", height=300)

    if st.button("Analyze Policy"):
        try:
            policy = json.loads(policy_text)
            findings = analyze_policy(policy)

            if not findings:
                st.success("✅ No high-risk IAM capabilities detected.")
            else:
                st.error(f"⚠️ {len(findings)} High-Risk Capabilities Found")
                for f in findings:
                    with st.expander(f"🚨 {f['capability']} — Risk {f['risk']}/10"):
                        st.write("Actions:", ", ".join(f["actions"]))
        except Exception as e:
            st.error(f"Invalid JSON: {e}")

# ----------------- Automated Hunting -------------------
with tab_auto:
    st.subheader("Public Archive IAM Hunting")
    domain = st.text_input("Target Domain (example.com)")

    if st.button("Start Hunting"):
        if not domain:
            st.error("Please enter a domain.")
        else:
            with st.spinner("Searching public archives..."):
                urls = fetch_json_urls(domain)

            st.info(f"Discovered {len(urls)} JSON files")

            total_findings = 0
            for url in urls[:40]:
                policy = extract_iam_policy(url)
                if not policy:
                    continue

                findings = analyze_policy(policy)
                if findings:
                    total_findings += len(findings)
                    st.warning(f"IAM Policy Found → {url}")
                    for f in findings:
                        st.write(f"• **{f['capability']}** (Risk {f['risk']}/10)")

            if total_findings == 0:
                st.success("No risky IAM policies discovered in public archives.")

# ----------------- Sidebar ----------------------------
st.sidebar.markdown("""
### 🧠 About
- Uses **Wayback Machine API** (no Go required)
- Defensive IAM risk analysis
- Reduced false positives
- Supports S3 leak, User persistence, Resource takeover
- Single-file Streamlit app, ready for Streamlit Cloud
""")
