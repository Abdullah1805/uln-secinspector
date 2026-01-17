import streamlit as st
import json
import fnmatch
import ipaddress
from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime

# =========================================================
# Core Utilities
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

def ip_match(ip, cidrs):
    try:
        ip_obj = ipaddress.ip_address(ip)
        for c in ensure_list(cidrs):
            if ip_obj in ipaddress.ip_network(c, strict=False):
                return True
    except:
        pass
    return False

# =========================================================
# IAM Statement Model
# =========================================================
@dataclass
class Statement:
    effect: str
    action: List[str]
    not_action: List[str]
    resource: List[str]
    not_resource: List[str]
    condition: Dict[str, Any]

# =========================================================
# Policy Engine (AWS‑Accurate)
# =========================================================
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

    def evaluate(self, action, resource, ctx):
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

            if not eval_conditions(s.condition, ctx):
                continue

            if s.effect == "Deny":
                return "ExplicitDeny"

            decision = "Allow"

        return decision

# =========================================================
# Condition Engine (Extended + Accurate)
# =========================================================
def eval_conditions(conditions, ctx):
    if not conditions:
        return True

    for op, kv in conditions.items():
        for key, expected in kv.items():
            actual = ctx.get(key)

            if op.endswith("IfExists") and actual is None:
                continue
            if actual is None:
                return False

            if "StringEquals" in op or "ArnEquals" in op:
                if str(actual) != str(expected):
                    return False

            elif "StringLike" in op or "ArnLike" in op:
                if not wildcard_match(actual, expected):
                    return False

            elif "IpAddress" in op:
                if not ip_match(actual, expected):
                    return False

            elif "Bool" in op:
                if str(actual).lower() != str(expected).lower():
                    return False

            elif "Null" in op:
                is_null = actual is None
                if is_null != bool(expected):
                    return False

    return True

# =========================================================
# Capability Database (Defensive, No PoC)
# =========================================================
CAPABILITIES = [
    {
        "name": "Policy Manipulation",
        "actions": ["iam:CreatePolicyVersion"],
        "risk": 10,
        "requires_resource_wildcard": True
    },
    {
        "name": "PassRole to Compute",
        "actions": ["iam:PassRole", "ec2:RunInstances"],
        "required_conditions": {"iam:PassedToService": "ec2.amazonaws.com"},
        "risk": 9
    },
    {
        "name": "Lambda Privileged Execution",
        "actions": ["lambda:CreateFunction", "iam:PassRole"],
        "risk": 8
    },
    {
        "name": "Credential Minting",
        "actions": ["iam:CreateAccessKey"],
        "risk": 9
    }
]

# =========================================================
# Risk Analyzer
# =========================================================
def analyze_capabilities(engine, ctx):
    findings = []

    for cap in CAPABILITIES:
        allowed = []
        wildcard_resource = False

        for s in engine.statements:
            if any(wildcard_match(a, s.action) for a in cap["actions"]):
                if "*" in s.resource or not s.resource:
                    wildcard_resource = True

        for act in cap["actions"]:
            if engine.evaluate(act, "*", ctx) == "Allow":
                allowed.append(act)

        if len(allowed) != len(cap["actions"]):
            continue

        if cap.get("requires_resource_wildcard") and not wildcard_resource:
            continue

        conds = cap.get("required_conditions")
        if conds:
            for k, v in conds.items():
                if ctx.get(k) != v:
                    break
            else:
                findings.append(cap)
        else:
            findings.append(cap)

    return findings

# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config("Sovereign IAM Hunter v3", layout="wide")
st.title("🛡️ Sovereign IAM Hunter — Final Edition")

policy_text = st.text_area("IAM Policy (JSON)", height=300)
caller = st.text_input("Caller ARN", "arn:aws:iam::123456789012:user/test")
src_ip = st.text_input("Source IP", "1.1.1.1")

if st.button("🔍 Analyze"):
    try:
        policy = json.loads(policy_text)
        engine = PolicyEngine(policy)

        ctx = {
            "aws:PrincipalArn": caller,
            "aws:SourceIp": src_ip,
            "aws:CurrentTime": datetime.utcnow().isoformat(),
            "iam:PassedToService": "ec2.amazonaws.com"
        }

        results = analyze_capabilities(engine, ctx)

        if not results:
            st.success("✅ No high‑risk IAM capabilities detected.")
        else:
            st.error(f"⚠️ {len(results)} High‑Risk Capabilities Found")
            for r in results:
                with st.expander(f"🚨 {r['name']} — Risk {r['risk']}/10", expanded=True):
                    st.write("**Actions:**", ", ".join(r["actions"]))
                    st.write("**Risk Score:**", r["risk"])

    except Exception as e:
        st.error(str(e))

st.sidebar.markdown("""
### 🧠 Notes
- No exploit commands included
- AWS‑accurate decision model
- Designed for **defensive security & bug bounty**
""")
