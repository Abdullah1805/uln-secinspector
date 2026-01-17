import streamlit as st
import requests
import json
import fnmatch
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# =========================================================
# إعدادات عامة
# =========================================================
HEADERS = {"User-Agent": "Mozilla/5.0 SovereignHunter/Ultimate"}
MAX_WORKERS = 10
REQUEST_TIMEOUT = 25
MAX_DEEP_SCAN_SIZE = 500_000  # 500 KB

# =========================================================
# أدوات مساعدة
# =========================================================
def wildcard_match(value, patterns):
    if isinstance(patterns, str):
        patterns = [patterns]
    return any(fnmatch.fnmatchcase(value, p) for p in patterns)

# =========================================================
# محرك IAM (Wildcard-aware)
# =========================================================
class PolicyEngine:
    def __init__(self, policy):
        stmts = policy.get("Statement", [])
        self.statements = stmts if isinstance(stmts, list) else [stmts]

    def evaluate(self, action):
        for s in self.statements:
            if s.get("Effect") != "Allow":
                continue
            acts = s.get("Action", [])
            acts = acts if isinstance(acts, list) else [acts]
            for a in acts:
                if a == "*" or wildcard_match(action, a):
                    return "Allow"
        return "ImplicitDeny"

CAPABILITIES = [
    {"name": "Full Admin Access (*)", "actions": ["*"], "risk": 10},
    {"name": "Full IAM Administrative", "actions": ["iam:*"], "risk": 10},
    {"name": "Full S3 Management", "actions": ["s3:*"], "risk": 9},
    {"name": "Privilege Escalation (PassRole)", "actions": ["iam:PassRole"], "risk": 9},
    {"name": "User Persistence", "actions": ["iam:CreateLoginProfile"], "risk": 8},
]

def analyze_policy(policy_json):
    engine = PolicyEngine(policy_json)
    findings = []
    for cap in CAPABILITIES:
        if any(engine.evaluate(act) == "Allow" for act in cap["actions"]):
            findings.append(cap)
    return findings

# =========================================================
# محرك الأنماط الشمولية (Pattern Radar)
# =========================================================
GLOBAL_PATTERNS = {
    "Google_API_Key": r'AIza[0-9A-Za-z-_]{35}',
    "GitHub_Token": r'ghp_[a-zA-Z0-9]{36}',
    "Slack_Webhook": r'https://hooks\.slack\.com/services/T[a-zA-Z0-9_]+/B[a-zA-Z0-9_]+/[a-zA-Z0-9_]+',
    "S3_Bucket_URL": r'[a-z0-9.-]+\.s3\.amazonaws\.com',
    "Private_IP": r'\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
}

def deep_scan_content(text):
    extra = []
    for name, pattern in GLOBAL_PATTERNS.items():
        matches = set(re.findall(pattern, text))
        for m in list(matches)[:3]:
            risk = 9 if "Key" in name or "Token" in name else 6
            extra.append({"name": f"{name}: {m}", "risk": risk})
    return extra

# =========================================================
# استخراج مفاتيح AWS
# =========================================================
AWS_KEY_REGEX = r'AKIA[0-9A-Z]{16}'

def extract_credentials(text):
    keys = set(re.findall(AWS_KEY_REGEX, text))
    return [{"name": f"Leaked AWS Access Key: {k}", "risk": 10} for k in keys]

# =========================================================
# Wayback Deep Recon
# =========================================================
def fetch_interesting_urls(domain):
    api = (
        "https://web.archive.org/cdx/search/cdx"
        f"?url=*.{domain}/*&output=json"
        "&collapse=urlkey&filter=statuscode:200&limit=1000"
    )
    try:
        r = requests.get(api, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return []
        data = r.json()
        urls = []
        keywords = ['.json', '.env', '.conf', '.config', '.txt', 'aws', 'iam', 'cred', 'secret']
        for row in data[1:]:
            path = row[2].lower()
            if any(k in path for k in keywords):
                urls.append(f"https://web.archive.org/web/{row[1]}id_/{row[2]}")
        return list(set(urls))
    except:
        return []

# =========================================================
# فحص الرابط (Core Scanner)
# =========================================================
def scan_url(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None

        text = r.text
        findings = []

        # IAM Analysis
        if '"Statement"' in text:
            try:
                policy = json.loads(text)
                findings.extend(analyze_policy(policy))
            except:
                pass

        # AWS Credentials
        findings.extend(extract_credentials(text))

        # Deep Pattern Scan (مشروط بالحجم)
        if len(text.encode()) <= MAX_DEEP_SCAN_SIZE:
            findings.extend(deep_scan_content(text))

        if findings:
            return {"url": url, "findings": findings}

    except:
        return None

    return None

# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(page_title="Sovereign Hunter Ultimate", layout="wide")
st.title("🛡️ Sovereign Hunter — Ultimate Recon & Leak Detector")

domains_input = st.text_area(
    "أدخل النطاقات (كل نطاق في سطر):",
    "tesla.com\nstarlink.com\nadobe.com",
    height=120
)

if st.button("🚀 Start Ultimate Scan"):
    domains = [d.strip() for d in domains_input.splitlines() if d.strip()]
    all_results = []
    report = []

    for domain in domains:
        st.subheader(f"🔍 Scanning {domain}")
        urls = fetch_interesting_urls(domain)
        st.write(f"📁 Found {len(urls)} candidate files")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
            futures = [exe.submit(scan_url, u) for u in urls]
            for f in as_completed(futures):
                res = f.result()
                if res:
                    all_results.append(res)

        if all_results:
            for r in all_results:
                with st.expander(f"⚠️ {r['url']}"):
                    for f in r["findings"]:
                        st.write(f"**{f['name']}** | Risk: {f['risk']}/10")
                        report.append({
                            "domain": domain,
                            "url": r["url"],
                            "finding": f["name"],
                            "risk": f["risk"]
                        })
        else:
            st.success("No critical findings in this domain")

    # =====================================================
    # التقرير الاحترافي
    # =====================================================
    st.markdown("---")
    st.header("📄 Professional Security Report")

    if report:
        report_json = {
            "generated_at": datetime.utcnow().isoformat(),
            "total_findings": len(report),
            "findings": report
        }

        st.download_button(
            "⬇️ Download Report (JSON)",
            data=json.dumps(report_json, indent=2),
            file_name="sovereign_hunter_report.json",
            mime="application/json"
        )

        st.success("🎯 Critical findings detected — review before disclosure")
    else:
        st.info("No high-risk patterns detected")
