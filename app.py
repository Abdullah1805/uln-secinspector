import streamlit as st
import requests
import json
import fnmatch
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# =========================================================
# إعدادات عامة
# =========================================================
HEADERS = {
    "User-Agent": "Mozilla/5.0 SovereignHunter/Ultimate"
}

MAX_WORKERS = 12
REQUEST_TIMEOUT = 25

# =========================================================
# أدوات مساعدة
# =========================================================
def wildcard_match(value, patterns):
    if isinstance(patterns, str):
        patterns = [patterns]
    return any(fnmatch.fnmatchcase(value, p) for p in patterns)

# =========================================================
# محرك IAM مبسط (Wildcard-aware)
# =========================================================
class PolicyEngine:
    def __init__(self, policy):
        self.statements = policy.get("Statement", [])
        if isinstance(self.statements, dict):
            self.statements = [self.statements]

    def evaluate(self, action):
        for s in self.statements:
            if s.get("Effect") != "Allow":
                continue

            acts = s.get("Action", [])
            if isinstance(acts, str):
                acts = [acts]

            for a in acts:
                if a == "*" or wildcard_match(action, a):
                    return "Allow"

        return "ImplicitDeny"

# =========================================================
# كشف القدرات الخطيرة (Golden Findings)
# =========================================================
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
# Deep Recon – Wayback Machine (محسّن + Subdomains)
# =========================================================
def fetch_interesting_urls(domain):
    try:
        api = (
            "https://web.archive.org/cdx/search/cdx"
            f"?url=*.{domain}/*"
            "&output=json"
            "&collapse=urlkey"
            "&filter=statuscode:200"
            "&limit=1000"
        )

        r = requests.get(api, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return []

        data = r.json()
        if len(data) <= 1:
            return []

        interesting_keywords = [
            ".json", ".env", ".conf", ".config", ".txt",
            "iam", "aws", "policy", "cred", "secret"
        ]

        urls = []
        for row in data[1:]:
            original = row[2]
            lower = original.lower()
            if any(k in lower for k in interesting_keywords):
                ts = row[1]
                urls.append(f"https://web.archive.org/web/{ts}id_/{original}")

        return list(set(urls))

    except Exception:
        return []

# =========================================================
# تحميل وتحليل الملفات (Multi-thread)
# =========================================================
def scan_url(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None

        text = r.text
        if '"Statement"' not in text:
            return None

        policy = json.loads(text)
        findings = analyze_policy(policy)

        if findings:
            return {
                "url": url,
                "findings": findings
            }

    except Exception:
        return None

    return None

# =========================================================
# واجهة Streamlit
# =========================================================
st.set_page_config(page_title="Sovereign IAM Hunter Ultimate", layout="wide")

st.title("🛡️ Sovereign IAM Hunter — Ultimate Edition")
st.markdown("### Mass Scanning + Deep Recon + Wildcard Detection")

domains_input = st.text_area(
    "أدخل النطاقات (كل نطاق في سطر):",
    value="tesla.com\nstarlink.com\nadobe.com",
    height=120
)

if st.button("🚀 Start Mass Scan"):
    domains = [d.strip() for d in domains_input.splitlines() if d.strip()]
    st.info(f"بدء الفحص لـ {len(domains)} نطاقات")

    all_results = []

    for domain in domains:
        st.subheader(f"🔍 Recon: {domain}")
        urls = fetch_interesting_urls(domain)
        st.write(f"📁 تم العثور على {len(urls)} ملفات محتملة")

        results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(scan_url, u) for u in urls]

            for future in as_completed(futures):
                res = future.result()
                if res:
                    results.append(res)

        if results:
            st.error(f"🔥 تم اكتشاف {len(results)} نتائج خطيرة")
            for r in results:
                with st.expander(f"⚠️ {r['url']}"):
                    for f in r["findings"]:
                        st.write(f"**{f['name']}** | Risk: {f['risk']}/10")
        else:
            st.success("لم يتم العثور على سياسات خطيرة في هذا النطاق")

        all_results.extend(results)

    st.markdown("---")
    st.header("📊 الخلاصة النهائية")
    st.write(f"إجمالي النتائج الخطيرة: **{len(all_results)}**")

    if all_results:
        st.success("🎯 هذا صيد حقيقي — راجع النتائج بعناية قبل أي بلاغ")

st.sidebar.markdown("""
### 💡 ملاحظات احترافية
- الأداة تبحث فقط في **بيانات مؤرشفة عامة**
- ركّز على النتائج ذات Risk 9–10
- دائماً تحقق يدوياً قبل الإبلاغ
- الأفضل تشغيلها على نطاقات Bug Bounty فقط
""")
