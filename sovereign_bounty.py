import streamlit as st
import requests
import re
import math
from collections import Counter
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

# =========================
# CONFIG
# =========================
HEADERS = {"User-Agent": "SovereignBounty/1.0"}
TIMEOUT = 8
MAX_JS = 40
SKIP_EXT = (".png", ".jpg", ".svg", ".woff", ".woff2", ".gif")

# =========================
# ENTROPY
# =========================
def entropy(s):
    if not s or len(s) < 12:
        return 0
    freq = Counter(s)
    l = len(s)
    return -sum((c/l) * math.log2(c/l) for c in freq.values())

# =========================
# PLATFORM RULES (STRICT)
# =========================
PLATFORMS = {
    "AWS": {
        "regex": r"AKIA[0-9A-Z]{16}",
        "entropy": (3.2, 4.6)
    },
    "Stripe": {
        "regex": r"sk_live_[A-Za-z0-9]{24,}",
        "entropy": (3.6, 4.8)
    },
    "GitHub": {
        "regex": r"ghp_[A-Za-z0-9]{36}",
        "entropy": (3.5, 4.6)
    },
    "Slack": {
        "regex": r"xox[baprs]-[0-9A-Za-z]{10,}",
        "entropy": (3.0, 4.8)
    },
    "SendGrid": {
        "regex": r"SG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}",
        "entropy": (3.8, 4.8)
    }
}

# =========================
# SCOPE CHECK
# =========================
def in_scope(base, url):
    try:
        return urlparse(url).netloc.endswith(base)
    except:
        return False

# =========================
# SCAN TEXT
# =========================
def scan_text(text, source):
    findings = []
    for name, rule in PLATFORMS.items():
        for m in re.finditer(rule["regex"], text):
            secret = m.group()
            e = entropy(secret)
            if rule["entropy"][0] <= e <= rule["entropy"][1]:
                findings.append({
                    "Platform": name,
                    "Masked": secret[:6] + "..." + secret[-4:],
                    "Entropy": round(e, 2),
                    "Source": source
                })
    return findings

# =========================
# MAIN SCAN
# =========================
def scan_target(url):
    parsed = urlparse(url)
    base_domain = ".".join(parsed.netloc.split(".")[-2:])
    results = []

    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    soup = BeautifulSoup(r.text, "html.parser")

    scripts = [urljoin(url, s["src"]) for s in soup.find_all("script", src=True)]
    scripts = scripts[:MAX_JS]

    for js in scripts:
        if js.endswith(SKIP_EXT):
            continue
        if not in_scope(base_domain, js):
            continue
        try:
            jsr = requests.get(js, headers=HEADERS, timeout=TIMEOUT)
            results.extend(scan_text(jsr.text, js))
        except:
            pass

    return results

# =========================
# REPORT
# =========================
def generate_report(findings):
    date = datetime.utcnow().strftime("%Y-%m-%d")
    out = ""
    for f in findings:
        out += f"""
# Exposed Secret in Client-Side Code

**Platform:** {f['Platform']}
**Source:** {f['Source']}
**Evidence:** `{f['Masked']}`
**Entropy:** {f['Entropy']}

**Impact:**
If valid, this credential may allow unauthorized access.

**Remediation:**
Revoke immediately and remove from client-side assets.

**Discovered:** {date}

---
"""
    return out

# =========================
# STREAMLIT UI
# =========================
st.set_page_config("Sovereign Bounty", layout="wide")
st.title("👑 Sovereign Bounty Scanner")

target = st.text_input("Target URL", placeholder="https://example.com")

if st.button("🚀 Scan"):
    if not target:
        st.error("Enter a target URL.")
    else:
        with st.spinner("Recon & Analysis..."):
            findings = scan_target(target)

        if findings:
            df = pd.DataFrame(findings)
            st.success(f"High-confidence findings: {len(df)}")
            st.dataframe(df)

            report = generate_report(findings)
            st.download_button("📄 Download Report", report, "report.md")
            st.text_area("Copy-Paste (HackerOne/Bugcrowd)", report, height=400)
        else:
            st.info("No high-confidence secrets found.")
