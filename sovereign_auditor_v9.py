import streamlit as st
import re
import math
from collections import Counter
from urllib.parse import urlparse
from datetime import datetime
import pandas as pd

# =====================================================
# Entropy Calculation
# =====================================================
def calculate_entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = Counter(value)
    entropy = 0.0
    for c in counts.values():
        p = c / len(value)
        entropy -= p * math.log2(p)
    return entropy


# =====================================================
# Platform Registry (Conservative & High Precision)
# =====================================================
PLATFORMS = {
    "AWS": {
        "prefix": r"(AKIA|ASIA)[A-Z0-9]{16}",
        "min_len": 20,
        "entropy": (3.5, 4.8),
        "require_context": False
    },
    "GITHUB": {
        "prefix": r"ghp_[A-Za-z0-9]{36}",
        "min_len": 40,
        "entropy": (3.5, 4.6),
        "require_context": False
    },
    "SLACK": {
        "prefix": r"xox[baprs]-[0-9A-Za-z\-]{10,}",
        "min_len": 20,
        "entropy": (3.0, 4.8),
        "require_context": True
    },
    "STRIPE": {
        "prefix": r"sk_live_[A-Za-z0-9]{16,}",
        "min_len": 24,
        "entropy": (3.6, 4.8),
        "require_context": False
    },
    "SENDGRID": {
        "prefix": r"SG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}",
        "min_len": 60,
        "entropy": (3.8, 4.8),
        "require_context": False
    },
    "TWILIO": {
        "prefix": r"SK[0-9a-fA-F]{32}",
        "min_len": 34,
        "entropy": (3.5, 4.8),
        "require_context": False
    },
    "MAILGUN": {
        "prefix": r"key-[0-9A-Za-z]{32}",
        "min_len": 36,
        "entropy": (3.2, 4.6),
        "require_context": False
    },
    "GCP": {
        "prefix": r"AIza[A-Za-z0-9\-_]{35}",
        "min_len": 39,
        "entropy": (3.8, 4.8),
        "require_context": False
    }
}


# =====================================================
# Core Auditor Engine V9
# =====================================================
class SovereignAuditorV9:
    def __init__(self, target_url: str):
        self.target_url = target_url
        parsed = urlparse(target_url)
        self.base_domain = ".".join(parsed.netloc.split(".")[-2:])
        self.findings = []

    def in_scope(self, url: str) -> bool:
        try:
            return urlparse(url).netloc.endswith(self.base_domain)
        except Exception:
            return False

    def classify(self, secret: str, context: str):
        entropy = calculate_entropy(secret)
        length = len(secret)

        # Reject encrypted / JWT blobs
        if entropy > 5.0:
            return None, entropy

        for platform, rule in PLATFORMS.items():
            if not re.search(rule["prefix"], secret):
                continue

            if length < rule["min_len"]:
                return None, entropy

            if not (rule["entropy"][0] <= entropy <= rule["entropy"][1]):
                return None, entropy

            if rule["require_context"]:
                if not re.search(r"(token|key|secret|auth)", context, re.I):
                    return None, entropy

            return f"{platform} API Key", entropy

        return None, entropy

    def scan_text(self, text: str, source_url: str):
        if not self.in_scope(source_url):
            return

        generic_pattern = r"(?i)(api[_-]?key|secret|token)[\"'\s:=]+([A-Za-z0-9_\-\.]{12,})"

        for match in re.finditer(generic_pattern, text):
            secret = match.group(2)
            start = max(0, match.start() - 60)
            end = min(len(text), match.end() + 60)
            context = text[start:end]

            classification, entropy = self.classify(secret, context)
            if not classification:
                continue

            self.findings.append({
                "URL": source_url,
                "Type": classification,
                "Entropy": round(entropy, 2),
                "Masked": secret[:6] + "..." + secret[-4:]
            })


# =====================================================
# Report Generator
# =====================================================
def generate_report(findings, platform: str):
    date = datetime.utcnow().strftime("%Y-%m-%d")
    output = ""

    for f in findings:
        if platform == "HackerOne":
            output += f"""
# Sensitive Data Exposure

## Summary
A high-confidence exposed API credential was identified in an in-scope resource.

## Affected Asset
{f['URL']}

## Evidence
{f['Masked']}

## Classification
{f['Type']}

## Entropy Score
{f['Entropy']}

## Impact
If valid, this credential may allow unauthorized access to third-party services.

## Remediation
Immediately revoke the exposed credential and remove it from client-side code.

## Timeline
Discovery Date: {date}

---
"""
        else:
            output += f"""
Title: Exposed API Credential

URL:
{f['URL']}

Evidence:
{f['Masked']}

Type:
{f['Type']}

Entropy:
{f['Entropy']}

Impact:
Potential unauthorized access.

---
"""
    return output.strip()


# =====================================================
# Streamlit UI
# =====================================================
st.set_page_config(page_title="Sovereign Auditor V9", layout="wide")
st.title("🛡️ Sovereign Auditor V9")

st.markdown(
    "**Conservative Secret Detection Engine**  \n"
    "Precision > Coverage • Designed for Bug Bounty reality"
)

target = st.text_input("🎯 Target URL", placeholder="https://example.com")

uploaded_file = st.file_uploader(
    "📄 Upload source file (JS / JSON / ENV / TXT)",
    type=["js", "json", "env", "txt", "html"]
)

if st.button("🚀 Run Scan"):
    if not target:
        st.error("Target URL is required.")
    elif not uploaded_file:
        st.error("Please upload a source file.")
    else:
        auditor = SovereignAuditorV9(target)
        content = uploaded_file.read().decode(errors="ignore")
        auditor.scan_text(content, target)

        if auditor.findings:
            df = pd.DataFrame(auditor.findings)
            st.success(f"High‑confidence findings: {len(df)}")
            st.dataframe(df, use_container_width=True)

            platform = st.selectbox("📤 Report Platform", ["HackerOne", "Bugcrowd"])

            if st.button("📄 Generate Report"):
                report = generate_report(auditor.findings, platform)
                st.download_button(
                    "⬇ Download report.md",
                    report,
                    file_name="report.md"
                )
                st.text_area("Copy‑Paste Report", report, height=420)
        else:
            st.info("No high‑confidence secrets detected.")
