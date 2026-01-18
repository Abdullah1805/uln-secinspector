import streamlit as st
import re
import math
from collections import Counter
from urllib.parse import urlparse
import pandas as pd
from datetime import datetime

# =====================================================
# Utilities
# =====================================================
def calculate_entropy(data: str) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    entropy = 0.0
    for c in counts.values():
        p = c / len(data)
        entropy -= p * math.log2(p)
    return entropy


# =====================================================
# Platform Rules (Conservative)
# =====================================================
PLATFORM_RULES = {
    "AWS": {
        "min_len": 12,
        "entropy": 3.2,
        "prefix": r"(AKIA|ASIA)[A-Z0-9]{8,}"
    },
    "STRIPE": {
        "min_len": 24,
        "entropy": 3.6,
        "prefix": r"sk_live_[A-Za-z0-9]{16,}"
    },
    "GCP": {
        "min_len": 32,
        "entropy": 4.0,
        "prefix": r"AIza[A-Za-z0-9\-_]{20,}"
    },
    "DEFAULT": {
        "min_len": 16,
        "entropy": 3.8,
        "prefix": None
    }
}


# =====================================================
# Auditor Engine V9
# =====================================================
class SovereignAuditorV9:
    def __init__(self, target_url, enable_chunked=False):
        self.target_url = target_url.lower()
        parsed = urlparse(self.target_url)
        self.base_domain = ".".join(parsed.netloc.split(".")[-2:])
        self.enable_chunked = enable_chunked
        self.findings = []

    # -------------------------
    # Scope Validation
    # -------------------------
    def is_in_scope(self, url):
        try:
            netloc = urlparse(url).netloc.lower()
            return netloc.endswith(self.base_domain)
        except:
            return False

    # -------------------------
    # Classification
    # -------------------------
    def classify_secret(self, secret: str):
        entropy = calculate_entropy(secret)
        length = len(secret)

        # High entropy bomb (JWT / encrypted)
        if entropy > 5.0:
            return None, entropy

        for name, rule in PLATFORM_RULES.items():
            if rule["prefix"] and re.search(rule["prefix"], secret):
                if length >= rule["min_len"] and entropy >= rule["entropy"]:
                    return f"{name} API Key", entropy
                return None, entropy

        # Generic fallback
        if length >= PLATFORM_RULES["DEFAULT"]["min_len"] and entropy >= PLATFORM_RULES["DEFAULT"]["entropy"]:
            return "Generic High-Entropy Secret", entropy

        return None, entropy

    # -------------------------
    # Scan Text
    # -------------------------
    def scan_text(self, text, source_url):
        if not self.is_in_scope(source_url):
            return

        pattern = r"(?i)(api[_-]?key|secret|token)[\"'\s:=]+([A-Za-z0-9_\-\.]{12,})"

        for match in re.finditer(pattern, text):
            secret = match.group(2)
            classification, entropy = self.classify_secret(secret)

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
def generate_report(findings, platform):
    date = datetime.utcnow().strftime("%Y-%m-%d")
    report = ""

    for f in findings:
        if platform == "HackerOne":
            report += f"""
# Sensitive Data Exposure

## Summary
A high-confidence secret was discovered in an in-scope client-side resource.

## Asset
{f['URL']}

## Evidence
{f['Masked']}
## Classification
{f['Type']}

## Entropy
{f['Entropy']}

## Impact
Potential unauthorized access if the credential is active.

## Remediation
Revoke the exposed secret and remove it from client-side code.

## Timeline
Discovery Date: {date}

---
"""
        else:
            report += f"""
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
Potential unauthorized usage.

---
"""
    return report


# =====================================================
# Streamlit UI
# =====================================================
st.set_page_config(page_title="Sovereign Auditor V9", layout="wide")
st.title("🛡️ Sovereign Auditor V9")

target = st.text_input("🎯 Target URL", placeholder="https://example.com")

enable_chunked = st.checkbox(
    "Enable Chunked-Secret Reconstruction (OFF by default)",
    value=False
)

if st.button("🚀 Run Scan"):
    if not target:
        st.error("Target URL required.")
    else:
        auditor = SovereignAuditorV9(target, enable_chunked)

        # ---- Simulation Input ----
        simulated_content = """
            const stripe = "sk_live_51MzX9abcdEFGH123456789";
            const aws = "AKIAIOSFODNN7EXAMPLE";
            const fake = "thisIsJustALongReadableString123456";
        """

        auditor.scan_text(simulated_content, target)

        if auditor.findings:
            df = pd.DataFrame(auditor.findings)
            st.success(f"Findings detected: {len(df)}")
            st.dataframe(df)

            platform = st.selectbox("Report Platform", ["HackerOne", "Bugcrowd"])

            if st.button("📄 Generate Report"):
                report = generate_report(auditor.findings, platform)
                st.download_button(
                    "⬇ Download Report.md",
                    report,
                    file_name="report.md"
                )
                st.text_area("Copy-Paste Version", report, height=400)
        else:
            st.info("No high-confidence findings detected.")
