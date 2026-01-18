import streamlit as st
import re
import math
from collections import Counter
from urllib.parse import urlparse
import pandas as pd
from datetime import datetime

# =====================================================
# Entropy
# =====================================================
def calculate_entropy(data: str) -> float:
    if not data or len(data) < 12:
        return 0.0
    counts = Counter(data)
    entropy = 0.0
    for c in counts.values():
        p = c / len(data)
        entropy -= p * math.log2(p)
    return entropy


# =====================================================
# Conservative Rules
# =====================================================
PLATFORMS = {
    "AWS": {
        "prefix": r"(AKIA|ASIA)[A-Z0-9]{12,}",
        "min_len": 20,
        "entropy": (3.2, 4.5),
        "require_context": True
    },
    "STRIPE": {
        "prefix": r"sk_live_[A-Za-z0-9]{16,}",
        "min_len": 24,
        "entropy": (3.5, 4.5),
        "require_context": False
    },
    "GCP": {
        "prefix": r"AIza[A-Za-z0-9\-_]{20,}",
        "min_len": 32,
        "entropy": (3.8, 4.6),
        "require_context": False
    }
}

DOC_KEYWORDS = [
    "example", "sample", "dummy", "test", "placeholder", "demo"
]

CONTEXT_REGEX = r"(?i)(api[_-]?key|secret|token|auth)"


# =====================================================
# Auditor Engine V9.2 (Conservative)
# =====================================================
class SovereignAuditorV9_2:
    def __init__(self, target_url):
        self.target_url = target_url.lower()
        parsed = urlparse(self.target_url)
        self.base_domain = ".".join(parsed.netloc.split(".")[-2:])
        self.findings = []

    def is_in_scope(self, url):
        try:
            netloc = urlparse(url).netloc.lower()
            return netloc.endswith(self.base_domain)
        except:
            return False

    def score_secret(self, secret, context, source_url):
        entropy = calculate_entropy(secret)
        length = len(secret)

        # Encrypted / JWT / Noise
        if entropy > 5.0:
            return None

        # Documentation trap
        lowered = context.lower()
        if any(k in lowered for k in DOC_KEYWORDS):
            return None

        for name, rule in PLATFORMS.items():
            if not re.search(rule["prefix"], secret):
                continue

            score = 0

            # Prefix
            score += 40

            # Length
            if length >= rule["min_len"]:
                score += 15
            else:
                continue

            # Entropy range
            if rule["entropy"][0] <= entropy <= rule["entropy"][1]:
                score += 25
            else:
                continue

            # Context
            has_context = re.search(CONTEXT_REGEX, context)
            if has_context:
                score += 15
            elif rule["require_context"]:
                continue  # strict AWS rule

            # Scope
            if self.is_in_scope(source_url):
                score += 10
            else:
                continue

            if score >= 85:
                return {
                    "URL": source_url,
                    "Type": f"{name} API Key",
                    "Entropy": round(entropy, 2),
                    "Masked": secret[:6] + "..." + secret[-4:],
                    "Score": score
                }

        return None

    def scan_text(self, text, source_url):
        if not self.is_in_scope(source_url):
            return

        pattern = r"([A-Za-z0-9_\-\.]{16,})"

        for match in re.finditer(pattern, text):
            secret = match.group(1)
            context_window = text[max(0, match.start()-40):match.end()+40]

            finding = self.score_secret(secret, context_window, source_url)
            if finding:
                self.findings.append(finding)


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
A high-confidence exposed API credential was identified in an in-scope asset.

## Asset
{f['URL']}

## Evidence
{f['Masked']}

## Classification
{f['Type']}

## Entropy
{f['Entropy']}

## Confidence Score
{f['Score']}/100

## Impact
If active, this credential could allow unauthorized access.

## Remediation
Immediately revoke the exposed key and remove it from client-side code.

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

Confidence Score:
{f['Score']}/100

Impact:
Potential unauthorized usage.

---
"""
    return report


# =====================================================
# Streamlit UI
# =====================================================
st.set_page_config(page_title="Sovereign Auditor V9.2", layout="wide")
st.title("🛡️ Sovereign Auditor V9.2 — Conservative Edition")

target = st.text_input("🎯 Target URL", placeholder="https://example.com")

if st.button("🚀 Run Conservative Scan"):
    if not target:
        st.error("Target URL required.")
    else:
        auditor = SovereignAuditorV9_2(target)

        # Aggressive Simulation Content
        simulated = """
        // Docs example (should be ignored)
        const apiKey = "AKIAIOSFODNN7EXAMPLE";

        // Real Stripe
        const stripeKey = "sk_live_51MzX9abcdEFGH123456789";

        // GCP bundle
        var a="AIzaSyD8f9EabcXYZ1234567890";
        """

        auditor.scan_text(simulated, target)

        if auditor.findings:
            df = pd.DataFrame(auditor.findings)
            st.success(f"High-confidence findings: {len(df)}")
            st.dataframe(df)

            platform = st.selectbox("Report Platform", ["HackerOne", "Bugcrowd"])
            if st.button("📄 Generate Report"):
                report = generate_report(auditor.findings, platform)
                st.download_button("⬇ Download Report.md", report, file_name="report.md")
                st.text_area("Copy Report", report, height=400)
        else:
            st.info("No high-confidence findings detected (as expected in conservative mode).")
