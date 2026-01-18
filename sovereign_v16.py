import streamlit as st
import aiohttp
import asyncio
import re
import math
import base64
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import Counter, defaultdict
from datetime import datetime
import itertools
import hashlib

# =========================================================
# 0. GLOBAL PHILOSOPHY
# =========================================================
# - Zero False Positives > Recall
# - Silence is better than wrong report
# - Math before Regex
# - Scope before Scan
# - Everything Toggle-able

# =========================================================
# 1. CONFIGURATION
# =========================================================
USER_AGENT = "Sovereign/16.0 (Scientific Bug Bounty Engine)"
HEADERS = {"User-Agent": USER_AGENT}
REQUEST_TIMEOUT = 10
MAX_JS_FILES = 60
MAX_TEXT_SIZE = 1_500_000  # 1.5MB safety cap
SKIP_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".woff", ".woff2", ".ttf", ".eot", ".mp4", ".mp3")

# =========================================================
# 2. PLATFORM REGISTRY (HARD PREFIX + ENTROPY WINDOWS)
# =========================================================
PLATFORMS = {
    "AWS": {
        "regex": r"(AKIA|ASIA)[0-9A-Z]{16}",
        "min_len": 20,
        "entropy": (3.1, 4.5),
        "context_required": False,
        "verify": False
    },
    "GITHUB": {
        "regex": r"ghp_[A-Za-z0-9]{36}",
        "min_len": 40,
        "entropy": (3.4, 4.7),
        "context_required": False,
        "verify": False
    },
    "STRIPE": {
        "regex": r"sk_live_[A-Za-z0-9]{24,}",
        "min_len": 32,
        "entropy": (3.5, 4.8),
        "context_required": False,
        "verify": False
    },
    "SLACK": {
        "regex": r"xox[baprs]-[0-9A-Za-z-]{10,}",
        "min_len": 20,
        "entropy": (3.0, 4.9),
        "context_required": True,
        "verify": False
    },
    "SENDGRID": {
        "regex": r"SG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}",
        "min_len": 60,
        "entropy": (3.7, 4.9),
        "context_required": False,
        "verify": False
    },
}

# =========================================================
# 3. MATHEMATICAL CORE
# =========================================================
def shannon_entropy(s: str) -> float:
    if not s or len(s) < 12:
        return 0.0
    freq = Counter(s)
    n = len(s)
    return -sum((c/n) * math.log2(c/n) for c in freq.values())

def normalized_entropy(s: str) -> float:
    if not s:
        return 0.0
    return shannon_entropy(s) / math.log2(len(set(s)) + 1)

def looks_like_binary_noise(s: str) -> bool:
    # extremely high entropy often means encrypted/binary blob
    return shannon_entropy(s) > 5.2

# =========================================================
# 4. BASE64 RECURSIVE DECODER (SAFE)
# =========================================================
def recursive_base64_decode(text: str, depth=2):
    results = {text}
    pattern = r'(?:[A-Za-z0-9+/]{40,}={0,2})'
    for _ in range(depth):
        new = set()
        for block in results:
            for b64 in re.findall(pattern, block):
                try:
                    decoded = base64.b64decode(b64).decode('utf-8', errors='ignore')
                    if 20 < len(decoded) < 5000:
                        new.add(decoded)
                except:
                    pass
        results |= new
    return results

# =========================================================
# 5. CHUNKED SECRET RECONSTRUCTION (OPTIONAL / OFF)
# =========================================================
def reconstruct_chunks(strings, max_join=3):
    # Conservative: limited combinations to avoid O(n^2) explosion
    results = set()
    for r in range(2, max_join + 1):
        for combo in itertools.combinations(strings, r):
            joined = "".join(combo)
            if 16 < len(joined) < 120:
                results.add(joined)
    return results

# =========================================================
# 6. SCOPE VALIDATION
# =========================================================
def extract_base_domain(url):
    p = urlparse(url)
    parts = p.netloc.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else p.netloc

def in_scope(url, base_domain):
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc == base_domain or netloc.endswith("." + base_domain)
    except:
        return False

# =========================================================
# 7. CLASSIFICATION ENGINE
# =========================================================
def classify_secret(secret: str, context: str):
    ent = shannon_entropy(secret)
    if looks_like_binary_noise(secret):
        return None

    for name, cfg in PLATFORMS.items():
        if not re.search(cfg["regex"], secret):
            continue

        if len(secret) < cfg["min_len"]:
            continue

        if not (cfg["entropy"][0] <= ent <= cfg["entropy"][1]):
            continue

        if cfg["context_required"]:
            if not re.search(r"(token|auth|secret|key)", context, re.I):
                continue

        return {
            "platform": name,
            "entropy": round(ent, 2),
            "confidence": "High (Math + Prefix)",
        }

    return None

# =========================================================
# 8. SCAN TEXT UNIT
# =========================================================
def scan_text(text, source_url, enable_chunked):
    findings = []
    base_blocks = recursive_base64_decode(text)

    # Extract quoted strings for chunk logic
    quoted = re.findall(r"['\"]([^'\"]{6,80})['\"]", text)
    if enable_chunked and quoted:
        base_blocks |= reconstruct_chunks(quoted)

    for block in base_blocks:
        for match in re.finditer(r"[A-Za-z0-9_\-\.]{16,}", block):
            secret = match.group()
            start = max(0, match.start() - 150)
            end = min(len(block), match.end() + 150)
            context = block[start:end]

            verdict = classify_secret(secret, context)
            if verdict:
                findings.append({
                    "URL": source_url,
                    "Platform": verdict["platform"],
                    "Entropy": verdict["entropy"],
                    "Masked": secret[:6] + "..." + secret[-4:],
                    "Confidence": verdict["confidence"]
                })
    return findings

# =========================================================
# 9. CRAWLER (HTML + JS AUTO DISCOVERY)
# =========================================================
async def fetch(session, url):
    try:
        async with session.get(url, timeout=REQUEST_TIMEOUT) as r:
            if r.status == 200:
                return await r.text()
    except:
        return None
    return None

async def crawl_and_scan(target, enable_chunked):
    base_domain = extract_base_domain(target)
    results = []

    async with aiohttp.ClientSession(headers=HEADERS) as session:
        html = await fetch(session, target)
        if not html:
            return results

        results.extend(scan_text(html[:MAX_TEXT_SIZE], target, enable_chunked))

        soup = BeautifulSoup(html, "html.parser")
        scripts = []
        for s in soup.find_all("script", src=True):
            js = urljoin(target, s["src"])
            if in_scope(js, base_domain) and not js.lower().endswith(SKIP_EXTENSIONS):
                scripts.append(js)

        for js in scripts[:MAX_JS_FILES]:
            js_text = await fetch(session, js)
            if js_text:
                results.extend(scan_text(js_text[:MAX_TEXT_SIZE], js, enable_chunked))

    # Deduplicate by hash
    seen = set()
    clean = []
    for f in results:
        h = hashlib.sha256((f["URL"] + f["Masked"]).encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            clean.append(f)

    return clean

# =========================================================
# 10. REPORT GENERATOR (HackerOne / Bugcrowd)
# =========================================================
def generate_report(findings, platform):
    date = datetime.utcnow().strftime("%Y-%m-%d")
    report = ""
    for f in findings:
        if platform == "HackerOne":
            report += f"""
# Sensitive Information Disclosure

## Summary
A high-confidence exposed credential was identified in client-side resources.

## Affected Asset
{f['URL']}

## Evidence
{f['Masked']}
## Classification
{f['Platform']} API Credential

## Entropy
{f['Entropy']}

## Impact
An attacker could potentially abuse this credential to access protected services.

## Recommendation
Immediately revoke the exposed credential and remove it from client-side code.

## Discovery Date
{date}

---
"""
        else:
            report += f"""
Title: Exposed API Key ({f['Platform']})

URL:
{f['URL']}

Evidence:
{f['Masked']}

Entropy:
{f['Entropy']}

Impact:
Potential unauthorized access.

---
"""
    return report

# =========================================================
# 11. STREAMLIT UI
# =========================================================
st.set_page_config(page_title="👑 Sovereign v16", layout="wide")
st.title("👑 Sovereign v16 — Scientific Bug Bounty Engine")
st.caption("Conservative • Mathematical • Zero False Positives")

target = st.text_input("🎯 Target URL (just the site)", placeholder="https://example.com")

col1, col2 = st.columns(2)
with col1:
    enable_chunked = st.checkbox("Enable Chunked-Secret Reconstruction (OFF recommended)", value=False)
with col2:
    report_platform = st.selectbox("Report Format", ["HackerOne", "Bugcrowd"])

if st.button("🚀 Run Scientific Scan"):
    if not target:
        st.error("Target required.")
    else:
        with st.spinner("Scanning conservatively..."):
            findings = asyncio.run(crawl_and_scan(target, enable_chunked))

        if findings:
            df = pd.DataFrame(findings)
            st.success(f"High-confidence findings: {len(df)}")
            st.dataframe(df)

            report = generate_report(findings, report_platform)
            st.download_button("📄 Download Report", report, file_name="report.md")
            st.text_area("Copy / Review", report, height=400)
        else:
            st.info("Scan finished: No high-confidence leaks found. Integrity preserved.")
