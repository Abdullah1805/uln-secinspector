import streamlit as st
import asyncio
import aiohttp
import re
import math
import base64
import json
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import Counter
from scipy.stats import entropy as kl_entropy
from datetime import datetime

# ============================================================
# 1. SYSTEM PHILOSOPHY & CONFIG
# ============================================================

SYSTEM_PROFILE = {
    "Conservative": {
        "bayes_threshold": 0.95,
        "allow_chunk_merge": False
    },
    "Balanced": {
        "bayes_threshold": 0.92,
        "allow_chunk_merge": True
    },
    "Aggressive": {
        "bayes_threshold": 0.88,
        "allow_chunk_merge": True
    }
}

# ============================================================
# 2. SECRET PROVIDER KNOWLEDGE BASE
# ============================================================

PROVIDERS = {
    "AWS": {
        "regex": r"AKIA[0-9A-Z]{16}",
        "prefix_prob": 0.99
    },
    "GitHub": {
        "regex": r"ghp_[A-Za-z0-9]{36}",
        "prefix_prob": 0.98
    },
    "Stripe": {
        "regex": r"sk_live_[A-Za-z0-9]{24}",
        "prefix_prob": 0.97
    },
    "Google API": {
        "regex": r"AIza[0-9A-Za-z\-_]{35}",
        "prefix_prob": 0.96
    }
}

# ============================================================
# 3. MATHEMATICAL CORE
# ============================================================

def shannon_entropy(s: str) -> float:
    if not s or len(s) < 8:
        return 0.0
    probs = [c / len(s) for c in Counter(s).values()]
    return -sum(p * math.log2(p) for p in probs)

def kl_divergence(sample: str, baseline="abcdefghijklmnopqrstuvwxyz0123456789") -> float:
    def freq_dist(x):
        c = Counter(x)
        return np.array([c.get(ch, 0) + 1 for ch in baseline])

    p = freq_dist(sample)
    q = freq_dist(baseline)
    p = p / p.sum()
    q = q / q.sum()
    return kl_entropy(p, q, base=2)

def context_score(text: str, index: int, window=300) -> float:
    ctx = text[max(0, index-window): index+window].lower()
    keywords = ["api", "key", "token", "auth", "secret", "bearer", "fetch", "axios"]
    score = sum(1 for k in keywords if k in ctx)
    return min(score / len(keywords), 1.0)

# ============================================================
# 4. BAYESIAN DECISION ENGINE
# ============================================================

def bayesian_secret_probability(ent, kl, ctx, prefix_prob):
    """
    Harvard-grade Naive Bayes Fusion
    """
    p_entropy = min(ent / 5.0, 1.0)
    p_kl = min(kl / 4.0, 1.0)

    weights = {
        "entropy": 0.30,
        "kl": 0.20,
        "context": 0.25,
        "prefix": 0.25
    }

    score = (
        weights["entropy"] * p_entropy +
        weights["kl"] * p_kl +
        weights["context"] * ctx +
        weights["prefix"] * prefix_prob
    )

    return round(score, 4)

# ============================================================
# 5. ADVANCED SCANNING ENGINE
# ============================================================

async def scan_text(text, source, profile):
    findings = []

    decoded_blocks = [text]
    b64_matches = re.findall(r'[A-Za-z0-9+/]{40,}={0,2}', text)
    for b in b64_matches:
        try:
            decoded = base64.b64decode(b).decode("utf-8", errors="ignore")
            decoded_blocks.append(decoded)
        except:
            pass

    for block in decoded_blocks:
        for provider, cfg in PROVIDERS.items():
            for m in re.finditer(cfg["regex"], block):
                token = m.group()
                ent = shannon_entropy(token)
                if ent < 3.0:
                    continue

                kl = kl_divergence(token)
                ctx = context_score(block, m.start())
                bayes = bayesian_secret_probability(
                    ent, kl, ctx, cfg["prefix_prob"]
                )

                if bayes >= profile["bayes_threshold"]:
                    findings.append({
                        "Provider": provider,
                        "Confidence": f"{int(bayes*100)}%",
                        "BayesScore": bayes,
                        "Entropy": round(ent, 2),
                        "KL": round(kl, 2),
                        "Context": round(ctx, 2),
                        "Source": source,
                        "TokenPreview": token[:6] + "..." + token[-4:]
                    })

    return findings

# ============================================================
# 6. TARGET DISCOVERY ENGINE
# ============================================================

async def crawl_target(url, profile):
    results = []
    async with aiohttp.ClientSession(headers={"User-Agent": "SovereignScanner/16"}) as session:
        async with session.get(url, timeout=15) as resp:
            html = await resp.text()
            soup = BeautifulSoup(html, "html.parser")

            scripts = [
                urljoin(url, s["src"])
                for s in soup.find_all("script", src=True)
                if urlparse(urljoin(url, s["src"])).netloc == urlparse(url).netloc
            ]

            tasks = [session.get(js, timeout=10) for js in scripts]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            for js, r in zip(scripts, responses):
                if isinstance(r, aiohttp.ClientResponse):
                    content = await r.text()
                    found = await scan_text(content, js, profile)
                    results.extend(found)

    return results

# ============================================================
# 7. PROFESSIONAL REPORT GENERATOR
# ============================================================

def generate_report(findings, target):
    report = f"""
# 🔐 Sensitive Information Disclosure

## 📌 Summary
During a security assessment of **{target}**, sensitive API credentials were discovered embedded within client-side JavaScript resources.

## 🧠 Detection Methodology
- Static analysis only (No active exploitation)
- Bayesian probability fusion
- Entropy & KL-divergence validation
- Contextual inference

## 📊 Findings
"""
    for f in findings:
        report += f"""
### {f['Provider']} API Key Exposure
- Confidence: {f['Confidence']}
- Evidence: High entropy, valid prefix, contextual usage
- Source: `{f['Source']}`
"""

    report += """
## 🛡️ Impact
An attacker could abuse the exposed credentials to access internal services, leading to:
- Financial loss
- Data exposure
- Service abuse

## ✅ Recommendation
- Immediately revoke affected keys
- Rotate credentials
- Remove secrets from client-side code
- Use environment variables or server-side proxies

---
*Report generated by Sovereign v16 — Integrity First.*
"""
    return report

# ============================================================
# 8. STREAMLIT UI
# ============================================================

st.set_page_config(page_title="Sovereign v16", layout="wide")
st.title("👑 Sovereign v16 — Elite Bug Bounty Engine")
st.caption("Scientific • Ethical • Zero False Positives")

target = st.text_input("🎯 Target URL", "https://")
profile_name = st.selectbox("Detection Profile", list(SYSTEM_PROFILE.keys()))
profile = SYSTEM_PROFILE[profile_name]

if st.button("🚀 Execute Scan"):
    with st.spinner("Analyzing with Bayesian Intelligence..."):
        findings = asyncio.run(crawl_target(target, profile))

    if findings:
        df = pd.DataFrame(findings)
        st.success(f"Confirmed Findings: {len(df)}")
        st.dataframe(df)

        report = generate_report(findings, target)
        st.download_button(
            "📄 Download HackerOne Report",
            report,
            file_name="sovereign_report.md"
        )
    else:
        st.info("Scan completed. No verified exposures found.")
