import streamlit as st
import aiohttp, asyncio, re, math, json, base64
from collections import Counter
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import os

# =====================================================
# CONFIG
# =====================================================
USER_AGENT = "SovereignScanner/Final"
ENTROPY_MIN = 3.2
ENTROPY_MAX = 4.9
BAYES_THRESHOLD = 0.92
LEARNING_RATE = 0.15
KNOWLEDGE_FILE = "sovereign_knowledge.json"

SKIP_EXT = (".png",".jpg",".jpeg",".gif",".svg",".woff",".woff2",".ttf",".pdf")

# =====================================================
# PROVIDER REGISTRY (50+ READY, sample shown)
# =====================================================
PROVIDERS = {
    "AWS":      {"regex": r"AKIA[0-9A-Z]{16}", "prefix_prob": 0.99},
    "GitHub":   {"regex": r"ghp_[A-Za-z0-9]{36}", "prefix_prob": 0.98},
    "Stripe":   {"regex": r"sk_live_[A-Za-z0-9]{24,}", "prefix_prob": 0.97},
    "Google":   {"regex": r"AIza[0-9A-Za-z\-_]{35}", "prefix_prob": 0.96},
    "Slack":    {"regex": r"xox[baprs]-[0-9A-Za-z\-]{10,}", "prefix_prob": 0.95},
    "Twilio":   {"regex": r"SK[0-9a-fA-F]{32}", "prefix_prob": 0.95},
    "SendGrid": {"regex": r"SG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}", "prefix_prob": 0.96},
}

# =====================================================
# UTILITIES
# =====================================================
def entropy(s):
    if not s or len(s) < 12:
        return 0
    freq = Counter(s)
    return -sum((c/len(s))*math.log2(c/len(s)) for c in freq.values())

def load_knowledge():
    if os.path.exists(KNOWLEDGE_FILE):
        return json.load(open(KNOWLEDGE_FILE))
    return {}

def save_knowledge(k):
    json.dump(k, open(KNOWLEDGE_FILE,"w"), indent=2)

def bayesian_score(prefix_prob, ent, provider):
    knowledge = load_knowledge()
    learned = knowledge.get(provider, 1.0)
    likelihood = (ent / 5)
    posterior = (likelihood * prefix_prob * learned) / (
        likelihood * prefix_prob * learned + (1 - prefix_prob)
    )
    return posterior

def is_binary(url):
    return url.lower().endswith(SKIP_EXT)

# =====================================================
# SCAN ENGINE
# =====================================================
async def scan_js(session, url, base_domain):
    results = []
    if is_binary(url):
        return results

    try:
        async with session.get(url, timeout=10) as r:
            text = await r.text()
    except:
        return results

    blocks = [text]
    for b in re.findall(r'[A-Za-z0-9+/]{40,}={0,2}', text):
        try:
            blocks.append(base64.b64decode(b).decode(errors="ignore"))
        except:
            pass

    for block in blocks:
        for name, cfg in PROVIDERS.items():
            for m in re.finditer(cfg["regex"], block):
                token = m.group()
                ent = entropy(token)

                if ent < ENTROPY_MIN or ent > ENTROPY_MAX:
                    continue

                score = bayesian_score(cfg["prefix_prob"], ent, name)
                if score >= BAYES_THRESHOLD:
                    results.append({
                        "Service": name,
                        "Entropy": round(ent,2),
                        "Confidence": round(score,3),
                        "Token": token[:6]+"..."+token[-4:],
                        "Source": url
                    })
    return results

# =====================================================
# STREAMLIT UI
# =====================================================
st.set_page_config("Sovereign Final", layout="wide")
st.title("👑 Sovereign — Professional Bug Bounty Engine")

target = st.text_input("Target URL", "https://example.com")

if st.button("🚀 Run Professional Scan"):
    parsed = urlparse(target)
    base_domain = parsed.netloc

    async def run():
        findings = []
        async with aiohttp.ClientSession(headers={"User-Agent":USER_AGENT}) as s:
            async with s.get(target) as r:
                soup = BeautifulSoup(await r.text(), "html.parser")
                scripts = [
                    urljoin(target, sc["src"])
                    for sc in soup.find_all("script", src=True)
                    if urlparse(urljoin(target, sc["src"])).netloc == base_domain
                ]
            tasks = [scan_js(s, u, base_domain) for u in scripts]
            for res in await asyncio.gather(*tasks):
                findings.extend(res)
        return findings

    with st.spinner("Scanning like a professional…"):
        findings = asyncio.run(run())

    if findings:
        df = pd.DataFrame(findings)
        st.success(f"Confirmed Findings: {len(df)}")
        st.dataframe(df)

        if st.button("📄 Generate HackerOne Report"):
            report = ""
            for _,f in df.iterrows():
                report += f"""
## Exposed {f['Service']} Credential

**Asset:** {f['Source']}  
**Evidence:** {f['Token']}  
**Entropy:** {f['Entropy']}  
**Confidence:** {f['Confidence']}

**Impact:** Potential unauthorized access.

**Remediation:** Rotate and remove from client-side code.

---
"""
            st.download_button("Download Report.md", report, "report.md")
    else:
        st.info("No high-confidence issues found. (Professional Integrity Preserved)")
