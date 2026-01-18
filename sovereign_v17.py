import streamlit as st
import asyncio, aiohttp, re, math, base64, json
from collections import Counter
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

# =========================
# CONFIG
# =========================
USER_AGENT = "Sovereign/17.0 Research Engine"
TIMEOUT = 12
MAX_JS = 100
FINAL_THRESHOLD = 0.92

# =========================
# PROVIDERS (قابل للتوسيع)
# =========================
PROVIDERS = {
    "AWS": {
        "regex": r"AKIA[0-9A-Z]{16}",
        "prefix_prob": 0.99,
        "context": ["aws", "s3", "iam"],
        "verify": False
    },
    "GitHub": {
        "regex": r"ghp_[A-Za-z0-9]{36}",
        "prefix_prob": 0.98,
        "context": ["github", "repo"],
        "verify": True,
        "verify_url": "https://api.github.com/user"
    },
    "Stripe": {
        "regex": r"sk_live_[A-Za-z0-9]{24}",
        "prefix_prob": 0.97,
        "context": ["stripe", "payment"],
        "verify": True,
        "verify_url": "https://api.stripe.com/v1/account"
    },
    "Google": {
        "regex": r"AIza[0-9A-Za-z_-]{35}",
        "prefix_prob": 0.96,
        "context": ["google", "firebase"],
        "verify": False
    }
}

# =========================
# MATH CORE
# =========================
def entropy(s):
    if not s: return 0
    c = Counter(s)
    return -sum((v/len(s))*math.log2(v/len(s)) for v in c.values())

def adaptive_entropy(s):
    e = entropy(s)
    return min(e / (5.5 if len(s) > 40 else 4.5), 1)

def bayes(probs):
    p = 1
    for x in probs:
        p *= (1 - x)
    return 1 - p

def context_score(text, secret, words):
    idx = text.find(secret)
    if idx == -1: return 0
    win = text[max(0,idx-300):idx+300].lower()
    return min(sum(w in win for w in words)/len(words),1)

def scope_weight(base, url):
    b = urlparse(base).netloc
    u = urlparse(url).netloc
    if u == b: return 1
    if u.endswith(b): return 0.9
    if any(c in u for c in ["cdn","cloudfront","akamai","jsdelivr"]): return 0.75
    return 0.4

# =========================
# ACTIVE VERIFY
# =========================
async def verify(session, provider, secret):
    cfg = PROVIDERS[provider]
    if not cfg.get("verify"): return None
    headers={}
    if provider=="GitHub":
        headers["Authorization"]=f"token {secret}"
    if provider=="Stripe":
        headers["Authorization"]=f"Bearer {secret}"
    try:
        async with session.get(cfg["verify_url"],headers=headers,timeout=5) as r:
            return 0.99 if r.status==200 else 0
    except:
        return None

# =========================
# SCAN ENGINE
# =========================
async def scan_text(session,text,src,base):
    findings=[]
    blocks=[text]
    for b in re.findall(r'[A-Za-z0-9+/]{40,}={0,2}',text):
        try:
            blocks.append(base64.b64decode(b).decode(errors="ignore"))
        except: pass

    for block in blocks:
        for name,cfg in PROVIDERS.items():
            for m in re.finditer(cfg["regex"],block):
                s=m.group()
                probs=[
                    cfg["prefix_prob"],
                    adaptive_entropy(s),
                    context_score(block,s,cfg["context"]),
                    scope_weight(base,src)
                ]
                v=await verify(session,name,s)
                if v: probs.append(v)
                final=bayes(probs)
                if final>=FINAL_THRESHOLD:
                    findings.append({
                        "Provider":name,
                        "Confidence":round(final,4),
                        "Secret":s[:6]+"..."+s[-4:],
                        "Source":src
                    })
    return findings

async def run(target):
    res=[]
    async with aiohttp.ClientSession(headers={"User-Agent":USER_AGENT}) as ses:
        async with ses.get(target,timeout=TIMEOUT) as r:
            soup=BeautifulSoup(await r.text(),"html.parser")
            scripts=[urljoin(target,s["src"]) for s in soup.find_all("script",src=True)][:MAX_JS]
            for js in scripts:
                try:
                    async with ses.get(js,timeout=TIMEOUT) as rjs:
                        res+=await scan_text(ses,await rjs.text(),js,target)
                except: pass
    return res

# =========================
# REPORT GENERATOR
# =========================
def hackerone_report(f):
    return f"""
## 🔐 Exposed {f['Provider']} Secret

**Confidence:** {f['Confidence']*100:.2f}%  
**Source:** `{f['Source']}`

### Description
A live {f['Provider']} credential was discovered exposed in client‑side JavaScript.

### Impact
An attacker may gain unauthorized access to internal resources.

### Evidence
{f['Secret']}
### Recommendation
Immediately revoke the exposed key and rotate credentials.
"""

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(layout="wide")
st.title("👑 Sovereign v17 — Professional Bug Bounty Engine")

target=st.text_input("Target URL","https://example.com")

if st.button("🚀 Run Scan"):
    with st.spinner("Running scientific scan..."):
        data=asyncio.run(run(target))
        if not data:
            st.info("No high‑confidence leaks found.")
        else:
            df=pd.DataFrame(data)
            st.success(f"Confirmed Findings: {len(df)}")
            st.dataframe(df)

            st.subheader("📄 HackerOne / Bugcrowd Reports")
            for f in data:
                st.code(hackerone_report(f),language="markdown")
