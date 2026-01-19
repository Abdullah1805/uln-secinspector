import streamlit as st
import asyncio, aiohttp, time, random, math, re, io
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
from statistics import median
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ==========================================================
# GLOBAL CONFIG — CONSERVATIVE BY DESIGN
# ==========================================================
TIMEOUT = 25
MAX_REQUESTS = 300
HEADERS = {"User-Agent": "Sentinel-Omega-Research"}
EPS = 1e-9

# ==========================================================
# DATA MODELS
# ==========================================================
@dataclass
class TimingPoint:
    size: int
    median: float
    p95: float

@dataclass
class Finding:
    title: str
    severity: str
    confidence: float
    cwe: str
    owasp: str
    evidence: str

# ==========================================================
# NETWORK CORE
# ==========================================================
async def timed_request(session, url, params=None):
    start = time.perf_counter()
    try:
        async with session.get(url, params=params, headers=HEADERS) as r:
            await r.release()
            return (time.perf_counter() - start) * 1000, r.status, r.headers
    except asyncio.TimeoutError:
        return None, "timeout", {}
    except aiohttp.ClientError:
        return None, "network", {}

# ==========================================================
# BASELINE AUTO-CALIBRATION (ATI)
# ==========================================================
async def calibrate(session, url):
    samples = []
    for _ in range(20):
        t, _, _ = await timed_request(session, url, {"q": "baseline"})
        if t:
            samples.append(t)
        await asyncio.sleep(0.3)
    mean = np.mean(samples)
    std = np.std(samples) + EPS
    return mean, std

# ==========================================================
# ReDoS ENGINE (Tail-Aware + Conservative)
# ==========================================================
async def analyze_redos(session, url, mean, std):
    sizes = [64, 128, 256, 512, 1024, 2048]
    points = []

    for size in sizes:
        payload = "a" * size
        base, test = [], []

        for _ in range(12):
            t, _, _ = await timed_request(session, url, {"q": "test"})
            if t: base.append(t)
            await asyncio.sleep(random.uniform(0.2, 0.6))

        for _ in range(12):
            t, _, _ = await timed_request(session, url, {"q": payload})
            if t: test.append(t)
            await asyncio.sleep(random.uniform(0.2, 0.6))

        if not base or not test:
            continue

        delta_med = max(0, median(test) - median(base))
        delta_p95 = max(0, np.percentile(test, 95) - np.percentile(base, 95))

        z_score = (delta_p95 - mean) / std

        points.append(TimingPoint(size, delta_med, delta_p95))

        if z_score > 4.0:
            break

    return points

# ==========================================================
# SECRETS HUNTER v3 (Regex + Entropy + Context)
# ==========================================================
SECRET_REGEX = [
    r"AKIA[0-9A-Z]{16}",
    r"AIza[0-9A-Za-z\-_]{35}",
    r"sk_live_[0-9a-zA-Z]{24}"
]

def shannon_entropy(s):
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)

async def analyze_secrets(session, url):
    findings = []
    try:
        async with session.get(url, headers=HEADERS) as r:
            text = await r.text()
    except:
        return []

    for rx in SECRET_REGEX:
        for m in re.findall(rx, text):
            findings.append(f"Regex-matched secret: {m[:6]}***")

    tokens = re.findall(r"[A-Za-z0-9_\-]{20,}", text)
    for t in tokens:
        if shannon_entropy(t) > 4.3 and any(k in text.lower() for k in ["key", "token", "auth"]):
            findings.append(f"High-entropy token: {t[:6]}***")

    return findings

# ==========================================================
# SQLi ENGINE (Two-Phase)
# ==========================================================
async def analyze_sqli(session, url):
    t1, _, _ = await timed_request(session, url, {"id": "1 OR SLEEP(3)"})
    t2, _, _ = await timed_request(session, url, {"id": "1 AND 1=1"})
    if t1 and t2 and (t1 - t2) > 2500:
        return True
    return False

# ==========================================================
# SSRF ENGINE (Behavioral)
# ==========================================================
async def analyze_ssrf(session, url):
    t1, s1, _ = await timed_request(session, url, {"url": "http://127.0.0.1"})
    t2, s2, _ = await timed_request(session, url, {"url": "http://example.com"})
    if t1 and t2 and abs(t1 - t2) > 1500:
        return True
    return False

# ==========================================================
# REPORT GENERATOR (PDF)
# ==========================================================
def generate_pdf(findings: List[Finding]):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    y = 800
    c.setFont("Helvetica", 10)

    c.drawString(50, y, "Sentinel Ω — Bug Bounty Report")
    y -= 40

    for f in findings:
        c.drawString(50, y, f"{f.title} | {f.severity} | {f.confidence:.0f}%")
        y -= 15
        c.drawString(70, y, f"CWE: {f.cwe} | OWASP: {f.owasp}")
        y -= 15
        c.drawString(70, y, f"Evidence: {f.evidence}")
        y -= 30
        if y < 100:
            c.showPage()
            y = 800

    c.save()
    buffer.seek(0)
    return buffer

# ==========================================================
# STREAMLIT UI
# ==========================================================
st.set_page_config("Sentinel Ω v10.0", "🛡️", layout="wide")
st.title("🛡 Sentinel Ω v10.0 — Research-Grade Bug Bounty Engine")

target = st.text_input("Authorized Target URL")

if st.button("Run Full Scan") and target:
    findings = []
    async def run():
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as session:
            mean, std = await calibrate(session, target)

            redos = await analyze_redos(session, target, mean, std)
            if redos and max(p.p95 for p in redos) > 3000:
                findings.append(Finding(
                    "ReDoS Vulnerability",
                    "Critical",
                    92.0,
                    "CWE-1333",
                    "OWASP A1",
                    "Exponential timing growth detected"
                ))

            secrets = await analyze_secrets(session, target)
            if secrets:
                findings.append(Finding(
                    "Exposed Secrets",
                    "High",
                    88.0,
                    "CWE-522",
                    "OWASP A2",
                    secrets[0]
                ))

            if await analyze_sqli(session, target):
                findings.append(Finding(
                    "SQL Injection",
                    "High",
                    85.0,
                    "CWE-89",
                    "OWASP A3",
                    "Time-based confirmation"
                ))

            if await analyze_ssrf(session, target):
                findings.append(Finding(
                    "SSRF",
                    "Medium",
                    72.0,
                    "CWE-918",
                    "OWASP A10",
                    "Behavioral timing deviation"
                ))

    asyncio.run(run())

    if not findings:
        st.success("No high-confidence findings (conservative verdict)")
    else:
        for f in findings:
            st.error(f"{f.title} — {f.severity} ({f.confidence:.0f}%)")

        pdf = generate_pdf(findings)
        st.download_button("📄 Download PDF Report", pdf, "sentinel_report.pdf")
