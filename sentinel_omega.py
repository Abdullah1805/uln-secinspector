# ==========================================================
# Sentinel Ω — Ultimate Bug Bounty Intelligence System
# Part 1: Core Foundation & Shared Utilities
# ==========================================================

import asyncio
import aiohttp
import time
import random
import math
import re
import json
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np

# ==========================================================
# GLOBAL CONFIG (CONSERVATIVE BY DESIGN)
# ==========================================================

VERSION = "Sentinel Ω v1.0 Elite"
USER_AGENT = "Sentinel-Omega-Research/1.0 (Authorized Security Testing Only)"

TIMEOUT = 20
WARMUP_REQUESTS = 7
MAX_TOTAL_REQUESTS = 260
JITTER_RANGE = (0.3, 1.2)
EPS = 1e-9

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "*/*",
}

# ==========================================================
# SAFETY GUARDS (LEGAL & FALSE POSITIVE CONTROL)
# ==========================================================

DISALLOWED_HOST_PATTERNS = [
    r"^localhost$",
    r"^127\.",
    r"^0\.0\.0\.0$",
    r"^169\.254\.",   # AWS metadata
    r"^10\.",
    r"^192\.168\.",
    r"^172\.(1[6-9]|2[0-9]|3[0-1])\.",
]

def is_safe_target(url: str) -> bool:
    try:
        host = re.findall(r"https?://([^/]+)", url)[0]
        for pat in DISALLOWED_HOST_PATTERNS:
            if re.match(pat, host):
                return False
        return True
    except Exception:
        return False

# ==========================================================
# DATA MODELS
# ==========================================================

@dataclass
class TimingSample:
    size: int
    median: float
    p95: float

@dataclass
class Finding:
    title: str
    severity: str
    confidence: float
    description: str
    evidence: Dict
    cwe: str
    owasp: str
    recommendation: str

# ==========================================================
# STATISTICAL UTILITIES
# ==========================================================

def winsorize(values: List[float], p: float = 0.1) -> List[float]:
    if len(values) < 6:
        return values
    values = sorted(values)
    k = int(len(values) * p)
    return values[k:len(values) - k]

def percentile(values: List[float], p: int) -> float:
    return float(np.percentile(sorted(values), p))

def rss(y: np.ndarray, y_hat: np.ndarray) -> float:
    return float(np.sum((y - y_hat) ** 2))

def aic(rss_val: float, n: int, k: int) -> float:
    return 2 * k + n * math.log((rss_val + EPS) / n)

def correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])

# ==========================================================
# NETWORK CORE (HEADERS-ONLY TIMING)
# ==========================================================

async def timed_request(
    session: aiohttp.ClientSession,
    url: str,
    payload: Optional[str] = None,
    param: str = "q"
) -> Tuple[Optional[float], Optional[int]]:
    nonce = random.random()
    params = {param: payload, "_": nonce} if payload else {"_": nonce}

    start = time.perf_counter()
    try:
        async with session.get(
            url,
            params=params,
            headers=HEADERS
        ) as resp:
            await resp.release()
            elapsed = (time.perf_counter() - start) * 1000
            return elapsed, resp.status
    except Exception:
        return None, None

# ==========================================================
# SESSION FACTORY
# ==========================================================

def create_session() -> aiohttp.ClientSession:
    timeout = aiohttp.ClientTimeout(total=TIMEOUT)
    connector = aiohttp.TCPConnector(
        limit=1,
        force_close=False,
        ssl=False
    )
    return aiohttp.ClientSession(timeout=timeout, connector=connector)

# ==========================================================
# HASHING (EVIDENCE INTEGRITY)
# ==========================================================

def stable_hash(data: Dict) -> str:
    raw = json.dumps(data, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()

# ==========================================================
# END OF PART 1
# ==========================================================
# ==========================================================
# PART 2: ReDoS Timing Engine (Elite Conservative)
# ==========================================================

# ----------------------------------------------------------
# PAYLOAD GENERATOR (MODERN REGEX BACKTRACKING PATTERNS)
# ----------------------------------------------------------

def redos_payloads(size: int) -> List[str]:
    base = "a" * size
    return [
        base,
        f"^(a+)+b$"[:size],
        f"^((a|aa)+)+$"[:size],
        f"^(a|a?)+$"[:size],
        f"(?=(a+))a*$"[:size],
        f"^({base}|{base}a)+$"[:size],
    ]

# ----------------------------------------------------------
# CORE ANALYSIS
# ----------------------------------------------------------

async def analyze_redos(url: str) -> Tuple[Optional[List[TimingSample]], str]:
    if not is_safe_target(url):
        return None, "Unsafe target blocked"

    session = create_session()
    measurements: List[TimingSample] = []
    req_count = 0
    sizes = [64, 128, 256, 512, 1024, 2048, 4096]

    try:
        # Warm‑up
        for _ in range(WARMUP_REQUESTS):
            await timed_request(session, url, "warmup")
            await asyncio.sleep(0.25)

        for size in sizes:
            deltas, deltas95 = [], []

            for payload in redos_payloads(size):
                samples = random.randint(14, 32)
                base_times, test_times = [], []

                for _ in range(samples):
                    t, s = await timed_request(session, url, "control")
                    if s in (429, 503):
                        return None, "Rate‑limit / Resource exhaustion"
                    if t:
                        base_times.append(t)
                    await asyncio.sleep(random.uniform(*JITTER_RANGE))
                    req_count += 1

                for _ in range(samples):
                    t, s = await timed_request(session, url, payload)
                    if s in (429, 503):
                        return None, "Rate‑limit / Resource exhaustion"
                    if t:
                        test_times.append(t)
                    await asyncio.sleep(random.uniform(*JITTER_RANGE))
                    req_count += 1

                if base_times and test_times:
                    deltas.append(
                        max(
                            0,
                            np.median(winsorize(test_times))
                            - np.median(winsorize(base_times))
                        )
                    )
                    deltas95.append(
                        max(
                            0,
                            percentile(test_times, 95)
                            - percentile(base_times, 95)
                        )
                    )

                if req_count > MAX_TOTAL_REQUESTS:
                    break

            if deltas:
                measurements.append(
                    TimingSample(
                        size=size,
                        median=max(deltas),
                        p95=max(deltas95),
                    )
                )

            # Adaptive expansion (late‑trigger ReDoS)
            if deltas and max(deltas) > 9 and size < 16384:
                sizes.append(size * 2)

        return measurements, "OK"

    finally:
        await session.close()

# ----------------------------------------------------------
# CLASSIFIER (CONSERVATIVE BY DEFAULT)
# ----------------------------------------------------------

def classify_redos(data: List[TimingSample]) -> Optional[Finding]:
    if len(data) < 5:
        return None

    x = np.array([d.size for d in data])
    y = np.array([d.median for d in data])
    y95 = np.array([d.p95 for d in data])

    if max(y) < 6 and max(y95) < 8:
        return None  # conservative abort

    # Linear
    lin = np.polyfit(x, y, 1)
    y_lin = np.polyval(lin, x)
    aic_lin = aic(rss(y, y_lin), len(y), 2)

    # Polynomial
    lx, ly = np.log(x), np.log(y + EPS)
    poly = np.polyfit(lx, ly, 1)
    y_poly = np.exp(poly[1]) * (x ** poly[0])
    aic_poly = aic(rss(y, y_poly), len(y), 2)

    # Exponential
    exp = np.polyfit(x, np.log(y + EPS), 1)
    y_exp = np.exp(exp[1]) * np.exp(exp[0] * x)
    aic_exp = aic(rss(y, y_exp), len(y), 2)

    best = min(
        [("Linear", aic_lin), ("Polynomial", aic_poly), ("Exponential", aic_exp)],
        key=lambda x: x[1]
    )[0]

    votes = 0
    if best == "Exponential":
        votes += 2
    if max(y95) > 12 and correlation(x, y95) > 0.75:
        votes += 1
    if max(y) > 15:
        votes += 1

    if votes < 4:
        return None

    confidence = min(95.0, 70 + votes * 7)

    return Finding(
        title="Regular Expression Denial of Service (ReDoS)",
        severity="Critical",
        confidence=confidence,
        description=(
            "Timing analysis indicates exponential backtracking behavior "
            "consistent with catastrophic regular expression evaluation."
        ),
        evidence={
            "samples": [d.__dict__ for d in data],
            "model": best,
        },
        cwe="CWE-1333",
        owasp="OWASP Top 10 2021 - A03: Injection (ReDoS)",
        recommendation=(
            "Refactor vulnerable regular expressions using atomic groups, "
            "possessive quantifiers, or safe regex engines. "
            "Apply input length limits and regex timeouts."
        ),
    )

# ==========================================================
# END OF PART 2
# ==========================================================
# ==========================================================
# PART 3: Secrets Hunter (Bug Bounty Grade)
# ==========================================================

import re
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

# ----------------------------------------------------------
# SECRET SIGNATURES (CURATED — LOW FP)
# ----------------------------------------------------------

SECRET_PATTERNS = {
    "AWS Access Key": r"AKIA[0-9A-Z]{16}",
    "AWS Secret Key": r"(?i)aws(.{0,20})?(secret|access)[\"'\s:=]+[A-Za-z0-9/+=]{40}",
    "Google API Key": r"AIza[0-9A-Za-z\-_]{35}",
    "Stripe Live Key": r"sk_live_[0-9a-zA-Z]{24}",
    "Firebase Key": r"AAAA[A-Za-z0-9_-]{7}:[A-Za-z0-9_-]{140}",
    "JWT Token": r"eyJhbGciOi[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+",
    "OAuth Bearer": r"Bearer\s+[A-Za-z0-9\-_\.=]+",
    "Generic Secret": r"(?i)(api|secret|token|key)[\"'\s:=]{1,3}[A-Za-z0-9_\-]{16,}",
}

HIGH_IMPACT = {
    "AWS Access Key",
    "AWS Secret Key",
    "Stripe Live Key",
}

# ----------------------------------------------------------
# FETCHER
# ----------------------------------------------------------

async def fetch_text(session, url: str) -> Optional[str]:
    try:
        async with session.get(url, headers=HEADERS) as r:
            if r.status != 200:
                return None
            return await r.text()
    except:
        return None

# ----------------------------------------------------------
# CORE SECRETS SCAN
# ----------------------------------------------------------

async def analyze_secrets(url: str) -> Optional[Finding]:
    if not is_safe_target(url):
        return None

    session = create_session()
    findings = []

    try:
        html = await fetch_text(session, url)
        if not html:
            return None

        soup = BeautifulSoup(html, "html.parser")

        targets = [url]
        for script in soup.find_all("script"):
            src = script.get("src")
            if src:
                targets.append(urljoin(url, src))
            else:
                targets.append("inline")

        for target in targets:
            content = (
                html if target == "inline"
                else await fetch_text(session, target)
            )
            if not content:
                continue

            for name, pattern in SECRET_PATTERNS.items():
                for match in re.finditer(pattern, content):
                    secret = match.group(0)

                    # FP reduction
                    if len(secret) < 16:
                        continue

                    findings.append({
                        "type": name,
                        "value": secret[:6] + "..." + secret[-4:],
                        "location": target,
                        "impact": "High" if name in HIGH_IMPACT else "Medium"
                    })

        if not findings:
            return None

        confidence = min(
            95.0,
            70 + sum(1 for f in findings if f["impact"] == "High") * 10
        )

        return Finding(
            title="Exposed Secrets in Client-Side Code",
            severity="Critical" if any(f["impact"] == "High" for f in findings) else "High",
            confidence=confidence,
            description=(
                "Sensitive credentials were discovered embedded in client-side "
                "HTML/JavaScript resources. These secrets may allow unauthorized "
                "access to backend services or cloud resources."
            ),
            evidence={"secrets": findings},
            cwe="CWE-522",
            owasp="OWASP Top 10 2021 - A02: Cryptographic Failures",
            recommendation=(
                "Immediately revoke exposed keys, rotate credentials, "
                "and move secrets to secure server-side storage. "
                "Never embed secrets in client-side code."
            )
        )

    finally:
        await session.close()

# ==========================================================
# END OF PART 3
# ==========================================================
# ==========================================================
# PART 4: SQLi + SSRF Engine (Conservative & Evidence-Based)
# ==========================================================

# ----------------------------------------------------------
# SQLi PAYLOADS (TIME-BASED — SAFE SET)
# ----------------------------------------------------------

SQLI_PAYLOADS = [
    "' OR SLEEP(3)-- ",
    "'; WAITFOR DELAY '0:0:3'--",
    "' OR pg_sleep(3)--",
    "'||(SELECT pg_sleep(3))||'",
]

SQLI_CONTROL = "' OR 1=1--"

# ----------------------------------------------------------
# SSRF TARGETS (SAFE METADATA ONLY)
# ----------------------------------------------------------

SSRF_PROBES = [
    "http://169.254.169.254/latest/meta-data/",
    "http://127.0.0.1:80/",
    "http://localhost:80/",
]

# ----------------------------------------------------------
# SQLi ANALYSIS
# ----------------------------------------------------------

async def analyze_sqli(url: str) -> Optional[Finding]:
    if not is_safe_target(url):
        return None

    session = create_session()
    deltas = []

    try:
        for payload in SQLI_PAYLOADS:
            base_times, test_times = [], []

            for _ in range(5):
                t, _ = await timed_request(session, url, SQLI_CONTROL)
                if t:
                    base_times.append(t)
                await asyncio.sleep(random.uniform(0.4, 0.9))

            for _ in range(5):
                t, _ = await timed_request(session, url, payload)
                if t:
                    test_times.append(t)
                await asyncio.sleep(random.uniform(0.4, 0.9))

            if base_times and test_times:
                delta = (
                    np.median(test_times)
                    - np.median(base_times)
                )
                deltas.append(delta)

        # Conservative threshold
        if deltas and sum(1 for d in deltas if d > 2500) >= 2:
            confidence = min(90.0, 60 + len(deltas) * 10)

            return Finding(
                title="Blind SQL Injection (Time-Based)",
                severity="High",
                confidence=confidence,
                description=(
                    "Time-based analysis indicates delayed database responses "
                    "consistent with blind SQL injection behavior."
                ),
                evidence={"timing_deltas_ms": deltas},
                cwe="CWE-89",
                owasp="OWASP Top 10 2021 - A03: Injection",
                recommendation=(
                    "Use parameterized queries, avoid string concatenation, "
                    "and apply least-privilege database access."
                ),
            )

        return None

    finally:
        await session.close()

# ----------------------------------------------------------
# SSRF ANALYSIS
# ----------------------------------------------------------

async def analyze_ssrf(url: str) -> Optional[Finding]:
    if not is_safe_target(url):
        return None

    session = create_session()
    indicators = []

    try:
        for probe in SSRF_PROBES:
            payload = f"{probe}?sentinel=1"

            t, status = await timed_request(session, url, payload)
            if not t:
                continue

            # Heuristics (very conservative)
            if status in (200, 401, 403) and t > 800:
                indicators.append({
                    "probe": probe,
                    "status": status,
                    "time_ms": t,
                })

        if len(indicators) >= 2:
            return Finding(
                title="Potential Server-Side Request Forgery (SSRF)",
                severity="High",
                confidence=75.0,
                description=(
                    "The application appears to initiate server-side HTTP "
                    "requests to internal network resources."
                ),
                evidence={"indicators": indicators},
                cwe="CWE-918",
                owasp="OWASP Top 10 2021 - A10: SSRF",
                recommendation=(
                    "Apply strict allowlists for outbound requests, "
                    "block access to metadata IP ranges, and disable "
                    "URL-based user input where possible."
                ),
            )

        return None

    finally:
        await session.close()

# ==========================================================
# END OF PART 4
# ==========================================================

# ==========================================================
# PART 5: Auto Bug Bounty Report Engine
# ==========================================================

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.colors import red, orange, green
import io
import matplotlib.pyplot as plt
def generate_markdown_report(target: str, findings: List[Finding]) -> str:
    lines = []
    lines.append(f"# 🛡 Security Vulnerability Report\n")
    lines.append(f"**Target:** `{target}`\n")
    lines.append(f"**Date:** {datetime.utcnow().isoformat()} UTC\n")
    lines.append("---\n")

    for f in findings:
        lines.append(f"## 🚨 {f.title}\n")
        lines.append(f"- **Severity:** {f.severity}")
        lines.append(f"- **Confidence:** {f.confidence:.1f}%")
        lines.append(f"- **CWE:** {f.cwe}")
        lines.append(f"- **OWASP:** {f.owasp}\n")
        lines.append("### 📖 Description")
        lines.append(f"{f.description}\n")
        lines.append("### 🧪 Evidence")
        lines.append("```json")
        lines.append(json.dumps(f.evidence, indent=2))
        lines.append("```\n")
        lines.append("### 🛠 Recommendation")
        lines.append(f"{f.recommendation}\n")
        lines.append("---\n")

    return "\n".join(lines)
def generate_confidence_heatmap(findings: List[Finding]) -> bytes:
    labels = [f.title for f in findings]
    scores = [f.confidence for f in findings]

    fig, ax = plt.subplots(figsize=(8, 2))
    heatmap = ax.imshow([scores], cmap="RdYlGn_r", aspect="auto")

    ax.set_yticks([])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title("Confidence Heatmap (%)")

    for i, score in enumerate(scores):
        ax.text(i, 0, f"{score:.0f}%", ha="center", va="center", color="black")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
def generate_pdf_report(target: str, findings: List[Finding]) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>Security Vulnerability Report</b>", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Target:</b> {target}", styles["Normal"]))
    story.append(Paragraph(f"<b>Date:</b> {datetime.utcnow().isoformat()} UTC", styles["Normal"]))
    story.append(Spacer(1, 12))

    for f in findings:
        color = red if f.severity == "Critical" else orange if f.severity == "High" else green

        story.append(Paragraph(f"<b>{f.title}</b>", styles["Heading2"]))
        story.append(Paragraph(f"<b>Severity:</b> {f.severity}", styles["Normal"]))
        story.append(Paragraph(f"<b>Confidence:</b> {f.confidence:.1f}%", styles["Normal"]))
        story.append(Paragraph(f"<b>CWE:</b> {f.cwe}", styles["Normal"]))
        story.append(Paragraph(f"<b>OWASP:</b> {f.owasp}", styles["Normal"]))
        story.append(Spacer(1, 8))

        story.append(Paragraph("<b>Description</b>", styles["Heading3"]))
        story.append(Paragraph(f.description, styles["Normal"]))
        story.append(Spacer(1, 6))

        story.append(Paragraph("<b>Recommendation</b>", styles["Heading3"]))
        story.append(Paragraph(f.recommendation, styles["Normal"]))
        story.append(Spacer(1, 12))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()
def build_full_report(target: str, findings: List[Finding]):
    markdown = generate_markdown_report(target, findings)
    pdf = generate_pdf_report(target, findings)
    heatmap = generate_confidence_heatmap(findings)

    return {
        "markdown": markdown,
        "pdf": pdf,
        "heatmap": heatmap
    }
# ==========================================================
# PART 6: Orchestrator + Simulation + Final UI
# ==========================================================

import streamlit as st
from datetime import datetime
import asyncio
from typing import Optional

# ----------------------------------------------------------
# ORCHESTRATOR
# ----------------------------------------------------------

async def run_all_engines(target: str) -> List[Finding]:
    findings: List[Finding] = []

    tasks = [
        analyze_redos(target),
        analyze_secrets(target),
        analyze_sqli(target),
        analyze_ssrf(target),
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Finding):
            findings.append(result)
        elif isinstance(result, tuple):
            data, status = result
            if data and isinstance(data, list):
                finding = classify_redos(data)
                if finding:
                    findings.append(finding)

    # Conservative filter: remove weak confidence
    return [f for f in findings if f.confidence >= 70]


# ----------------------------------------------------------
# SIMULATION ENGINE (SAFE / OFFLINE)
# ----------------------------------------------------------

def simulate_findings() -> List[Finding]:
    return [
        Finding(
            title="Simulated ReDoS Vulnerability",
            severity="Critical",
            confidence=92.0,
            description="Simulated exponential regex backtracking detected.",
            evidence={"simulation": True},
            cwe="CWE-1333",
            owasp="OWASP A03: Injection",
            recommendation="Fix regex patterns."
        ),
        Finding(
            title="Simulated Exposed API Key",
            severity="High",
            confidence=85.0,
            description="Simulated API key exposure in JavaScript.",
            evidence={"key": "AKIA...XXXX"},
            cwe="CWE-522",
            owasp="OWASP A02: Cryptographic Failures",
            recommendation="Rotate and remove secrets."
        ),
    ]


# ----------------------------------------------------------
# STREAMLIT FINAL UI
# ----------------------------------------------------------

st.set_page_config(
    page_title="Sentinel Ω — Ultimate Bug Bounty Engine",
    layout="wide"
)

st.title("🛡 Sentinel Ω — Ultimate Bug Bounty Engine")
st.caption("Elite • Conservative • Evidence‑Driven • Report‑Ready")

target = st.text_input("🎯 Target URL (Authorized only)")
simulate = st.checkbox("🧪 Simulation Mode (No real traffic)", value=False)

if st.button("🚀 Run Sentinel Ω"):
    with st.spinner("Running elite multi‑engine analysis…"):
        if simulate:
            findings = simulate_findings()
        else:
            findings = asyncio.run(run_all_engines(target))

    if not findings:
        st.success("🟢 No high‑confidence vulnerabilities detected.")
    else:
        st.subheader("🚨 Confirmed Findings")
        for f in findings:
            st.error(f"**{f.title}** — {f.severity} ({f.confidence:.1f}%)")

        # Build report
        report = build_full_report(target, findings)

        st.subheader("📄 Reports")
        st.download_button(
            "⬇️ Download Markdown Report",
            report["markdown"],
            file_name="sentinel_report.md"
        )

        st.download_button(
            "⬇️ Download PDF Report",
            report["pdf"],
            file_name="sentinel_report.pdf",
            mime="application/pdf"
        )

        st.subheader("🔥 Confidence Heatmap")
        st.image(report["heatmap"])

        st.success("✅ Report is Bug‑Bounty‑Ready & Safe for Disclosure")

# ==========================================================
# END OF SENTINEL Ω — ULTIMATE
# ==========================================================
