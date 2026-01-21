# ============================================================
# Sentinel Ω — Evidence‑Grade Bug Bounty Engine (STREAMLIT)
#
# Silent • Conservative • HackerOne‑Ready
#
# Researcher : Abdullah Abbas
# Contact    : WhatsApp +964 07881296737
#
# ============================================================

import streamlit as st
import asyncio
import aiohttp
import re
import json
import math
import logging
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass
from typing import List, Optional, Set

# =========================
# CONFIG
# =========================

REQUEST_TIMEOUT = 12
MAX_JS_DEPTH = 2
ENTROPY_THRESHOLD = 3.7
MIN_CONFIDENCE = 0.92
MAX_CONCURRENT_REQUESTS = 10

HIGH_RISK_ENDPOINT_WORDS = (
    "admin", "internal", "debug", "delete", "config", "root"
)

CDN_BLACKLIST = (
    "cloudflare", "jsdelivr", "googleapis", "gstatic", "cdnjs"
)

HEADERS = {
    "User-Agent": "Sentinel-Omega/Research",
    "Accept": "*/*"
}

logging.basicConfig(
    filename="scan_debug.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# =========================
# DATA MODEL
# =========================

@dataclass
class Evidence:
    kind: str
    value: str
    location: str
    confidence: float
    proof: str

# =========================
# UTILS
# =========================

def entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    return -sum((v/len(s)) * math.log2(v/len(s)) for v in freq.values())

def root_domain(url: str) -> str:
    parts = urlparse(url).netloc.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else url

def same_scope(base: str, target: str) -> bool:
    if any(cdn in target for cdn in CDN_BLACKLIST):
        return False
    return root_domain(base) == root_domain(target)

def normalize_url(base: str, src: str) -> str:
    if src.startswith("//"):
        return "https:" + src
    return urljoin(base, src)

# =========================
# PATTERNS
# =========================

SECRET_PATTERNS = {
    "AWS Access Key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "Google API Key": re.compile(r"AIza[0-9A-Za-z\-_]{35}"),
    "Slack Webhook": re.compile(
        r"https://hooks\.slack\.com/services/T[a-zA-Z0-9_]+/B[a-zA-Z0-9_]+/[a-zA-Z0-9_]+"
    ),
}

SENSITIVE_COMMENT = re.compile(
    r"(TODO|FIXME|PASSWORD|SECRET|CREDS|API[_-]?KEY)",
    re.IGNORECASE
)

ENDPOINT_PATTERN = re.compile(r"['\"](/[^'\"\s]{3,100})['\"]")

# =========================
# ENGINE
# =========================

class SentinelEngine:

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session: Optional[aiohttp.ClientSession] = None
        self.evidence: List[Evidence] = []
        self.headers_findings: List[str] = []
        self.endpoints_seen: Set[str] = set()
        self.sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers=HEADERS,
            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        )
        return self

    async def __aexit__(self, *args):
        await self.session.close()

    async def fetch(self, url: str) -> Optional[str]:
        async with self.sem:
            try:
                async with self.session.get(url) as r:
                    self.analyze_headers(r.headers, url)
                    if r.status == 200:
                        return await r.text(errors="ignore")
            except Exception:
                pass
        return None

    def analyze_headers(self, headers, url):
        if "strict-transport-security" not in headers:
            self.headers_findings.append(
                f"{url} → Missing Strict-Transport-Security"
            )
        if "x-powered-by" in headers:
            self.headers_findings.append(
                f"{url} → X-Powered-By: {headers.get('x-powered-by')}"
            )

    def extract_endpoints(self, text: str, location: str):
        for ep in ENDPOINT_PATTERN.findall(text):
            if ep in self.endpoints_seen:
                continue
            self.endpoints_seen.add(ep)
            if any(w in ep.lower() for w in HIGH_RISK_ENDPOINT_WORDS):
                self.evidence.append(Evidence(
                    kind="High‑Risk Endpoint (Context)",
                    value=ep,
                    location=location,
                    confidence=0.60,
                    proof="Endpoint discovered in client-side source"
                ))

    async def analyze_source_map(self, sm_url: str):
        raw = await self.fetch(sm_url)
        if not raw or len(raw) > 2_000_000:
            return
        try:
            data = json.loads(raw)
        except Exception:
            return

        for src, content in zip(
            data.get("sources", []),
            data.get("sourcesContent", [])
        ):
            self.extract_endpoints(content, f"{sm_url} :: {src}")
            for name, rx in SECRET_PATTERNS.items():
                for m in rx.findall(content):
                    if entropy(m) >= ENTROPY_THRESHOLD:
                        self.evidence.append(Evidence(
                            kind=f"{name} (SourceMap)",
                            value=m,
                            location=f"{sm_url} :: {src}",
                            confidence=0.97,
                            proof="Secret exposed in original source via source map"
                        ))

    async def analyze_js(self, js_url: str, depth: int = 0):
        if depth > MAX_JS_DEPTH:
            return

        body = await self.fetch(js_url)
        if not body:
            return

        self.extract_endpoints(body, js_url)

        for name, rx in SECRET_PATTERNS.items():
            for m in rx.findall(body):
                if entropy(m) >= ENTROPY_THRESHOLD:
                    self.evidence.append(Evidence(
                        kind=name,
                        value=m,
                        location=js_url,
                        confidence=0.95,
                        proof="High‑entropy secret exposed in JavaScript"
                    ))

        for m in re.findall(r"(?:import|require)\s*['\"]([^'\"]+\.js)", body):
            nxt = normalize_url(js_url, m)
            if same_scope(self.base_url, nxt):
                await self.analyze_js(nxt, depth + 1)

        if js_url.endswith(".js"):
            await self.analyze_source_map(js_url + ".map")

    async def run(self):
        html = await self.fetch(self.base_url)
        if not html:
            return

        scripts = set(re.findall(
            r"<script[^>]+src=['\"]([^'\"]+)",
            html,
            re.IGNORECASE
        ))

        await asyncio.gather(*[
            self.analyze_js(normalize_url(self.base_url, s))
            for s in scripts
            if s.endswith(".js") and same_scope(self.base_url, s)
        ])

# =========================
# REPORT
# =========================

def generate_report(target: str, evidence: List[Evidence], headers: List[str]) -> str:
    out = []
    out.append("# HackerOne Security Report\n")
    out.append(f"**Target:** {target}")
    out.append("**Researcher:** Abdullah Abbas")
    out.append("**Contact:** WhatsApp +964 07881296737\n")

    out.append("## Summary")
    out.append(
        "A verified security issue was identified using conservative, "
        "evidence‑driven analysis.\n"
    )

    out.append("## Verified Evidence")
    for e in evidence:
        out.append(f"- **Type:** {e.kind}")
        out.append(f"  - Location: `{e.location}`")
        out.append(f"  - Value: `{e.value}`")
        out.append(f"  - Proof: {e.proof}\n")

    if headers:
        out.append("## Passive Observations")
        for h in headers:
            out.append(f"- {h}")

    out.append("\n---")
    out.append("## Digital Signature")
    out.append("Researcher: **Abdullah Abbas**")
    out.append("Contact: **WhatsApp +964 07881296737**")

    return "\n".join(out)

# =========================
# STREAMLIT UI
# =========================

st.set_page_config(page_title="Sentinel Ω", layout="centered")
st.title("Sentinel Ω — Bug Bounty Scanner")
st.caption("Silent • Evidence‑Grade • HackerOne‑Ready")

target = st.text_input("Authorized Target URL")

if st.button("Start Scan") and target:
    with st.spinner("Scanning…"):
        async def run_scan():
            async with SentinelEngine(target) as engine:
                await engine.run()
                confirmed = [e for e in engine.evidence if e.confidence >= MIN_CONFIDENCE]
                return confirmed, engine.headers_findings

        findings, headers = asyncio.run(run_scan())

    if not findings:
        st.success("Scan completed. No verified vulnerabilities found.")
    else:
        report = generate_report(target, findings, headers)
        st.error("Verified vulnerability found")
        st.download_button(
            "Download HackerOne Report",
            report,
            file_name="HackerOne_Report.md"
        )
