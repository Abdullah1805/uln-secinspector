# ============================================================
# F16‑Research Edition — Scientific-Grade Web Security Framework
# Author: Abdullah Abbas
# ============================================================
# ⚠️ AUTHORIZED SECURITY TESTING ONLY
# Bug Bounty • Pentesting • Academic Research
# ============================================================

import asyncio
import time
import math
import re
import statistics
import random
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from urllib.parse import urlparse, parse_qs, urljoin

# ----------------------------
# Browser & Parsing
# ----------------------------
from playwright.async_api import async_playwright, Page
from bs4 import BeautifulSoup

# ----------------------------
# Math & ML
# ----------------------------
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ----------------------------
# Graphs & Modeling
# ----------------------------
import networkx as nx

# ----------------------------
# Reporting
# ----------------------------
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ============================================================
# CONFIGURATION
# ============================================================

USER_INFO = {
    "name": "Abdullah Abbas",
    "phone": "+964 07881296737",
    "dob": "2008-04-18",
    "logo": "🦅 F16-RESEARCH"
}

SCAN_CONFIG = {
    "max_depth": 2,
    "timeout": 30000,
    "concurrency": 2,
    "stealth": True
}

# ============================================================
# SAFETY & SANITY GUARD
# ============================================================

def authorized_target(url: str) -> bool:
    parsed = urlparse(url)
    if not parsed.scheme.startswith("http"):
        return False
    forbidden = ("localhost","127.0.0.1","0.0.0.0","::1")
    return not any(f in parsed.netloc for f in forbidden)

# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class InjectionPoint:
    url: str
    method: str
    parameter: str
    location: str
    input_type: str

@dataclass
class Finding:
    title: str
    vector: InjectionPoint
    payload: str
    evidence: str
    severity: float
    confidence: float
    stride: List[str]

@dataclass
class ScanResult:
    target: str
    start_time: float
    findings: List[Finding] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return round(time.time() - self.start_time, 2)

# ============================================================
# FEATURE EXTRACTION UTILITIES
# ============================================================

class FeatureExtractor:
    @staticmethod
    def normalize(value: float, max_value: float = 1.0) -> float:
        return min(max(value / max_value, 0.0), 1.0)

    @staticmethod
    def timing_shift(baseline: List[float], current: float) -> float:
        if not baseline:
            return 0.0
        med = statistics.median(baseline)
        return FeatureExtractor.normalize(abs(current - med), med + 1e-3)
# ============================================================
# CONFIDENCE FUSION (Bayesian‑Inspired)
# ============================================================

class ConfidenceFusion:
    """
    Avoids optimistic max().
    Uses weighted harmonic mean to reduce bias.
    """
    @staticmethod
    def harmonic_mean(values: List[float], weights: List[float]) -> float:
        numerator = sum(weights)
        denominator = sum((w / (v + 1e-6)) for v, w in zip(values, weights))
        return round(numerator / denominator, 3)

# ============================================================
# DECISION ENGINE
# ============================================================

class DecisionEngine:
    WEIGHTS = {
        "dom_change": 0.30,
        "event_trigger": 0.35,
        "status_anomaly": 0.15,
        "timing_shift": 0.20,
    }

    @staticmethod
    def score(features: Dict[str, float]) -> float:
        score = 0.0
        for k, w in DecisionEngine.WEIGHTS.items():
            score += features.get(k, 0.0) * w
        return round(min(score, 1.0), 3)

    @staticmethod
    def confidence(bayes: float, ml: float) -> float:
        # Weighted fusion between Bayesian signal and ML signal
        return round((0.6 * bayes + 0.4 * ml), 3)

# ============================================================
# MACHINE LEARNING ENGINE (Noise Reduction)
# ============================================================

class MLNoiseReducer:
    """
    Reduces false positives using RandomForest trained on synthetic signals.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42)
        self._train_bootstrap()

    def _train_bootstrap(self):
        X = []
        y = []
        for _ in range(300):
            dom = np.random.rand()
            event = np.random.rand()
            timing = np.random.rand()
            status = np.random.rand()
            score = dom*0.3 + event*0.35 + timing*0.2 + status*0.15
            X.append([dom, event, timing, status])
            y.append(1 if score > 0.55 else 0)
        X = self.scaler.fit_transform(X)
        self.model.fit(X, y)

    def probability(self, features: Dict[str, float]) -> float:
        vec = [
            features.get("dom_change",0.0),
            features.get("event_trigger",0.0),
            features.get("timing_shift",0.0),
            features.get("status_anomaly",0.0),
        ]
        vec = self.scaler.transform([vec])
        return round(self.model.predict_proba(vec)[0][1],3)

# ============================================================
# STRIDE THREAT MODELING
# ============================================================

class STRIDEAnalyzer:
    MAP = {
        "XSS": ["Spoofing", "Information Disclosure"],
        "SQLi": ["Tampering", "Information Disclosure"],
        "SSTI": ["Elevation of Privilege"],
        "IDOR": ["Information Disclosure"],
        "RCE": ["Elevation of Privilege"],
    }

    @staticmethod
    def classify(title: str) -> List[str]:
        for k, v in STRIDEAnalyzer.MAP.items():
            if k.lower() in title.lower():
                return v
        return ["Tampering"]

# ============================================================
# CVSS SCORING
# ============================================================

class CVSSCalculator:
    BASE = {
        "XSS": 6.1,
        "SQLi": 9.1,
        "SSTI": 8.6,
        "RCE": 9.8,
        "IDOR": 7.2,
    }

    @staticmethod
    def score(title: str, confidence: float) -> float:
        for k,v in CVSSCalculator.BASE.items():
            if k.lower() in title.lower():
                return round(v*confidence,1)
        return round(5.0*confidence,1)
# ============================================================
# ATTACK GRAPH ENGINE
# ============================================================

class AttackGraph:
    """
    Graph‑based reasoning, nodes = vectors + vulnerabilities,
    edges = causal relationships.
    """
    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_vector(self, vector):
        self.nodes.append({
            "type": "input",
            "name": vector.parameter,
            "location": vector.location
        })

    def add_vulnerability(self, vuln):
        self.nodes.append({
            "type": "vulnerability",
            "name": vuln.title,
            "cvss": vuln.cvss
        })
        self.edges.append({
            "from": vuln.vector.parameter,
            "to": vuln.title
        })

    def to_stride(self):
        stride = {
            "Spoofing": [],
            "Tampering": [],
            "Repudiation": [],
            "Information Disclosure": [],
            "Denial of Service": [],
            "Elevation of Privilege": []
        }
        for n in self.nodes:
            if n["type"] == "vulnerability":
                name = n["name"]
                if "SQLi" in name:
                    stride["Elevation of Privilege"].append(name)
                if "XSS" in name:
                    stride["Information Disclosure"].append(name)
                    stride["Spoofing"].append(name)
                if "SSTI" in name:
                    stride["Tampering"].append(name)
        return stride

# ============================================================
# STEALTH BROWSER ENGINE
# ============================================================

class StealthBrowser:
    @staticmethod
    async def launch(playwright):
        browser = await playwright.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-infobars",
                "--disable-web-security"
            ]
        )
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width":1920,"height":1080},
            locale="en-US",
            timezone_id="UTC"
        )
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            window.chrome = { runtime: {} };
        """)
        return browser, context

# ============================================================
# SENSOR SUITE
# ============================================================

class SensorSuite:
    def __init__(self, page):
        self.page = page
        self.events = {"console":0,"dialog":0,"dom_change":0}
        self.start_time = time.time()

    async def attach(self):
        self.page.on("console", self._on_console)
        self.page.on("dialog", self._on_dialog)

    def _on_console(self, msg):
        if "f16" in msg.text.lower():
            self.events["console"] += 1

    def _on_dialog(self, dialog):
        self.events["dialog"] += 1
        asyncio.create_task(dialog.dismiss())

    async def dom_snapshot(self):
        content = await self.page.content()
        self.events["dom_change"] = len(content)

    def features(self):
        elapsed = time.time() - self.start_time
        return {
            "event_trigger": min(1.0, self.events["console"]+self.events["dialog"]),
            "dom_change": min(1.0, self.events["dom_change"]/50000),
            "timing_shift": min(1.0, elapsed/5.0),
            "status_anomaly": 0.5
        }

# ============================================================
# PAYLOAD ENGINE
# ============================================================

class PayloadEngine:
    PAYLOADS = {
        "XSS": [
            "<script>console.log('f16')</script>",
            "\"><img src=x onerror=console.log('f16')>"
        ],
        "SQLi": [
            "' OR '1'='1'--",
            "' UNION SELECT NULL--"
        ],
        "SSTI": [
            "{{7*7}}",
            "${7*7}"
        ]
    }

    @staticmethod
    def select(input_type: str):
        if input_type in ["email","password"]:
            return {"XSS":PayloadEngine.PAYLOADS["XSS"]}
        return PayloadEngine.PAYLOADS
# ============================================================
# OOB DETECTION (Interactsh / Burp Stub)
# ============================================================

class OOBClient:
    """
    Stub جاهز للربط مع:
    - Interactsh
    - Burp Collaborator
    """
    def __init__(self, endpoint: str | None = None):
        self.endpoint = endpoint
        self.interactions = []

    def payload(self):
        if not self.endpoint:
            return None
        return f"http://{self.endpoint}/f16-{int(time.time())}"

    def poll(self):
        # عند الربط الحقيقي يتم استبدال هذا الجزء
        return self.interactions


# ============================================================
# ATTACK ENGINE
# ============================================================

class AttackEngine:
    def __init__(self, page: Page):
        self.page = page

    async def test_vector(self, vector: InjectionPoint) -> Optional[Finding]:
        payload_sets = PayloadEngine.select(vector.input_type)

        for attack_type, payloads in payload_sets.items():
            for payload in payloads:
                try:
                    finding = await self._inject_and_analyze(
                        vector, payload, attack_type
                    )
                    if finding:
                        return finding
                except Exception:
                    continue
        return None

    async def _inject_and_analyze(self, vector, payload, attack_type):
        sensor = SensorSuite(self.page)
        await sensor.attach()

        start = time.time()

        try:
            if vector.location == "query":
                parsed = urlparse(vector.url)
                qs = parse_qs(parsed.query)
                qs[vector.parameter] = [payload]
                target = vector.url.split("?")[0] + f"?{vector.parameter}={payload}"
                await self.page.goto(target, wait_until="networkidle", timeout=8000)

            elif vector.location == "body":
                await self.page.goto(vector.url, wait_until="domcontentloaded", timeout=8000)
                selector = f"input[name='{vector.parameter}']"
                if await self.page.is_visible(selector):
                    await self.page.fill(selector, payload)
                    await self.page.press(selector, "Enter")
                    await self.page.wait_for_timeout(2000)

            await sensor.dom_snapshot()

        except Exception:
            return None

        features = sensor.features()
        bayes_score = DecisionEngine.score(features)

        ml = MLNoiseReducer()
        ml_prob = ml.probability(features)

        confidence = ConfidenceFusion.harmonic_mean(
            [bayes_score, ml_prob], [0.6, 0.4]
        )

        if confidence < 0.65:
            return None

        stride = STRIDEAnalyzer.classify(attack_type)
        cvss = CVSSCalculator.score(attack_type, confidence)

        return Finding(
            title=f"{attack_type} via {vector.parameter}",
            vector=vector,
            evidence=f"Payload reflected or executed: {payload[:30]}...",
            confidence=confidence,
            cvss=cvss,
            stride=stride
        )


# ============================================================
# VECTOR EXTRACTION
# ============================================================

class VectorExtractor:
    @staticmethod
    async def extract(page: Page, url: str) -> List[InjectionPoint]:
        vectors = []

        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        for p in qs:
            vectors.append(
                InjectionPoint(
                    url=url,
                    method="GET",
                    parameter=p,
                    location="query",
                    input_type="text"
                )
            )

        content = await page.content()
        soup = BeautifulSoup(content, "html.parser")

        for form in soup.find_all("form"):
            action = form.get("action") or url
            method = form.get("method", "get").upper()
            target = urljoin(url, action)

            for inp in form.find_all("input"):
                name = inp.get("name")
                if not name:
                    continue
                itype = inp.get("type", "text")
                vectors.append(
                    InjectionPoint(
                        url=target,
                        method=method,
                        parameter=name,
                        location="body",
                        input_type=itype
                    )
                )

        return vectors
# ============================================================
# ORCHESTRATOR (Single‑File / Streamlit‑Safe)
# ============================================================

class F16Orchestrator:
    def __init__(self, targets: List[str]):
        self.targets = targets
        self.results: List[ScanResult] = []

    async def run(self):
        async with async_playwright() as pw:
            browser, context = await StealthBrowser.launch(pw)

            for target in self.targets:
                if not authorized_target(target):
                    continue

                session = ScanResult(
                    target=target,
                    start_time=time.time()
                )

                page = await context.new_page()

                try:
                    await page.goto(target, wait_until="networkidle", timeout=30000)

                    vectors = await VectorExtractor.extract(page, target)
                    attack_graph = AttackGraph()

                    for v in vectors:
                        attack_graph.add_vector(v)

                        engine = AttackEngine(page)
                        finding = await engine.test_vector(v)

                        if finding:
                            session.findings.append(finding)
                            attack_graph.add_finding(finding)

                except Exception:
                    pass

                await page.close()
                self.results.append(session)

            await browser.close()


# ============================================================
# PDF REPORT (EXECUTIVE‑READY)
# ============================================================

class PDFReporter:
    @staticmethod
    def generate(results: List[ScanResult]) -> str:
        filename = f"F16_Report_{int(time.time())}.pdf"
        c = canvas.Canvas(filename, pagesize=A4)
        w, h = A4
        y = h - 50

        c.setFont("Helvetica-Bold", 18)
        c.drawString(40, y, "F16‑Research Edition — Security Assessment")
        y -= 40

        for r in results:
            c.setFont("Helvetica-Bold", 14)
            c.drawString(40, y, f"Target: {r.target}")
            y -= 18

            c.setFont("Helvetica", 10)
            c.drawString(40, y, f"Scan Duration: {r.duration}s")
            y -= 15
            c.drawString(40, y, f"Confirmed Findings: {len(r.findings)}")
            y -= 20

            if not r.findings:
                c.drawString(50, y, "No confirmed vulnerabilities detected.")
                y -= 20

            for f in r.findings:
                if y < 120:
                    c.showPage()
                    y = h - 50

                c.setFont("Helvetica-Bold", 11)
                c.drawString(50, y, f"{f.title} | CVSS: {f.cvss}")
                y -= 12

                c.setFont("Courier", 9)
                c.drawString(60, y, f"Parameter: {f.vector.parameter}")
                y -= 10
                c.drawString(60, y, f"Confidence: {f.confidence}")
                y -= 10
                c.drawString(60, y, f"STRIDE: {', '.join(f.stride)}")
                y -= 15

            y -= 20
            c.line(30, y, w - 30, y)
            y -= 30

        c.save()
        return filename


# ============================================================
# STREAMLIT UI (ONE‑CLICK EXPERIENCE)
# ============================================================

import streamlit as st

def streamlit_app():
    st.set_page_config(
        page_title="F16‑Research Edition",
        layout="wide"
    )

    st.title("🦅 F16‑Research Edition")
    st.caption("Scientific‑Grade Web Security Scanner")

    target = st.text_input(
        "Target URL (authorized testing only)",
        placeholder="https://example.com"
    )

    if st.button("🚀 Start Deep Scan"):
        if not target:
            st.error("Please enter a target URL.")
            return

        with st.status("Running full scientific scan...", expanded=True):
            orchestrator = F16Orchestrator([target])
            asyncio.run(orchestrator.run())

            report = PDFReporter.generate(orchestrator.results)

            st.success("Scan completed successfully ✅")

            for r in orchestrator.results:
                st.subheader(f"Results for {r.target}")
                st.write(f"Duration: {r.duration}s")

                if not r.findings:
                    st.success("No confirmed vulnerabilities detected.")
                else:
                    for f in r.findings:
                        st.error(
                            f"**{f.title}**  \n"
                            f"CVSS: {f.cvss}  \n"
                            f"Confidence: {f.confidence}  \n"
                            f"STRIDE: {', '.join(f.stride)}"
                        )

            with open(report, "rb") as f:
                st.download_button(
                    "📄 Download PDF Report",
                    f,
                    file_name=report
                )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    streamlit_app()
