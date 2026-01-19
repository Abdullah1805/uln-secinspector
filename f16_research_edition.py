# ============================================================
# F16‑Research Edition
# Scientific‑Grade Web Security Analysis Framework
# ============================================================
# ⚠️ AUTHORIZED SECURITY TESTING ONLY
# Bug Bounty • Pentesting • Academic Research
# ============================================================

import asyncio
import time
import math
import re
import statistics
from typing import List, Dict, Optional, Tuple
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
# SAFETY & SELF‑GUARD
# ============================================================

def authorized_target(url: str) -> bool:
    """
    Basic sanity guard.
    Prevents obvious misuse (localhost, private IPs).
    """
    parsed = urlparse(url)
    if not parsed.scheme.startswith("http"):
        return False

    forbidden = (
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "::1"
    )
    return not any(f in parsed.netloc for f in forbidden)


# ============================================================
# DATA MODELS (Research‑Grade)
# ============================================================

@dataclass(frozen=True)
class InjectionPoint:
    url: str
    method: str
    parameter: str
    location: str        # query / body / dom
    input_type: str      # text / hidden / js / unknown


@dataclass
class Finding:
    title: str
    vector: InjectionPoint
    evidence: str
    confidence: float
    cvss: float
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
    """
    Converts behavioral signals into normalized numerical features.
    """

    @staticmethod
    def normalize(value: float, max_value: float = 1.0) -> float:
        return min(max(value / max_value, 0.0), 1.0)

    @staticmethod
    def timing_shift(baseline: List[float], current: float) -> float:
        if not baseline:
            return 0.0
        med = statistics.median(baseline)
        return FeatureExtractor.normalize(abs(current - med), med + 0.001)


# ============================================================
# CONFIDENCE FUSION (Bayesian‑Inspired, Conservative)
# ============================================================

class ConfidenceFusion:
    """
    Avoids optimistic max().
    Uses weighted harmonic mean to reduce bias.
    """

    @staticmethod
    def harmonic_mean(values: List[float], weights: List[float]) -> float:
        numerator = sum(weights)
        denominator = sum(
            (w / (v + 1e-6)) for v, w in zip(values, weights)
        )
        return round(numerator / denominator, 3)


# ============================================================
# DECISION ENGINE (NO FAKE AI)
# ============================================================

class DecisionEngine:
    """
    Scientific decision scoring based on multiple orthogonal signals.
    """

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
# ============================================================
# MACHINE LEARNING ENGINE (Noise Reduction)
# ============================================================

class MLNoiseReducer:
    """
    Trained on synthetic + heuristic signals.
    الهدف: تقليل الضجيج وليس "التنبؤ بالثغرات".
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=120,
            max_depth=8,
            random_state=42
        )
        self._train_bootstrap()

    def _train_bootstrap(self):
        """
        تدريب أولي (Bootstrapped).
        في الأنظمة العملاقة يتم استبداله ببيانات حقيقية.
        """
        X = []
        y = []

        for _ in range(300):
            dom = np.random.rand()
            event = np.random.rand()
            timing = np.random.rand()
            status = np.random.rand()

            score = (
                dom * 0.3 +
                event * 0.35 +
                timing * 0.2 +
                status * 0.15
            )

            X.append([dom, event, timing, status])
            y.append(1 if score > 0.55 else 0)

        X = self.scaler.fit_transform(X)
        self.model.fit(X, y)

    def probability(self, features: Dict[str, float]) -> float:
        vec = [
            features.get("dom_change", 0.0),
            features.get("event_trigger", 0.0),
            features.get("timing_shift", 0.0),
            features.get("status_anomaly", 0.0),
        ]
        vec = self.scaler.transform([vec])
        return round(self.model.predict_proba(vec)[0][1], 3)


# ============================================================
# STRIDE THREAT MODELING (Automatic)
# ============================================================

class STRIDEAnalyzer:
    """
    Maps vulnerability patterns to STRIDE categories.
    """

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
# CVSS SCORING (Conservative)
# ============================================================

class CVSSCalculator:
    """
    ليس CVSS رسمي كامل، لكنه واقعي وغير متفائل.
    """

    BASE = {
        "XSS": 6.1,
        "SQLi": 9.1,
        "SSTI": 8.6,
        "RCE": 9.8,
        "IDOR": 7.2,
    }

    @staticmethod
    def score(title: str, confidence: float) -> float:
        for k, v in CVSSCalculator.BASE.items():
            if k.lower() in title.lower():
                return round(v * confidence, 1)
        return round(5.0 * confidence, 1)


# ============================================================
# ATTACK GRAPH (Research‑Grade)
# ============================================================

class AttackGraph:
    """
    Graph‑based reasoning similar to academic APT models.
    """

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_vector(self, vector: InjectionPoint):
        self.graph.add_node(
            f"VECTOR::{vector.parameter}",
            type="vector",
            location=vector.location
        )

    def add_finding(self, finding: Finding):
        vuln_node = f"VULN::{finding.title}"
        self.graph.add_node(
            vuln_node,
            type="vulnerability",
            cvss=finding.cvss
        )
        self.graph.add_edge(
            f"VECTOR::{finding.vector.parameter}",
            vuln_node
        )

    def critical_paths(self) -> List[str]:
        """
        يعيد أخطر المسارات (High Impact).
        """
        paths = []
        for node, data in self.graph.nodes(data=True):
            if data.get("type") == "vulnerability" and data.get("cvss", 0) >= 8.0:
                paths.append(node)
        return paths
# ============================================================
# STEALTH BROWSER ENGINE
# ============================================================

class StealthBrowser:
    """
    Browser fingerprinting + behavior masking.
    مستوحى من أبحاث:
    - NDSS
    - BlackHat WAF Evasion papers
    """

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
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
            timezone_id="UTC"
        )

        # Mask webdriver
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
            window.chrome = { runtime: {} };
        """)

        return browser, context
# ============================================================
# SENSOR SUITE
# ============================================================

class SensorSuite:
    def __init__(self, page):
        self.page = page
        self.events = {
            "console": 0,
            "dialog": 0,
            "dom_change": 0
        }
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
            "event_trigger": min(1.0, self.events["console"] + self.events["dialog"]),
            "dom_change": min(1.0, self.events["dom_change"] / 50000),
            "timing_shift": min(1.0, elapsed / 5.0),
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
        if input_type in ["email", "password"]:
            return {"XSS": PayloadEngine.PAYLOADS["XSS"]}
        return PayloadEngine.PAYLOADS
# ============================================================
# OOB DETECTION (HOOK)
# ============================================================

class OOBClient:
    """
    Stub for Interactsh / Burp Collaborator
    """

    def __init__(self, endpoint=None):
        self.endpoint = endpoint
        self.interactions = []

    def payload(self):
        if not self.endpoint:
            return None
        return f"http://{self.endpoint}/f16-{int(time.time())}"

    def check(self):
        # هنا يتم ربط API الحقيقي
        return self.interactions
# ============================================================
# DECISION ENGINE
# ============================================================

class DecisionEngine:

    @staticmethod
    def confidence(bayes: float, ml: float) -> float:
        return round((0.6 * bayes + 0.4 * ml), 3)
# ============================================================
# ATTACK GRAPH
# ============================================================

class AttackGraph:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_vector(self, vector):
        self.nodes.append({
            "type": "input",
            "name": vector.param_name,
            "location": vector.location
        })

    def add_vulnerability(self, vuln):
        self.nodes.append({
            "type": "vulnerability",
            "name": vuln.title,
            "severity": vuln.severity
        })
        self.edges.append({
            "from": vuln.vector.param_name,
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
            if "SQLi" in n["name"]:
                stride["Elevation of Privilege"].append(n["name"])
            if "XSS" in n["name"]:
                stride["Information Disclosure"].append(n["name"])
                stride["Spoofing"].append(n["name"])
            if "SSTI" in n["name"]:
                stride["Tampering"].append(n["name"])

        return stride
# ============================================================
# CVSS ENGINE
# ============================================================

class CVSS:
    @staticmethod
    def score(vuln_type: str):
        base = {
            "SQLi": 9.0,
            "SSTI": 8.5,
            "XSS": 6.5
        }
        return base.get(vuln_type, 5.0)
# ============================================================
# ORCHESTRATOR
# ============================================================

class F16ResearchEngine:
    def __init__(self, targets: list[str]):
        self.targets = targets
        self.sessions = []

    async def run(self):
        async with async_playwright() as pw:
            browser, context = await StealthBrowser.launch(pw)

            for target in self.targets:
                page = await context.new_page()
                session = {
                    "target": target,
                    "vectors": [],
                    "vulns": [],
                    "graph": AttackGraph()
                }

                try:
                    await page.goto(target, wait_until="networkidle", timeout=30000)

                    # 1. Extract vectors
                    vectors = await VectorExtractor.extract(page, target)
                    session["vectors"] = vectors

                    # 2. Test vectors
                    for v in vectors:
                        session["graph"].add_vector(v)

                        sensors = SensorSuite(page)
                        await sensors.attach()

                        engine = AttackEngine(page)
                        vuln = await engine.test_vector(v)

                        if vuln:
                            vuln.severity = CVSS.score(vuln.title.split()[0])
                            session["vulns"].append(vuln)
                            session["graph"].add_vulnerability(vuln)

                except Exception as e:
                    pass

                await page.close()
                self.sessions.append(session)

            await browser.close()
# ============================================================
# PDF REPORT
# ============================================================

class PDFReport:
    @staticmethod
    def generate(sessions):
        filename = f"F16_Report_{int(time.time())}.pdf"
        c = canvas.Canvas(filename, pagesize=A4)
        w, h = A4
        y = h - 50

        c.setFont("Helvetica-Bold", 18)
        c.drawString(40, y, "F16‑Research Edition — Security Assessment")
        y -= 40

        for s in sessions:
            c.setFont("Helvetica-Bold", 14)
            c.drawString(40, y, f"Target: {s['target']}")
            y -= 20

            c.setFont("Helvetica", 10)
            c.drawString(40, y, f"Vectors analyzed: {len(s['vectors'])}")
            y -= 15
            c.drawString(40, y, f"Confirmed vulnerabilities: {len(s['vulns'])}")
            y -= 20

            for v in s["vulns"]:
                if y < 100:
                    c.showPage()
                    y = h - 50

                c.setFont("Helvetica-Bold", 11)
                c.drawString(50, y, f"{v.title} | CVSS: {v.severity}")
                y -= 12

                c.setFont("Courier", 9)
                c.drawString(60, y, f"Vector: {v.vector.param_name}")
                y -= 12
                c.drawString(60, y, f"Payload: {v.payload}")
                y -= 12
                c.drawString(60, y, f"Evidence: {v.evidence}")
                y -= 18

            # STRIDE
            stride = s["graph"].to_stride()
            c.setFont("Helvetica-Bold", 12)
            c.drawString(40, y, "Threat Model (STRIDE):")
            y -= 15

            c.setFont("Helvetica", 9)
            for k, v in stride.items():
                if v:
                    c.drawString(50, y, f"{k}: {', '.join(v)}")
                    y -= 12

            y -= 30
            c.line(30, y, w-30, y)
            y -= 30

        c.save()
        print(f"[+] Report generated: {filename}")
# ============================================================
# MAIN
# ============================================================

async def main():
    targets = [
        "http://testphp.vulnweb.com/search.php?test=query"
    ]

    engine = F16ResearchEngine(targets)
    await engine.run()
    PDFReport.generate(engine.sessions)

if __name__ == "__main__":
    asyncio.run(main())
