# ============================================================
# F16 Orchestrator — Unified Scan Controller
# ============================================================

import asyncio
from core.crawler import SmartCrawler
from core.analyzer import BayesianAnalyzer
from core.payloads import PayloadFactory
from core.sensors import SensorSuite
from core.oob import OOBClient
from modeling.attack_graph import AttackGraph
from modeling.stride import STRIDEAnalyzer


class F16Orchestrator:
    """
    العقل المركزي:
    - ينسق كل الوحدات
    - يمنع False Positives
    - ينتج Findings جاهزة للتقرير
    """

    def __init__(self, target: str):
        self.target = target
        self.crawler = SmartCrawler(target)
        self.analyzer = BayesianAnalyzer()
        self.payload_factory = PayloadFactory()
        self.attack_graph = AttackGraph()
        self.oob = OOBClient()
        self.findings = []

    async def run(self):
        vectors = await self.crawler.discover()

        for vector in vectors:
            self.attack_graph.add_vector(vector)
            payloads = self.payload_factory.for_vector(vector)

            for payload in payloads:
                result = await self._test_payload(vector, payload)
                if result:
                    self._register_finding(vector, result)

        return self._finalize()

    async def _test_payload(self, vector, payload):
        """
        اختبار محافظ + Evidence‑based
        """
        page = await vector.fire(payload)
        sensor = SensorSuite(page)
        await sensor.attach()

        features = sensor.features()
        confidence = self.analyzer.score(features)

        if confidence > 0.75:
            return {
                "payload": payload,
                "confidence": confidence,
                "features": features
            }
        return None

    def _register_finding(self, vector, result):
        title = vector.category
        stride = STRIDEAnalyzer.classify(title)

        finding = {
            "title": title,
            "vector": vector,
            "confidence": result["confidence"],
            "stride": stride,
            "evidence": result["features"]
        }

        self.findings.append(finding)
        self.attack_graph.add_vulnerability(finding)

    def _finalize(self):
        """
        إخراج نهائي نظيف
        """
        return {
            "target": self.target,
            "findings": self.findings,
            "critical_paths": self.attack_graph.critical_paths()
        }
