# ============================================================
# Attack Execution Engine
# ============================================================

import asyncio
from core.decision import DecisionEngine
from core.ml import MLNoiseReducer
from attack.payloads import PayloadEngine


class AttackEngine:
    """
    حلقة:
    Inject → Sense → Decide
    """

    def __init__(self, page):
        self.page = page
        self.ml = MLNoiseReducer()

    async def test_vector(self, vector, sensors) -> dict | None:
        payloads = PayloadEngine.select(vector.input_type)

        for attack_type, plist in payloads.items():
            for payload in plist:
                try:
                    await self._inject(vector, payload)
                    await asyncio.sleep(1)

                    features = sensors.features()

                    bayes = DecisionEngine.score(features)
                    ml_score = self.ml.probability(features)
                    confidence = DecisionEngine.confidence(bayes, ml_score)

                    if confidence > 0.75:
                        return {
                            "title": f"{attack_type} on {vector.parameter}",
                            "vector": vector,
                            "payload": payload,
                            "confidence": confidence,
                            "features": features
                        }

                except Exception:
                    continue

        return None

    async def _inject(self, vector, payload: str):
        """
        تنفيذ فعلي حسب نوع الـ vector.
        """
        if vector.location == "query":
            target = f"{vector.url.split('?')[0]}?{vector.parameter}={payload}"
            await self.page.goto(target, wait_until="networkidle", timeout=15000)

        elif vector.location == "body":
            await self.page.goto(vector.url, wait_until="domcontentloaded", timeout=15000)
            selector = f"input[name='{vector.parameter}']"
            if await self.page.is_visible(selector):
                await self.page.fill(selector, payload)
                await self.page.press(selector, "Enter")
