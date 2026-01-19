# ============================================================
# Sensor Suite — Behavioral Evidence Collection
# ============================================================

import asyncio
import time


class SensorSuite:
    """
    لا نكتشف الثغرة من النص
    بل من السلوك الحقيقي للنظام
    """

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
        await self._snapshot_dom()

    def _on_console(self, msg):
        if "f16" in msg.text.lower():
            self.events["console"] += 1

    def _on_dialog(self, dialog):
        self.events["dialog"] += 1
        asyncio.create_task(dialog.dismiss())

    async def _snapshot_dom(self):
        content = await self.page.content()
        self.events["dom_change"] = len(content)

    def features(self):
        elapsed = time.time() - self.start_time
        return {
            "event_trigger": min(1.0, self.events["console"] + self.events["dialog"]),
            "dom_change": min(1.0, self.events["dom_change"] / 60000),
            "timing_shift": min(1.0, elapsed / 5.0),
            "status_anomaly": 0.4  # محفوظ، غير متفائل
        }
