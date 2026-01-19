# ============================================================
# OOB Detection Hook (Collaborator‑style)
# ============================================================

import time


class OOBClient:
    """
    Hook جاهز لـ:
    - Interactsh
    - Burp Collaborator
    """

    def __init__(self, endpoint: str | None = None):
        self.endpoint = endpoint
        self.interactions = []

    def payload(self) -> str | None:
        if not self.endpoint:
            return None
        return f"http://{self.endpoint}/f16-{int(time.time())}"

    def check(self):
        """
        هنا يتم ربط API الحقيقي لاحقًا
        """
        return self.interactions
