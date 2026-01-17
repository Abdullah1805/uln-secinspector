import aiohttp, asyncio
from urllib.parse import urljoin

COMMON_ENDPOINTS = [
    "/api/user", "/api/profile", "/api/orders",
    "/admin", "/dashboard", "/graphql"
]

COMMON_PARAMS = ["id", "user_id", "account", "uid"]

class ReconEngine:
    def __init__(self, concurrency):
        self.sem = asyncio.Semaphore(concurrency)

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
                                        "type": "PARAM_TEST",
                                        "url": url,
                                        "param": p
                                    })
                    except:
                        pass
        return findings
