import aiohttp, asyncio, time, re
from urllib.parse import urlparse, urljoin, parse_qs

XSS_PAYLOAD = "<svg/onload=confirm(1)>"
REDIRECT_PAYLOAD = "https://evil.com"

COMMON_ENDPOINTS = [
    "/", "/search", "/api/user", "/profile", "/redirect", "/view"
]


class SovereignBBEngine:
    def __init__(self, base_scope, concurrency=15):
        self.base = base_scope.rstrip("/")
        self.domain = urlparse(self.base).netloc
        self.sem = asyncio.Semaphore(concurrency)

    def in_scope(self, url):
        p = urlparse(url)
        return p.scheme in ("http", "https") and p.netloc == self.domain

    async def run(self):
        timeout = aiohttp.ClientTimeout(total=20)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            self.s = s
            endpoints = await self.discover()
            findings = []

            for f in endpoints:
                checks = [
                    self.check_sqli(f),
                    self.check_xss(f),
                    self.check_redirect(f),
                    self.check_idor(f)
                ]
                for c in checks:
                    r = await c
                    if r:
                        findings.append(r)

            return findings

    async def discover(self):
        return [urljoin(self.base + "/", e) for e in COMMON_ENDPOINTS if self.in_scope(urljoin(self.base + "/", e))]

    # ---------- SQLi ----------
    async def check_sqli(self, url):
        params = parse_qs(urlparse(url).query)
        for p in params or ["id"]:
            b = await self.timed(url, {p: "1"})
            i = await self.timed(url, {p: "1' AND SLEEP(5)--"})
            if i - b >= 4:
                return self.finding(url, p, "Time-Based SQL Injection", 95,
                                    {"baseline": b, "injected": i})
        return None

    # ---------- XSS ----------
    async def check_xss(self, url):
        for p in ["q", "search", "input"]:
            try:
                async with self.s.get(url, params={p: XSS_PAYLOAD}) as r:
                    text = await r.text()
                    if XSS_PAYLOAD in text:
                        return self.finding(url, p, "Reflected XSS", 90)
            except:
                pass
        return None

    # ---------- Open Redirect ----------
    async def check_redirect(self, url):
        for p in ["next", "url", "redirect"]:
            try:
                async with self.s.get(url, params={p: REDIRECT_PAYLOAD}, allow_redirects=False) as r:
                    if r.status in (301, 302) and REDIRECT_PAYLOAD in r.headers.get("Location", ""):
                        return self.finding(url, p, "Open Redirect", 92)
            except:
                pass
        return None

    # ---------- IDOR ----------
    async def check_idor(self, url):
        for p in ["id", "user_id"]:
            try:
                async with self.s.get(url, params={p: "1"}) as r1:
                    async with self.s.get(url, params={p: "2"}) as r2:
                        if r1.status == 200 and r2.status == 200:
                            if await r1.text() != await r2.text():
                                return self.finding(url, p, "Potential IDOR", 80)
            except:
                pass
        return None

    async def timed(self, url, params):
        start = time.time()
        try:
            async with self.s.get(url, params=params):
                pass
        except:
            return 0
        return time.time() - start

    def finding(self, url, param, impact, confidence, evidence=None):
        return {
            "url": url,
            "param": param,
            "impact": impact,
            "confidence": confidence,
            "evidence": evidence or {}
        }
