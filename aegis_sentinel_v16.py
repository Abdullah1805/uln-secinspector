"""
# ==============================================================================
# AEGIS-X: SENTINEL v16.5 (Ghost Protocol Edition)
# World-Class Automated Security Framework
# ==============================================================================
#
# [GHOST PROTOCOL UPGRADES]
# 1. Stealth Integration: Uses playwright-stealth to mask WebDriver signals.
# 2. Behavioral Biometrics: Gaussian-distributed typing delays and mouse jitter.
# 3. Fingerprint Randomization: Spoofs Hardware Concurrency, Memory, and User-Agent.
# 4. Intelligent Auth: Auto-extracts Bearer/Session tokens from LocalStorage/Cookies.
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
# 0. AUTO-SETUP (Streamlit Cloud Compatible)
# ==========================================
def setup_env():
    """
    Ensures Playwright, Stealth, and Chromium are installed.
    """
    lock_file = "playwright_v16_5.lock"
    if os.path.exists(lock_file): return

    print("⚙️ [Auto-Setup] Initializing Ghost Protocol Environment...")
    
    packages = ["playwright", "playwright-stealth"]
    for pkg in packages:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            print(f"   [+] Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

    print("   [+] Installing Chromium browser...")
    try:
        subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
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
import time
import base64
import aiodns
import ipaddress
import psutil
import traceback
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
from playwright_stealth import stealth_async

# Reporting
from reportlab.lib.pagesizes import LETTER
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# ==========================================
# 1. CONFIGURATION & IDENTITY
# ==========================================
st.set_page_config(page_title="Aegis-X: Ghost Protocol", layout="wide", page_icon="👻")

class SecurityConfig:
    ADMIN_PASSWORD = "AegisX_Sentinel_2026"
    BOUNTY_IDENTITY = "Abdullah1805" 
    
    HEADERS = {
        "User-Agent": "Aegis-X/Sentinel v16.5 (Ghost Protocol)",
        "X-Hackerone": BOUNTY_IDENTITY,
        "Accept": "*/*"
    }
    
    # Ghost Protocol: High-Reputation User Agents
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0"
    ]

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
# 3. DATABASE
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
        self.engine = create_async_engine("sqlite+aiosqlite:///aegis_v16_5.db", echo=False)
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
# 4. GHOST PROTOCOL (STEALTH ENGINE)
# ==========================================
class GhostEngine:
    """
    [STEALTH CORE] Handles browser fingerprinting, human-like interaction, and auth extraction.
    """
    @staticmethod
    async def create_stealth_context(p):
        # 1. Randomize Fingerprint
        user_agent = random.choice(SecurityConfig.USER_AGENTS)
        viewport = {"width": 1920, "height": 1080}
        locale = "en-US"
        timezone = "America/New_York"
        
        # 2. Launch Browser with Anti-Detection Args
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-infobars",
                "--disable-dev-shm-usage",
                "--disable-extensions",
                "--disable-gpu"
            ]
        )
        
        # 3. Create Context with Randomized Hardware Signals
        context = await browser.new_context(
            user_agent=user_agent,
            viewport=viewport,
            locale=locale,
            timezone_id=timezone,
            has_touch=False,
            is_mobile=False,
            device_scale_factor=1,
            color_scheme="dark",
            extra_http_headers={"X-Hackerone": SecurityConfig.BOUNTY_IDENTITY}
        )
        
        # 4. Inject Stealth Scripts (Hardware Concurrency, Memory, Webdriver)
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            Object.defineProperty(navigator, 'hardwareConcurrency', { get: () => """ + str(random.choice([4, 8, 16])) + """ });
            Object.defineProperty(navigator, 'deviceMemory', { get: () => """ + str(random.choice([4, 8, 16])) + """ });
            window.chrome = { runtime: {} };
        """)
        
        return browser, context

    @staticmethod
    async def human_type(page, selector, text):
        """Simulates human typing with Gaussian delays."""
        try:
            await page.hover(selector)
            await page.click(selector)
            for char in text:
                await page.keyboard.type(char)
                # Random delay between 50ms and 150ms
                delay = random.uniform(0.05, 0.15)
                await asyncio.sleep(delay)
        except: pass

    @staticmethod
    async def human_click(page, selector):
        """Simulates mouse jitter before clicking."""
        try:
            box = await page.locator(selector).bounding_box()
            if box:
                # Jitter movement
                start_x = box['x'] - random.randint(10, 50)
                start_y = box['y'] - random.randint(10, 50)
                await page.mouse.move(start_x, start_y)
                await asyncio.sleep(random.uniform(0.1, 0.3))
                await page.mouse.move(box['x'] + box['width']/2, box['y'] + box['height']/2)
                await asyncio.sleep(random.uniform(0.05, 0.1))
                await page.click(selector)
        except: pass

    @staticmethod
    async def extract_auth(page) -> str:
        """Intelligently extracts Bearer tokens or Session IDs."""
        try:
            await page.wait_for_load_state("networkidle")
            
            # 1. Get Storage
            cookies = await page.context.cookies()
            local_storage = await page.evaluate("() => JSON.stringify(localStorage)")
            session_storage = await page.evaluate("() => JSON.stringify(sessionStorage)")
            
            # 2. Heuristic Search
            candidates = []
            
            # Check Cookies
            for c in cookies:
                if any(k in c['name'].lower() for k in ['session', 'token', 'auth', 'id']):
                    candidates.append(f"{c['name']}={c['value']}")
            
            # Check Local/Session Storage
            all_storage = json.loads(local_storage)
            all_storage.update(json.loads(session_storage))
            
            for k, v in all_storage.items():
                if any(key in k.lower() for key in ['token', 'auth', 'bearer', 'jwt']):
                    # Clean value
                    clean_v = v.strip('"')
                    if "Bearer" not in clean_v:
                        candidates.append(f"Bearer {clean_v}")
                    else:
                        candidates.append(clean_v)
            
            if candidates:
                # Return the longest candidate (likely the JWT)
                return max(candidates, key=len)
            
            return None
        except: return None

# ==========================================
# 5. DEEP MINING & THE HARVESTER
# ==========================================
@dataclass
class InjectionPoint:
    url: str
    method: str
    params: Dict[str, str]
    context: str = "generic"
    priority: int = 1
    server_type: str = "Unknown"
    baseline_len: int = 0
    baseline_status: int = 200
    baseline_body: str = "" 

class TheHarvester:
    discovered_ids: Set[str] = set()
    @staticmethod
    def harvest(content: str):
        ids = re.findall(r'\b\d{4,10}\b', content)
        uuids = re.findall(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b', content)
        for i in ids: TheHarvester.discovered_ids.add(i)
        for u in uuids: TheHarvester.discovered_ids.add(u)

class DeepMiner:
    @staticmethod
    def calculate_priority(url: str, params: Dict[str, str]) -> int:
        score = 1
        high_value_keywords = ["admin", "login", "api", "v1", "v2", "billing", "profile", "account", "upload"]
        for kw in high_value_keywords:
            if kw in url.lower(): score += 2
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
                # [GHOST PROTOCOL] Use Stealth Context
                browser, context = await GhostEngine.create_stealth_context(p)
                
                # Inject Auth if provided
                if session_token:
                    await context.add_cookies([{
                        "name": "AEGIS_AUTH", "value": session_token, "url": url
                    }])

                page = await context.new_page()
                # Apply Stealth Scripts
                await stealth_async(page)
                
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
                
                try:
                    await page.goto(url, wait_until="networkidle", timeout=15000)
                    
                    # [GHOST PROTOCOL] Human-Like Interaction
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await asyncio.sleep(random.uniform(0.5, 1.5))
                    
                    # Try to find a login form to demonstrate typing
                    email_input = await page.query_selector("input[type='email'], input[name='email']")
                    if email_input:
                        await GhostEngine.human_type(page, "input[type='email']", "bwd400080@gmail.com")
                    
                except: pass
                await browser.close()
        except Exception as e:
            sys_logger.warning(f"Deep Mining Warning: {e}")
        return points

# ==========================================
# 6. SENTINEL ENGINE
# ==========================================
class PayloadFactory:
    @staticmethod
    def get_base_payloads(vuln_type: str) -> List[str]:
        if vuln_type == "XSS":
            trigger = "window.AEGIS_EXECUTED()"
            return [f"<script>{trigger}</script>", f"\"><img src=x onerror={trigger}>"]
        elif vuln_type == "SQLI_TIME":
            return ["1' WAITFOR DELAY '0:0:5'--", "1' AND SLEEP(5)--"]
        elif vuln_type == "LFI":
            return ["../../../../etc/passwd"]
        return []

class SentinelEngine:
    def __init__(self, db: StateManager, limiter: Any, watchdog: Any, tokens: Dict[str, str]):
        self.db = db
        self.limiter = limiter
        self.watchdog = watchdog
        self.tokens = tokens
        
        self.clients = {}
        for role, token in tokens.items():
            headers = SecurityConfig.HEADERS.copy()
            if token:
                if "Bearer" in token: headers["Authorization"] = token
                else: headers["Cookie"] = token
            self.clients[role] = httpx.AsyncClient(verify=False, timeout=10.0, follow_redirects=True, headers=headers)

        self.validator_queue = asyncio.Queue()

    async def scan_target(self, target_url: str, stop_event: asyncio.Event):
        try:
            sys_logger.info("Phase 1: Ghost Protocol Mining...")
            await self.limiter.acquire()
            
            discovery_role = "User" if self.tokens.get("User") else "Guest"
            res = await self.clients[discovery_role].get(target_url)
            
            points = DeepMiner.mine_static(target_url, res.text)
            dynamic_points = await DeepMiner.mine_dynamic(target_url, self.tokens.get(discovery_role))
            points.extend(dynamic_points)
            
            points.sort(key=lambda x: x.priority, reverse=True)
            sys_logger.info(f"Discovery: {len(points)} endpoints. Harvested IDs: {len(TheHarvester.discovered_ids)}")

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
        except: pass

    async def _fuzz_injection(self, point: InjectionPoint):
        vectors = ["SQLI_TIME", "LFI", "XSS"]
        client = self.clients.get("User") or self.clients.get("Guest")
        
        for v_type in vectors:
            payloads = PayloadFactory.get_base_payloads(v_type)
            for payload in payloads:
                await self.watchdog.is_safe.wait()
                await self.limiter.acquire()
                if await self._inject(client, point, payload, v_type): break

        if await self._inject(client, point, "AEGIS_PROBE", "Probe", check_reflection=True):
            await self.validator_queue.put(point)

    async def _inject(self, client, point: InjectionPoint, payload: str, v_type: str, check_reflection=False) -> bool:
        try:
            params = point.params.copy()
            target_key = list(params.keys())[0]
            params[target_key] = payload
            
            atk_logger.info(f"[{v_type}] {point.url}")
            start_t = time.time()
            
            try:
                if point.method == "GET": res = await client.get(point.url, params=params)
                else: res = await client.post(point.url, data=params)
                elapsed = time.time() - start_t
            except httpx.TimeoutException:
                elapsed = 10.0
                res = None

            if res is None: return False

            if v_type == "SQLI_TIME" and "sleep" in unquote(payload).lower():
                if elapsed > 4.5:
                    await self.db.save_finding(
                        url=point.url, vuln_type=v_type, severity="Critical",
                        payload=payload, mutation_used="Time-Based",
                        evidence=f"Delay: {elapsed:.2f}s",
                        curl_command=f"curl '{point.url}'",
                        remediation="Sanitize inputs.", screenshot_b64=None
                    )
                    return True

            if check_reflection and payload in res.text: return True
            return False
        except: return False

    async def close(self): 
        for c in self.clients.values(): await c.aclose()

# ==========================================
# 7. VALIDATOR
# ==========================================
class XSSValidator:
    def __init__(self, db: StateManager, watchdog: Any):
        self.db = db
        self.watchdog = watchdog

    async def worker(self, queue: asyncio.Queue, stop_event: asyncio.Event):
        try:
            async with async_playwright() as p:
                # [GHOST PROTOCOL] Stealth Validator
                browser, context = await GhostEngine.create_stealth_context(p)
                
                while not stop_event.is_set():
                    try:
                        try:
                            point = await asyncio.wait_for(queue.get(), timeout=1.0)
                        except asyncio.TimeoutError: continue

                        await self.watchdog.is_safe.wait()
                        page = await context.new_page()
                        await stealth_async(page)
                        
                        vuln_confirmed = False
                        async def on_exec():
                            nonlocal vuln_confirmed
                            vuln_confirmed = True
                        await page.expose_function("AEGIS_EXECUTED", on_exec)
                        
                        payloads = [f"<script>window.AEGIS_EXECUTED()</script>"]
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
                                        payload=payload, mutation_used="Ghost Validation",
                                        evidence="JS Hook Triggered",
                                        curl_command=f"curl '{point.url}?{qs}'",
                                        remediation="Encoding", screenshot_b64=b64_img
                                    )
                                    break
                            except: pass
                        await page.close()
                        queue.task_done()
                    except Exception: pass
                await browser.close()
        except Exception: pass

# ==========================================
# 8. REPORTING
# ==========================================
class ProfessionalReporter:
    @staticmethod
    def generate_pdf(findings: List[Finding]) -> bytes:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=LETTER)
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph("Aegis-X Sentinel v16.5", styles['Title']))
        elements.append(Paragraph("Ghost Protocol Assessment Report", styles['Heading2']))
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

        sys_logger.info("Phase 1: Ghost Protocol Initialized...")
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
        st.title("🔒 Aegis-X Sentinel v16.5")
        if st.text_input("Access Key", type="password") == SecurityConfig.ADMIN_PASSWORD:
            if st.button("Unlock System"):
                st.session_state['auth'] = True
                st.rerun()
        return

    st.title("👻 Aegis-X: Sentinel v16.5 (Ghost Protocol)")
    st.caption(f"Identity: {SecurityConfig.BOUNTY_IDENTITY} | Stealth: Active | Biometrics: Human-Like")
    
    with st.sidebar:
        st.header("Mission Control")
        target = st.text_input("Target URL", "http://testphp.vulnweb.com/listproducts.php?cat=1")
        
        st.subheader("Identity Management")
        token_user = st.text_input("User Token", placeholder="Required for Auth Tests")
        
        rps = st.slider("Intensity (RPS)", 1, 50, 10)
        
        if st.button("🚀 Initiate Ghost Scan"):
            if 'thread' not in st.session_state:
                tokens = {"User": token_user}
                t = TaskOrchestrator(target, rps, tokens)
                t.start()
                st.session_state['thread'] = t
                st.toast("Ghost Protocol Engaged")
        
        if st.button("🛑 Abort"):
            if 'thread' in st.session_state:
                st.session_state['thread'].stop()
                st.session_state['thread'].join()
                del st.session_state['thread']
                st.toast("Engine Stopped")

        st.divider()
        if st.button("📄 Download Ghost Report"):
            async def get_data():
                db = StateManager()
                return await db.get_all_findings()
            findings = asyncio.run(get_data())
            pdf = ProfessionalReporter.generate_pdf(findings)
            st.download_button("Download PDF", pdf, "Aegis_v16_5_Report.pdf", "application/pdf")

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
