# ============================================================
# Sovereign Scanner v7.7 PRO - Streamlit Edition
# Passive Bug Bounty Scanner (Secrets Detection)
# ============================================================

import streamlit as st
import asyncio
import aiohttp
import random
import re
from urllib.parse import urljoin, urlparse
from datetime import datetime
from bs4 import BeautifulSoup
from fpdf import FPDF

# ============================================================
# CONFIG
# ============================================================

MAX_PAGES = 25
REQUEST_DELAY = (1.5, 3.0)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_1 like Mac OS X)"
]

SECRETS_PATTERNS = {
    "AWS Access Key": r"AKIA[0-9A-Z]{16}",
    "Google API Key": r"AIza[0-9A-Za-z-_]{35}",
    "GitHub Token": r"ghp_[0-9A-Za-z]{36}",
    "Slack Token": r"xox[baprs]-[0-9a-zA-Z]{10,48}",
    "Generic Secret": r"(?i)(api_key|secret|token|password)\s*[:=]\s*['\"]([0-9a-zA-Z-_]{10,})['\"]"
}

# ============================================================
# ENGINE
# ============================================================

class Scanner:
    def __init__(self, target):
        self.target = target
        self.domain = urlparse(target).netloc
        self.visited = set()
        self.findings = []
        self.sem = asyncio.Semaphore(5)

    def allowed(self, url):
        return urlparse(url).netloc == self.domain

    def headers(self):
        return {"User-Agent": random.choice(USER_AGENTS)}

    async def fetch(self, session, url):
        async with self.sem:
            await asyncio.sleep(random.uniform(*REQUEST_DELAY))
            async with session.get(url, headers=self.headers(), timeout=15) as r:
                return await r.text(errors="ignore")

    def extract_links(self, base, html):
        soup = BeautifulSoup(html, "html.parser")
        links = set()
        for tag in soup.find_all(["a", "script"]):
            src = tag.get("href") or tag.get("src")
            if src:
                full = urljoin(base, src)
                if self.allowed(full):
                    links.add(full)
        return links

    def find_secrets(self, url, text):
        for name, pattern in SECRETS_PATTERNS.items():
            for match in re.findall(pattern, text):
                value = match[1] if isinstance(match, tuple) else match
                self.findings.append({
                    "Service": name,
                    "URL": url,
                    "Evidence": value[:6] + "..." + value[-4:],
                    "Severity": "HIGH"
                })

    async def crawl(self):
        queue = [self.target]
        async with aiohttp.ClientSession() as session:
            while queue and len(self.visited) < MAX_PAGES:
                url = queue.pop(0)
                if url in self.visited:
                    continue
                self.visited.add(url)

                try:
                    html = await self.fetch(session, url)
                except Exception:
                    continue

                self.find_secrets(url, html)
                queue.extend(self.extract_links(url, html))

# ============================================================
# PDF REPORT
# ============================================================

class Report(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "Sovereign Scanner v7.7 PRO Report", ln=1, align="C")

def generate_pdf(findings, target):
    pdf = Report()
    pdf.add_page()
    pdf.set_font("Arial", size=11)

    pdf.multi_cell(0, 8, f"""
Target: {target}
Scan Date: {datetime.utcnow()} UTC
Findings Count: {len(findings)}

Passive Bug Bounty Scanner – No exploitation performed.
""")

    pdf.ln(4)
    for i, f in enumerate(findings, 1):
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 8, f"{i}. {f['Service']} (HIGH)", ln=1)
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 7, f"URL: {f['URL']}\nEvidence: {f['Evidence']}")
        pdf.ln(2)

    return pdf.output(dest="S").encode("latin-1")

# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(page_title="Sovereign Scanner v7.7 PRO", layout="wide")
st.title("🛡️ Sovereign Scanner v7.7 PRO")
st.caption("Passive Bug Bounty Scanner – Secrets Detection Only")

target = st.text_input("🎯 Target URL (authorized only)", placeholder="https://example.com")

if st.button("🚀 Start Scan") and target:
    with st.spinner("Scanning target safely..."):
        scanner = Scanner(target)
        asyncio.run(scanner.crawl())
        st.session_state.results = scanner.findings
        st.success(f"Scan completed – Findings: {len(scanner.findings)}")

if "results" in st.session_state:
    st.subheader("🔍 Findings")
    st.dataframe(st.session_state.results, use_container_width=True)

    pdf = generate_pdf(st.session_state.results, target)
    st.download_button(
        "📄 Download PDF Report",
        data=pdf,
        file_name="sovereign_report.pdf",
        mime="application/pdf"
    )
