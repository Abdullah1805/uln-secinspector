import streamlit as st
import asyncio
import aiohttp
import re
import pandas as pd
import validators
from urllib.parse import urljoin, urlparse, urlunparse
from bs4 import BeautifulSoup
from datetime import datetime
import json
import random

# =========================================================
# Utilities
# =========================================================
def mask_secret(s, show=4):
    if len(s) <= show * 2:
        return "***"
    return f"{s[:show]}{'*' * (len(s)-show*2)}{s[-show:]}"

def clean_url(url):
    try:
        parsed = urlparse(url)
        return urlunparse((parsed.scheme, parsed.netloc.lower(), parsed.path, '', '', ''))
    except: return url

# =========================================================
# Long-Term Memory (LTM)
# =========================================================
memory_db = {}  # Could be replaced by SQLite/Redis for persistence

def check_long_term_memory(finding):
    secret_hash = hash(finding['evidence'] + finding['location'] + finding['secret_type'])
    
    if secret_hash in memory_db:
        past = memory_db[secret_hash]
        finding['duplicate'] = True
        # Adjust decision for duplicates
        if finding['decision'] == "Send Now":
            finding['decision'] = "Send Carefully"
        elif finding['decision'] == "Send Carefully":
            finding['decision'] = "Hold"
        past['last_seen'] = datetime.utcnow().isoformat()
        past['occurrences'] += 1
        past['decision_history'].append(finding['decision'])
        past['platform_reports'].append(finding.get('platform', 'generic'))
    else:
        memory_db[secret_hash] = {
            "first_seen": datetime.utcnow().isoformat(),
            "last_seen": datetime.utcnow().isoformat(),
            "occurrences": 1,
            "context": [finding['context']],
            "decision_history": [finding['decision']],
            "platform_reports": [finding.get('platform', 'generic')]
        }
    return finding

# =========================================================
# Evidence Attachment
# =========================================================
def attach_evidence(finding):
    evidence = {
        "file": finding['location'],
        "snippet": mask_secret(finding['evidence']),
        "permissions": finding.get("permissions", []),
        "temporal": finding.get("temporal", ""),
        "context": finding.get("context", ""),
        "poc": finding.get("poc", "Safe verification only")
    }
    if "screenshots" in finding:
        evidence["screenshots"] = finding["screenshots"]
    return evidence

# =========================================================
# Platform Mode Profiles
# =========================================================
def platform_profile(platform):
    return {
        "hackerone": {"max_summary_lines":4, "allow_safe_poc":True, "tone":"assertive", "impact_style":"concise"},
        "bugcrowd": {"max_summary_lines":6, "allow_safe_poc":False, "tone":"collaborative", "impact_style":"explanatory"},
        "generic": {"max_summary_lines":5, "allow_safe_poc":True, "tone":"neutral", "impact_style":"balanced"}
    }[platform]

# =========================================================
# Report Generator
# =========================================================
def generate_report(finding, platform="hackerone"):
    profile = platform_profile(platform)
    title = f"{finding['secret_type']} exposure in {finding['context']}"
    severity = "Critical" if finding["impact"]=="Critical" else "High"
    poc_text = finding.get("poc","No active exploitation was performed.") if profile["allow_safe_poc"] else "No PoC provided."
    attachments = [attach_evidence(finding)]
    memory_notes = ""
    if finding.get("duplicate"):
        memory_notes = f"- Duplicate detected. First seen: {memory_db[hash(finding['evidence'] + finding['location'] + finding['secret_type'])]['first_seen']}, Occurrences: {memory_db[hash(finding['evidence'] + finding['location'] + finding['secret_type'])]['occurrences']}"
    
    report_body = f"""# {title}

## Summary
{finding.get('summary', 'No summary provided.')}

## Location
- {finding['location']}

## Evidence & Attachments
- Snippet: `{mask_secret(finding['evidence'])}`
- Permissions: {", ".join(finding.get("permissions", []))}
- Temporal: {finding.get("temporal", "")}
- Context: {finding.get("context", "")}
- PoC: {poc_text}
{memory_notes}

## Impact
{finding.get('impact_description', 'Potential unauthorized access.')}

## Remediation
{finding.get('remediation', 'Rotate credentials and restrict permissions.')}

---
Report generated automatically on {datetime.utcnow().isoformat()} UTC
"""
    return {"title": title, "severity": severity, "body": report_body, "attachments": attachments}

# =========================================================
# Auditor Engine
# =========================================================
class SovereignAuditorV10:
    def __init__(self, target_url, max_pages=50):
        self.target_url = clean_url(target_url)
        parsed_target = urlparse(self.target_url)
        self.base_domain = ".".join(parsed_target.netloc.split('.')[-2:])
        self.findings = []
        self.visited = set()
        self.queue = asyncio.Queue()
        self.max_pages = max_pages
        self.url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    async def extract_links(self, text, base_url):
        links = set()
        soup = BeautifulSoup(text, 'lxml' if 'lxml' in globals() else 'html.parser')
        for tag in soup.find_all(['a','script','link'], href=True, src=True):
            links.add(urljoin(base_url, tag.get('href') or tag.get('src')))
        links.update(re.findall(self.url_pattern, text))
        return links

    async def worker(self, session):
        while True:
            try:
                url, depth = await self.queue.get()
                if url in self.visited or len(self.visited)>=self.max_pages:
                    self.queue.task_done()
                    continue
                self.visited.add(url)
                await asyncio.sleep(random.uniform(0.5,1.2))
                async with session.get(url, timeout=12) as resp:
                    if resp.status!=200:
                        self.queue.task_done()
                        continue
                    content_type = resp.headers.get('Content-Type','').lower()
                    if 'image' in content_type or 'pdf' in content_type:
                        self.queue.task_done()
                        continue
                    raw_data = await resp.read()
                    text = raw_data.decode('utf-8', errors='replace')
                    # Phase-2 Patterns
                    patterns = {
                        "Critical: Private Key": r"-----BEGIN .* PRIVATE KEY-----",
                        "High: API Token": r"(?i)(api_key|client_secret|access_token)['\"]?\s*[:=]\s*['\"]([a-zA-Z0-9\-_]{16,})",
                        "High: Firebase URL": r"https://[a-z0-9.-]+\.firebaseio\.com"
                    }
                    for name,p in patterns.items():
                        for m in re.finditer(p, text):
                            finding = {
                                "Type": name,
                                "Evidence": m.group(0)[:50],
                                "URL": url,
                                "secret_type": name,
                                "location": url,
                                "context": "Production JS",
                                "impact": "High" if "High" in name else "Critical",
                                "decision": "Send Now",
                                "summary": f"Detected {name} in {url}",
                                "impact_description": "Potential unauthorized access.",
                                "remediation": "Rotate credentials and restrict permissions.",
                                "poc": "Verified format and presence only."
                            }
                            finding = check_long_term_memory(finding)
                            self.findings.append(finding)
                    # Crawl links
                    if depth<2:
                        links = await self.extract_links(text, url)
                        for link in links:
                            full_url = clean_url(link)
                            if self.base_domain in urlparse(full_url).netloc:
                                await self.queue.put((full_url, depth+1))
            except:
                pass
            finally:
                self.queue.task_done()

    async def run(self):
        await self.queue.put((self.target_url,0))
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=5, ssl=False)) as session:
            workers = [asyncio.create_task(self.worker(session)) for _ in range(5)]
            try:
                await asyncio.wait_for(self.queue.join(), timeout=300)
            except:
                pass
            finally:
                for w in workers: w.cancel()

# =========================================================
# Streamlit Dashboard
# =========================================================
st.set_page_config(page_title="Sovereign Auditor V10-Full", layout="wide")
st.title("🛡️ Sovereign Auditor V10-Full")

target = st.text_input("أدخل الرابط (سيتم فحص النطاقات الفرعية):", "https://example.com")
max_p = st.sidebar.number_input("حد الصفحات الأقصى", 10, 500, 50)
platform = st.selectbox("اختر المنصة:", ["hackerone","bugcrowd","generic"])
export_format = st.selectbox("صيغة التصدير:", ["markdown","json","platform"])

if st.button("🚀 إطلاق التدقيق الشامل"):
    if validators.url(target):
        auditor = SovereignAuditorV10(target, max_pages=max_p)
        with st.spinner("جاري فحص النطاق والروابط..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(auditor.run())
            loop.close()

        reports = []
        for f in auditor.findings:
            r = generate_report(f, platform)
            reports.append(r)

        if reports:
            # Export
            if export_format=="markdown":
                content = "\n\n".join([r["body"] for r in reports])
            elif export_format=="json":
                content = json.dumps(reports, indent=2)
            else:
                content = "\n\n".join([r["body"] for r in reports])  # platform template can be extended
            st.download_button("⬇️ تحميل التقرير", data=content, file_name=f"report.{export_format}")
            st.success(f"تم اكتشاف {len(reports)} تسريب محتمل، التقرير جاهز للتحميل.")
        else:
            st.success("لم يتم العثور على تسريبات.")
    else:
        st.error("الرابط غير صحيح.")
