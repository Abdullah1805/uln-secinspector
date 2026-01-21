============================================================

Market Spy Ω — Competitive Intelligence & Price Tracker

Business-Grade • High-Speed • Excel-Ready

Author: Abdullah Abbas

WhatsApp: +96407881296737

============================================================

import asyncio import aiohttp import re import csv import hashlib from urllib.parse import urljoin, urlparse from bs4 import BeautifulSoup import streamlit as st from datetime import datetime

============================================================

CONFIGURATION (التكوين التجاري)

============================================================

HEADERS = { "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) " "AppleWebKit/537.36 (KHTML, like Gecko) " "Chrome/120.0.0.0 Safari/537.36", "Accept-Language": "en-US,en;q=0.9,ar;q=0.8" }

CONCURRENT_REQUESTS = 5

PRICE_PATTERNS = [ r"$\s?[\d,]+(.\d{2})?", r"[\d,]+(.\d{2})?\s?USD", r"[\d,]+\s?IQD", r"€\s?[\d,]+(.\d{2})?", r"[\d,]+(.\d{2})?\s?د.ع" ]

STOCK_KEYWORDS = { "in_stock": ["in stock", "available", " متوفر", "متاح", "add to cart"], "out_of_stock": ["out of stock", "sold out", "نفذت الكمية", "غير متوفر", "coming soon"] }

============================================================

UTILITY FUNCTIONS

============================================================

async def fetch(session, url): try: async with session.get(url, headers=HEADERS, timeout=15) as response: if response.status == 200: return await response.text(errors='ignore') return None except Exception: return None

def extract_price(text): for pattern in PRICE_PATTERNS: match = re.search(pattern, text) if match: # تنظيف الرقم فقط return re.sub(r'[^\d.]+', '', match.group(0)) return "N/A"

def check_availability(text): text_lower = text.lower() for keyword in STOCK_KEYWORDS["out_of_stock"]: if keyword in text_lower: return "Out of Stock ❌" for keyword in STOCK_KEYWORDS["in_stock"]: if keyword in text_lower: return "In Stock ✅" return "Unknown ❓"

def clean_text(text): if text: return text.strip().replace("\n", " ").replace(",", " ") return "Unknown Product"

============================================================

CORE ENGINE

============================================================

async def analyze_product_page(session, url): html = await fetch(session, url) if not html: return {"URL": url, "Name": "Error/Blocked", "Original Price": "N/A", "Discounted Price": "N/A", "Stock": "N/A"}

soup = BeautifulSoup(html, "html.parser")
h1 = soup.find("h1")
product_name = clean_text(h1.text) if h1 else clean_text(soup.title.text)

# البحث عن الأسعار
prices_tags = soup.find_all(class_=re.compile(r"price|amount|offer", re.I))
original_price = discounted_price = "N/A"
if prices_tags:
    if len(prices_tags) >= 2:
        original_price = extract_price(prices_tags[0].get_text())
        discounted_price = extract_price(prices_tags[1].get_text())
    else:
        discounted_price = extract_price(prices_tags[0].get_text())

if discounted_price == "N/A":
    discounted_price = extract_price(soup.get_text()[:2000])

stock_status = check_availability(html[:5000])

return {
    "URL": url,
    "Name": product_name,
    "Original Price": original_price,
    "Discounted Price": discounted_price,
    "Stock": stock_status,
    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
}

============================================================

STREAMLIT UI

============================================================

st.set_page_config(page_title="Market Spy Ω", layout="wide") st.title("📊 Market Spy Ω — Competitive Intelligence Tool") st.markdown(""" Data Engineer: Abdullah Abbas
Service: Rapid Competitor Price Tracking
Output: Business-Ready CSV/Excel """)

urls_input = st.text_area("Paste Product URLs (Line by line or comma separated):", height=150)

if st.button("🚀 Extract Market Data"): urls = [u.strip() for u in urls_input.replace('\n', ',').split(",") if u.strip()] if not urls: st.warning("Please enter at least one URL.") else: st.success(f"Tracking {len(urls)} products... Please wait.") async def run_scan(): async with aiohttp.ClientSession() as session: tasks = [analyze_product_page(session, url) for url in urls] return await asyncio.gather(*tasks)

results = asyncio.run(run_scan())
    st.dataframe(results, use_container_width=True)

    import io
    output = io.StringIO()
    headers = ["Name", "Original Price", "Discounted Price", "Stock", "URL", "Timestamp"]
    writer = csv.DictWriter(output, fieldnames=headers)
    writer.writeheader()
    for row in results:
        writer.writerow(row)
    csv_bytes = output.getvalue().encode('utf-8')

    st.download_button(
        label="📥 Download Competitor Data (CSV)",
        data=csv_bytes,
        file_name=f"Market_Spy_Report_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
    st.balloons()
