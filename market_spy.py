# ============================================================
# Market Spy Ω — Competitive Intelligence & Price Tracker
# Business-Grade • High-Speed • Excel-Ready
# Author: Abdullah Abbas
# WhatsApp: +96407881296737
# ============================================================

import asyncio
import aiohttp
import re
import csv
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import streamlit as st
import pandas as pd
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9,ar;q=0.8"
}

CONCURRENT_REQUESTS = 5

# الأسعار (دولار، يورو، دينار)
PRICE_PATTERNS = [
    r"\$\s?[\d,]+(\.\d{2})?",
    r"[\d,]+(\.\d{2})?\s?USD",
    r"€\s?[\d,]+(\.\d{2})?",
    r"[\d,]+(\.\d{2})?\s?د\.ع"
]

STOCK_KEYWORDS = {
    "in_stock": ["in stock", "available", " متوفر", "متاح", "add to cart"],
    "out_of_stock": ["out of stock", "sold out", "نفذت الكمية", "غير متوفر", "coming soon"]
}

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

async def fetch(session, url):
    try:
        async with session.get(url, headers=HEADERS, timeout=15) as response:
            if response.status == 200:
                return await response.text(errors='ignore')
            return None
    except Exception:
        return None

def clean_text(text):
    return text.strip().replace("\n", " ").replace(",", " ") if text else "Unknown"

def extract_prices(text):
    prices = []
    for pattern in PRICE_PATTERNS:
        matches = re.findall(pattern, text)
        for match in matches:
            price_clean = re.sub(r"[^\d\.]", "", match)
            if price_clean:
                prices.append(price_clean)
    if len(prices) == 0:
        return ("N/A", "N/A")
    elif len(prices) == 1:
        return (prices[0], prices[0])
    else:
        return (prices[0], prices[1])  # original, discounted

def check_availability(text):
    text_lower = text.lower()
    for keyword in STOCK_KEYWORDS["out_of_stock"]:
        if keyword in text_lower:
            return "Out of Stock ❌"
    for keyword in STOCK_KEYWORDS["in_stock"]:
        if keyword in text_lower:
            return "In Stock ✅"
    return "Unknown ❓"

# ============================================================
# CORE ENGINE
# ============================================================

async def analyze_product_page(session, url):
    html = await fetch(session, url)
    if not html:
        return {"URL": url, "Name": "Error/Blocked", "Original Price": "N/A",
                "Discounted Price": "N/A", "Stock": "N/A", "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")}
    
    soup = BeautifulSoup(html, "lxml")
    
    # الاسم
    h1 = soup.find("h1")
    product_name = clean_text(h1.text) if h1 else clean_text(soup.title.text)
    
    # الأسعار
    price_text = soup.get_text()[:5000]
    original_price, discounted_price = extract_prices(price_text)
    
    # المخزون
    stock_status = check_availability(price_text)
    
    return {
        "URL": url,
        "Name": product_name,
        "Original Price": original_price,
        "Discounted Price": discounted_price,
        "Stock": stock_status,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
    }

# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(page_title="Market Spy Ω", layout="wide")
st.title("📊 Market Spy Ω — Competitive Intelligence Tool")
st.markdown("""
**Data Engineer:** Abdullah Abbas  
**Service:** Rapid Competitor Price Tracking  
**Output:** Business-Ready Excel
""")

urls_input = st.text_area("Paste Product URLs (Line by line or comma separated):", height=150)

if st.button("🚀 Extract Market Data"):
    urls = [u.strip() for u in urls_input.replace('\n', ',').split(",") if u.strip()]
    if not urls:
        st.warning("Please enter at least one URL.")
    else:
        st.success(f"Tracking {len(urls)} products... Please wait.")
        
        async def run_scan():
            sem = asyncio.Semaphore(CONCURRENT_REQUESTS)
            async with aiohttp.ClientSession() as session:
                async def sem_task(url):
                    async with sem:
                        return await analyze_product_page(session, url)
                tasks = [sem_task(url) for url in urls]
                return await asyncio.gather(*tasks)
        
        results = asyncio.run(run_scan())
        
        # عرض النتائج
        st.dataframe(results, use_container_width=True)
        
        # حفظ CSV/Excel
        df = pd.DataFrame(results)
        file_name = f"Market_Spy_Report_{datetime.now().strftime('%Y%m%d')}.xlsx"
        df.to_excel(file_name, index=False, engine='openpyxl')
        
        with open(file_name, "rb") as f:
            st.download_button(
                label="📥 Download Competitor Data (Excel)",
                data=f,
                file_name=file_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        st.balloons()
