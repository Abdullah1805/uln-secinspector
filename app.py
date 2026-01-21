# ============================================================
# Market Spy Ω — Competitive Intelligence & Price Tracker
# Business-Grade • High-Speed • Excel/CSV + Graphs
# Author: Abdullah Abbas
# WhatsApp: +96407881296737
# ============================================================

import asyncio
import aiohttp
import re
import csv
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import streamlit as st
from datetime import datetime
import io
import matplotlib.pyplot as plt
import pandas as pd

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

PRICE_PATTERNS = [
    r"\$\s?[\d,]+(\.\d{2})?",       
    r"[\d,]+(\.\d{2})?\s?USD",      
    r"[\d,]+\s?IQD",                
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

def extract_price(text):
    for pattern in PRICE_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return "N/A"

def price_to_number(price_text):
    if not price_text or price_text == "N/A":
        return None
    cleaned = re.sub(r"[^\d.]", "", price_text.replace(",", ""))
    try:
        return float(cleaned)
    except:
        return None

def check_availability(text):
    text_lower = text.lower()
    for keyword in STOCK_KEYWORDS["out_of_stock"]:
        if keyword in text_lower:
            return "Out of Stock ❌"
    for keyword in STOCK_KEYWORDS["in_stock"]:
        if keyword in text_lower:
            return "In Stock ✅"
    return "Unknown ❓"

def clean_text(text):
    if text:
        return text.strip().replace("\n", " ").replace(",", " ")
    return "Unknown Product"

# ============================================================
# CORE ENGINE
# ============================================================

async def analyze_product_page(session, url):
    html = await fetch(session, url)
    if not html:
        return {"URL": url, "Name": "Error/Blocked", "Price": "N/A", "Price_Num": None, "Stock": "N/A", "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")}
    
    soup = BeautifulSoup(html, "html.parser")
    h1 = soup.find("h1")
    product_name = clean_text(h1.text) if h1 else clean_text(soup.title.text)
    
    # السعر
    price = "N/A"
    price_tag = soup.find(class_=re.compile(r"price|amount|offer", re.I))
    if price_tag:
        price = extract_price(price_tag.get_text())
    if price == "N/A":
        price = extract_price(soup.get_text()[:2000])
    price_num = price_to_number(price)
    
    stock_status = check_availability(html[:5000])
    
    return {
        "URL": url,
        "Name": product_name,
        "Price": price,
        "Price_Num": price_num,
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
**Output:** Business-Ready CSV/Excel + Graphs
""")

urls_input = st.text_area("Paste Product URLs (Line by line or comma separated):", height=150)

if st.button("🚀 Extract Market Data"):
    urls = [u.strip() for u in urls_input.replace('\n', ',').split(",") if u.strip()]
    
    if not urls:
        st.warning("Please enter at least one URL.")
    else:
        st.success(f"Tracking {len(urls)} products... Please wait.")
        
        async def run_scan():
            async with aiohttp.ClientSession() as session:
                tasks = [analyze_product_page(session, url) for url in urls]
                return await asyncio.gather(*tasks)

        results = asyncio.run(run_scan())
        
        # عرض النتائج
        st.dataframe(results, use_container_width=True)
        
        # CSV/Excel للتحميل
        headers = ["Name", "Price", "Stock", "URL", "Timestamp"]
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=headers)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in headers})
        csv_bytes = output.getvalue().encode('utf-8')

        st.download_button(
            label="📥 Download Competitor Data (CSV)",
            data=csv_bytes,
            file_name=f"Market_Spy_Report_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

        # رسم الأسعار
        numeric_prices = [r["Price_Num"] for r in results if r["Price_Num"] is not None]
        names = [r["Name"] for r in results if r["Price_Num"] is not None]
        if numeric_prices:
            fig, ax = plt.subplots(figsize=(10,5))
            ax.barh(names, numeric_prices, color='skyblue')
            ax.set_xlabel("Price")
            ax.set_title("Product Prices Comparison")
            st.pyplot(fig)
        else:
            st.info("No numeric prices found to plot chart.")

        st.balloons()
