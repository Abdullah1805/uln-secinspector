# ============================================================
# Market Spy Ω — Professional Price Tracker
# Handles Original & Discounted Price • CSV Ready
# Author: Abdullah Abbas
# WhatsApp: +96407881296737
# ============================================================

import requests
import re
import csv
from bs4 import BeautifulSoup
import streamlit as st
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

def fetch(url):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            return resp.text
        return None
    except Exception:
        return None

def extract_prices(text):
    # البحث عن كل الأسعار في النص (Original و Discounted)
    prices = []
    for pattern in PRICE_PATTERNS:
        matches = re.findall(pattern, text)
        for m in matches:
            clean = re.sub(r"[^\d.,]", "", m)
            if clean not in prices:
                prices.append(clean)
    # إعادة ترتيب: السعر الأول عادة Original، الثاني Discounted
    original = prices[0] if len(prices) >= 1 else "N/A"
    discounted = prices[1] if len(prices) >= 2 else original
    return original, discounted

def check_availability(text):
    text_lower = text.lower()
    for kw in STOCK_KEYWORDS["out_of_stock"]:
        if kw in text_lower:
            return "Out of Stock ❌"
    for kw in STOCK_KEYWORDS["in_stock"]:
        if kw in text_lower:
            return "In Stock ✅"
    return "Unknown ❓"

def clean_text(text):
    if text:
        return text.strip().replace("\n", " ").replace(",", " ")
    return "Unknown Product"

# ============================================================
# CORE ENGINE
# ============================================================

def analyze_product_page(url):
    html = fetch(url)
    if not html:
        return {"URL": url, "Name": "Error/Blocked", "Original Price": "N/A", 
                "Discounted Price": "N/A", "Stock": "N/A"}

    soup = BeautifulSoup(html, "html.parser")

    # 1. استخراج الاسم
    h1 = soup.find("h1")
    product_name = clean_text(h1.text) if h1 else clean_text(soup.title.text)

    # 2. استخراج الأسعار
    price_tag = soup.find(class_=re.compile(r"price|amount|offer", re.I))
    text_for_prices = price_tag.get_text() if price_tag else soup.get_text()[:2000]
    original, discounted = extract_prices(text_for_prices)

    # 3. التوفر
    stock_status = check_availability(html[:5000])

    return {
        "URL": url,
        "Name": product_name,
        "Original Price": original,
        "Discounted Price": discounted,
        "Stock": stock_status,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
    }

# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(page_title="Market Spy Ω Pro", layout="wide")
st.title("📊 Market Spy Ω Pro — Competitor Price Tracker")
st.markdown("""
**Data Engineer:** Abdullah Abbas  
**Service:** Rapid Competitor Price Tracking  
**Output:** Business-Ready CSV/Excel
""")

urls_input = st.text_area("Paste Product URLs (Line by line or comma separated):", height=150)

if st.button("🚀 Extract Market Data"):
    urls = [u.strip() for u in urls_input.replace('\n', ',').split(",") if u.strip()]
    
    if not urls:
        st.warning("Please enter at least one URL.")
    else:
        st.success(f"Tracking {len(urls)} products... Please wait.")
        
        results = [analyze_product_page(url) for url in urls]
        
        st.dataframe(results, use_container_width=True)
        
        # تصدير CSV جاهز للعميل
        headers = ["Name", "Original Price", "Discounted Price", "Stock", "URL", "Timestamp"]
        import io
        output = io.StringIO()
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
