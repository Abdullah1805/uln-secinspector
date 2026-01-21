# ============================================================
# Market Spy Ω — Client-Ready Excel + PDF Report
# Author: Abdullah Abbas
# WhatsApp: +96407881296737
# ============================================================

import asyncio
import aiohttp
import re
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
import streamlit as st
import io
import pdfkit

# ============================================================
# CONFIG
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
    r"[\d,]+(\.\d{2})?\s?IQD",
    r"€\s?[\d,]+(\.\d{2})?",
    r"[\d,]+(\.\d{2})?\s?د\.ع"
]

STOCK_KEYWORDS = {
    "in_stock": ["in stock", "available", " متوفر", "متاح", "add to cart"],
    "out_of_stock": ["out of stock", "sold out", "نفذت الكمية", "غير متوفر", "coming soon"]
}

# ============================================================
# UTILS
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
            price = re.sub(r"[^\d.]", "", match.group(0))
            try:
                return float(price.replace(",", ""))
            except:
                return None
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
        return {"URL": url, "Name": "Error/Blocked", "Price": None, "Stock": "N/A"}
    
    soup = BeautifulSoup(html, "html.parser")
    
    # Extract Name
    h1 = soup.find("h1")
    product_name = clean_text(h1.text) if h1 else clean_text(soup.title.text)
    
    # Extract Price
    price_tag = soup.find(class_=re.compile(r"price|amount|offer", re.I))
    price = extract_price(price_tag.get_text()) if price_tag else extract_price(soup.get_text()[:2000])

    # Stock Status
    stock_status = check_availability(html[:5000])

    return {
        "URL": url,
        "Name": product_name,
        "Price": price,
        "Stock": stock_status,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
    }

# ============================================================
# REPORT GENERATOR (Excel + PDF)
# ============================================================
def generate_client_report(results):
    df = pd.DataFrame(results)
    df["Price_Num"] = df["Price"].fillna(0)

    # Summary
    numeric_df = df.dropna(subset=["Price_Num"])
    lowest = numeric_df.loc[numeric_df["Price_Num"].idxmin()]
    highest = numeric_df.loc[numeric_df["Price_Num"].idxmax()]
    gap = highest["Price_Num"] - lowest["Price_Num"]

    summary_df = pd.DataFrame({
        "Metric": [
            "Cheapest Product",
            "Cheapest Price",
            "Most Expensive Product",
            "Highest Price",
            "Price Gap",
            "Recommendation"
        ],
        "Value": [
            lowest["Name"], lowest["Price"],
            highest["Name"], highest["Price"],
            f"{gap:.2f}",
            "Adjust pricing to stay competitive"
        ]
    })

    # Excel file
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="Executive Summary", index=False)
        df.to_excel(writer, sheet_name="Market Data", index=False)
        workbook = writer.book
        worksheet = writer.sheets["Market Data"]

        chart = workbook.add_chart({"type": "column"})
        chart.add_series({
            "name": "Prices",
            "categories": f"=Market Data!$B$2:$B${len(df)+1}",
            "values": f"=Market Data!$C$2:$C${len(df)+1}",
            "data_labels": {"value": True},
        })
        chart.set_title({"name": "Competitor Price Comparison"})
        chart.set_x_axis({"name": "Product"})
        chart.set_y_axis({"name": "Price"})
        worksheet.insert_chart("H2", chart)
    output.seek(0)

    # Generate PDF
    html_table = df.to_html(index=False)
    pdf_output = io.BytesIO()
    pdfkit.from_string(html_table, pdf_output, options={"enable-local-file-access": ""})
    pdf_output.seek(0)

    return output, pdf_output

# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="Market Spy Ω", layout="wide")
st.title("📊 Market Spy Ω — Client-Ready Price Tracker")
st.markdown("""
**Data Engineer:** Abdullah Abbas  
**WhatsApp:** +96407881296737  
**Output:** Excel + PDF ready for client
""")

urls_input = st.text_area("Paste Product URLs (one per line or comma separated):", height=150)

if st.button("🚀 Run Scan & Generate Report"):
    urls = [u.strip() for u in urls_input.replace("\n", ",").split(",") if u.strip()]
    
    if not urls:
        st.warning("Please enter at least one URL.")
    else:
        st.success(f"Tracking {len(urls)} products... Please wait.")
        async def run_scan():
            async with aiohttp.ClientSession() as session:
                tasks = [analyze_product_page(session, url) for url in urls]
                return await asyncio.gather(*tasks)
        results = asyncio.run(run_scan())

        excel_file, pdf_file = generate_client_report(results)

        st.download_button(
            label="📥 Download Client-Ready Excel Report",
            data=excel_file,
            file_name=f"Market_Report_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.download_button(
            label="📥 Download Client-Ready PDF Report",
            data=pdf_file,
            file_name=f"Market_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf"
        )

        st.dataframe(results, use_container_width=True)
        st.balloons()
