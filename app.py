# ============================================================
# Market Spy Ω Web — Competitive Intelligence Dashboard
# Author: Abdullah Abbas
# WhatsApp: +96407881296737
# ============================================================

import asyncio
import aiohttp
import re
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from datetime import datetime
import matplotlib.pyplot as plt

# ================= CONFIG =================

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept-Language": "en-US,en;q=0.9,ar;q=0.8"
}

PRICE_REGEX = {
    "discounted": r"(sale|now|price|offer)?\s*[$€]\s?[\d,]+(\.\d{2})?",
    "original": r"(was|original|regular)\s*[$€]\s?[\d,]+(\.\d{2})?"
}

STOCK_WORDS = {
    "in": ["in stock", "available", "add to cart", "متوفر"],
    "out": ["out of stock", "sold out", "غير متوفر"]
}

# ================= CORE =================

async def fetch(session, url):
    try:
        async with session.get(url, headers=HEADERS, timeout=15) as r:
            if r.status == 200:
                return await r.text()
    except:
        pass
    return None

def extract_price(text, kind):
    m = re.search(PRICE_REGEX[kind], text, re.I)
    return m.group(0) if m else "N/A"

def stock_status(text):
    t = text.lower()
    if any(w in t for w in STOCK_WORDS["out"]):
        return "Out of Stock ❌"
    if any(w in t for w in STOCK_WORDS["in"]):
        return "In Stock ✅"
    return "Unknown ❓"

async def analyze(url, session):
    html = await fetch(session, url)
    if not html:
        return {
            "URL": url,
            "Product": "Blocked / Error",
            "Original Price": "N/A",
            "Discount Price": "N/A",
            "Stock": "N/A",
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
        }

    soup = BeautifulSoup(html, "html.parser")
    name = soup.find("h1")
    product = name.text.strip() if name else soup.title.text.strip()

    text = soup.get_text(" ", strip=True)[:5000]

    return {
        "URL": url,
        "Product": product,
        "Original Price": extract_price(text, "original"),
        "Discount Price": extract_price(text, "discounted"),
        "Stock": stock_status(text),
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
    }

async def run(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [analyze(u, session) for u in urls]
        return await asyncio.gather(*tasks)

# ================= STREAMLIT UI =================

st.set_page_config("Market Spy Ω", layout="wide")

st.title("📊 Market Spy Ω — Competitive Price Intelligence")
st.markdown("""
**Developer:** Abdullah Abbas  
**WhatsApp:** +96407881296737  
**Use only on authorized targets**
""")

urls_input = st.text_area(
    "Paste product URLs (one per line):",
    height=200
)

if st.button("🚀 Analyze Market"):
    urls = [u.strip() for u in urls_input.splitlines() if u.strip()]

    if not urls:
        st.warning("Please enter at least one URL.")
    else:
        with st.spinner("Collecting competitor data..."):
            data = asyncio.run(run(urls))

        df = pd.DataFrame(data)
        st.success("Analysis completed ✔")

        st.dataframe(df, use_container_width=True)

        # CSV Download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download CSV",
            csv,
            file_name="market_spy_report.csv",
            mime="text/csv"
        )

        # Chart
        def clean_price(x):
            try:
                return float(re.sub(r"[^\d.]", "", x))
            except:
                return None

        df["Original_Num"] = df["Original Price"].apply(clean_price)
        df["Discount_Num"] = df["Discount Price"].apply(clean_price)

        st.subheader("📈 Price Comparison Chart")
        fig, ax = plt.subplots()
        df.plot(
            x="Product",
            y=["Original_Num", "Discount_Num"],
            kind="bar",
            ax=ax
        )
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
