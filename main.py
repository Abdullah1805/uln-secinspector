import asyncio
import streamlit as st
from engine import SovereignBBEngine

# هذا السطر ضروري في Streamlit للتعامل مع asyncio
if "loop" not in st.session_state:
    st.session_state.loop = asyncio.new_event_loop()

async def start_scan(target):
    engine = SovereignBBEngine(concurrency=20)
    results_found = await engine.run(target)
    if not results_found:
        st.warning("لم يتم العثور على ثغرات أو الهدف خارج النطاق المسموح.")

def run_async_main(target):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(start_scan(target))

st.set_page_config(page_title="Sovereign Scanner", page_icon="🛡️")

st.title("🛡️ Sovereign BB Engine")
st.write("أداة فحص ثغرات SQL Injection (Time-based)")

target_input = st.text_input("أدخل رابط الهدف:", "https://example.com")

if st.button("بدء الفحص"):
    if target_input:
        with st.spinner("جاري الفحص... يرجى الانتظار"):
            run_async_main(target_input)
    else:
        st.error("يرجى إدخال رابط!")
