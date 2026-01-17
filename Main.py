import asyncio
import streamlit as st
from engine import SovereignBBEngine

async def main():
    st.title("Sovereign BB Scanner")
    target = st.text_input("Enter Target URL", "https://example.com")
    
    if st.button("Start Scan"):
        engine = SovereignBBEngine(concurrency=40)
        with st.spinner("Scanning..."):
            await engine.run(target)

if __name__ == "__main__":
    asyncio.run(main())
