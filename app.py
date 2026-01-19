import streamlit as st
import asyncio
import nest_asyncio
from core.orchestrator import F16Orchestrator

nest_asyncio.apply()

st.set_page_config(
    page_title="F16 Security Inspector",
    layout="wide"
)

st.title("🛡️ F16 – Web Security Inspector")
st.caption("Authorized Security Testing Only")

target = st.text_input(
    "Target URL",
    placeholder="https://example.com"
)

run = st.button("Start Scan")

if run and target:
    st.info("Scan started… this may take a moment")

    async def runner():
        engine = F16Orchestrator(target)
        result = await engine.run()
        return result

    findings = asyncio.run(runner())

    if not findings:
        st.success("No confirmed vulnerabilities found")
    else:
        st.error(f"Found {len(findings)} issue(s)")

        for f in findings:
            with st.expander(f["title"]):
                st.write("**Parameter:**", f["parameter"])
                st.write("**Type:**", f["type"])
                st.write("**Confidence:**", f["confidence"])
                st.code(f["evidence"])
