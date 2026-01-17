import streamlit as st
import asyncio
from engine import SovereignBBEngine
from report import ProofBuilder

st.set_page_config(page_title="Sovereign BB Engine", layout="wide")
st.title("🛡️ Sovereign Bug Bounty Engine")

st.sidebar.header("⚙️ Scan Settings")
target_url = st.sidebar.text_input("Target URL", placeholder="https://example.com")
concurrency = st.sidebar.slider("Concurrency Limit", 5, 100, 20)
start_scan = st.sidebar.button("🚀 Start Scan")

if start_scan and target_url:
    st.info(f"Starting scan on {target_url} with concurrency {concurrency}...")

    engine = SovereignBBEngine(concurrency=concurrency)
    findings = asyncio.run(engine.run(target_url))

    if findings:
        st.success(f"Scan completed: {len(findings)} potential issues found!")
        for f in findings:
            st.markdown(f"**{f['impact']}** at `{f['url']}` param `{f['param']}`")
        
        if st.button("📄 Download Markdown Report"):
            proof_gen = ProofBuilder()
            report_md = proof_gen.build_bulk(target_url, findings)
            st.download_button(
                label="Download Report",
                data=report_md,
                file_name="bugbounty_report.md",
                mime="text/markdown"
            )
    else:
        st.info("No issues found or all findings discarded as false positives.")
