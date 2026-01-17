import streamlit as st
import asyncio
from engine import SovereignBBEngine
from bb_report import ProofBuilder

st.set_page_config(
    page_title="Sovereign Bug Bounty Engine",
    layout="wide"
)

st.title("🛡️ Sovereign Bug Bounty Engine")
st.caption("Scope-Aware Autonomous Bug Bounty Scanner")

st.sidebar.header("⚙️ Scan Settings")
target_url = st.sidebar.text_input(
    "Target URL (In-Scope)",
    placeholder="https://example.com"
)
concurrency = st.sidebar.slider("Concurrency", 5, 100, 20)
start_scan = st.sidebar.button("🚀 Start Scan")

if start_scan and target_url:
    st.info("Scope locked. Starting scan…")

    engine = SovereignBBEngine(
        base_scope=target_url,
        concurrency=concurrency
    )

    with st.spinner("Scanning target within scope..."):
        findings = asyncio.run(engine.run())

    if not findings:
        st.success("Scan completed. No valid in-scope vulnerabilities found.")
    else:
        st.error(f"{len(findings)} in-scope vulnerability found")

        for i, f in enumerate(findings, 1):
            st.markdown(f"""
### 🔴 Finding #{i}
- **Type:** {f['impact']}
- **URL:** `{f['url']}`
- **Parameter:** `{f['param']}`
- **Evidence:** {f['evidence']['delay']}s delay
""")

        if st.button("📄 Download Bug Bounty Report"):
            builder = ProofBuilder()
            report = builder.build_bulk(target_url, findings)
            st.download_button(
                "⬇️ Download report.md",
                report,
                "bugbounty_report.md",
                "text/markdown"
            )
else:
    st.info("Enter a valid in-scope URL to begin.")
