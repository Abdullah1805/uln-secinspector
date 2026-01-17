import streamlit as st
import asyncio
from engine import SovereignBBEngine
from bb_report import ProofBuilder

st.set_page_config(
    page_title="Sovereign Bug Bounty Engine",
    layout="wide"
)

st.title("🛡️ Sovereign Bug Bounty Engine v1.0")
st.caption("Precision In-Scope Bug Bounty Scanner")

st.sidebar.header("⚙️ Scan Settings")
target_url = st.sidebar.text_input(
    "Target URL (In Scope)",
    placeholder="https://example.com"
)
concurrency = st.sidebar.slider("Concurrency", 5, 50, 15)
start_scan = st.sidebar.button("🚀 Start Scan")

if start_scan and target_url:
    st.info("🔒 Scope locked. Scan started.")

    engine = SovereignBBEngine(
        base_scope=target_url,
        concurrency=concurrency
    )

    with st.spinner("Scanning target (strict scope)…"):
        findings = asyncio.run(engine.run())

    if not findings:
        st.success("Scan completed. No valid vulnerabilities confirmed.")
    else:
        st.error(f"{len(findings)} confirmed in-scope vulnerabilities found")

        for i, f in enumerate(findings, 1):
            st.markdown(f"""
### 🔴 Finding #{i}
- **Type:** {f['impact']}
- **URL:** `{f['url']}`
- **Parameter:** `{f['param']}`
- **Baseline:** {f['evidence']['baseline']}s  
- **Injected:** {f['evidence']['injected']}s
""")

        builder = ProofBuilder()
        report = builder.build_bulk(target_url, findings)

        st.download_button(
            "📄 Download Bug Bounty Report",
            report,
            "bugbounty_report.md",
            "text/markdown"
        )
else:
    st.info("Enter a valid in-scope URL to begin.")
