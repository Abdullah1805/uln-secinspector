import streamlit as st
import asyncio
from engine import SovereignBBEngine
from bb_report import ProofBuilder

st.set_page_config(page_title="Sovereign BB Engine v2.0", layout="wide")

st.title("🛡️ Sovereign Bug Bounty Engine v2.0")
st.caption("High-Precision Multi-Vulnerability Scanner")

st.sidebar.header("⚙️ Settings")
target = st.sidebar.text_input("Target (In Scope)", "https://example.com")
concurrency = st.sidebar.slider("Concurrency", 5, 40, 15)
start = st.sidebar.button("🚀 Start Scan")

if start and target:
    engine = SovereignBBEngine(target, concurrency)
    with st.spinner("Scanning with precision..."):
        results = asyncio.run(engine.run())

    if not results:
        st.success("No confirmed vulnerabilities found.")
    else:
        st.error(f"{len(results)} confirmed vulnerabilities")

        for i, f in enumerate(results, 1):
            st.markdown(f"""
### 🔴 Finding #{i}
- **Type:** {f['impact']}
- **URL:** `{f['url']}`
- **Parameter:** `{f.get('param', '-')}`
- **Confidence:** {f['confidence']}%
""")

        report = ProofBuilder().build_bulk(target, results)
        st.download_button(
            "📄 Download Report",
            report,
            "bugbounty_report.md",
            "text/markdown"
        )
