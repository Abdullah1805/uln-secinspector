# ============================================================
# F16‑Research Edition — Streamlit Interface
# ============================================================

import streamlit as st
import asyncio
from core.orchestrator import F16Orchestrator


st.set_page_config(
    page_title="F16 Research Edition",
    layout="wide"
)

st.title("🛡️ F16‑Research Edition")
st.caption("Evidence‑based Security Research Scanner")

target = st.text_input(
    "Target URL",
    placeholder="https://example.com"
)

if st.button("🚀 Start Scan") and target:
    with st.spinner("Running deep research scan..."):
        orchestrator = F16Orchestrator(target)
        result = asyncio.run(orchestrator.run())

    st.success("Scan completed")

    st.subheader("Findings")
    if not result["findings"]:
        st.info("No high‑confidence vulnerabilities detected.")
    else:
        for f in result["findings"]:
            with st.expander(f["title"]):
                st.write("**Confidence:**", round(f["confidence"], 2))
                st.write("**STRIDE:**", ", ".join(f["stride"]))
                st.json(f["evidence"])

    st.subheader("Critical Attack Paths")
    st.write(result["critical_paths"])
