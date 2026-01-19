# ============================================================
# F16 – Research Edition
# Streamlit Control Panel (Production Entry Point)
# ============================================================
# AUTHORIZED SECURITY TESTING ONLY
# ============================================================

import streamlit as st
import asyncio
import time
from typing import List

# ----------------------------
# Core Imports
# ----------------------------
from browser.stealth import StealthBrowser
from browser.sensors import SensorSuite
from attack.vectors import VectorExtractor, InjectionPoint
from attack.engine import AttackEngine
from attack.graph import AttackGraph
from attack.stride import STRIDEAnalyzer
from attack.cvss import CVSSCalculator
from core.decision import DecisionEngine
from core.ml import MLNoiseReducer
from reporting.pdf import PDFReporter

from playwright.async_api import async_playwright


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(
    page_title="F16 – Research Edition",
    page_icon="🦅",
    layout="wide"
)

st.title("🦅 F16 – Research Edition")
st.caption("Scientific‑Grade Web Security Analysis Framework")

st.warning(
    "⚠️ استخدم الأداة فقط على أهداف لديك إذن قانوني لاختبارها",
    icon="⚠️"
)

target_url = st.text_input(
    "🔗 Target URL",
    placeholder="https://example.com"
)

run_scan = st.button("🚀 Run Security Analysis", type="primary")

status_box = st.empty()
result_box = st.container()


# ============================================================
# Async Scan Logic
# ============================================================

async def run_f16_scan(target: str):
    results = {
        "target": target,
        "vectors": [],
        "vulnerabilities": [],
        "attack_graph": AttackGraph()
    }

    ml_reducer = MLNoiseReducer()

    async with async_playwright() as pw:
        browser, context = await StealthBrowser.launch(pw)
        page = await context.new_page()

        try:
            status_box.info("🔍 Navigating to target …")
            await page.goto(target, wait_until="networkidle", timeout=30000)

            # =========================
            # 1. Vector Extraction
            # =========================
            status_box.info("🧠 Extracting injection vectors …")
            vectors: List[InjectionPoint] = await VectorExtractor.extract(page, target)
            results["vectors"] = vectors

            for vec in vectors:
                results["attack_graph"].add_vector(vec)

            # =========================
            # 2. Active Testing
            # =========================
            status_box.info("⚔️ Executing context‑aware attacks …")

            engine = AttackEngine(page)

            for vec in vectors:
                sensors = SensorSuite(page)
                await sensors.attach()

                vuln = await engine.test_vector(vec)

                if vuln:
                    # Feature fusion
                    features = sensors.features()
                    bayes_score = DecisionEngine.score(features)
                    ml_score = ml_reducer.probability(features)

                    confidence = DecisionEngine.confidence(
                        bayes=bayes_score,
                        ml=ml_score
                    )

                    vuln.confidence = confidence
                    vuln.cvss = CVSSCalculator.score(vuln.title, confidence)
                    vuln.stride = STRIDEAnalyzer.classify(vuln.title)

                    results["vulnerabilities"].append(vuln)
                    results["attack_graph"].add_vulnerability(vuln)

                await asyncio.sleep(0.4)  # conservative pacing

        finally:
            await page.close()
            await browser.close()

    return results


# ============================================================
# Streamlit Execution
# ============================================================

if run_scan and target_url:
    with st.spinner("Running scientific security analysis …"):
        try:
            scan_results = asyncio.run(run_f16_scan(target_url))
        except RuntimeError:
            # Streamlit event loop workaround
            scan_results = asyncio.get_event_loop().run_until_complete(
                run_f16_scan(target_url)
            )

    status_box.success("✅ Scan completed successfully")

    # ========================================================
    # Results Rendering
    # ========================================================

    with result_box:
        st.subheader("📊 Scan Summary")

        col1, col2, col3 = st.columns(3)
        col1.metric("Injection Vectors", len(scan_results["vectors"]))
        col2.metric("Confirmed Vulnerabilities", len(scan_results["vulnerabilities"]))
        col3.metric(
            "Critical Issues",
            len([v for v in scan_results["vulnerabilities"] if v.cvss >= 8.0])
        )

        st.divider()

        if not scan_results["vulnerabilities"]:
            st.success("🎉 No exploitable vulnerabilities detected with current payload set.")
        else:
            for v in scan_results["vulnerabilities"]:
                with st.expander(f"🚨 {v.title} | CVSS {v.cvss}"):
                    st.write("**Parameter:**", v.vector.parameter)
                    st.write("**Location:**", v.vector.location)
                    st.write("**Payload:**", f"`{v.payload}`")
                    st.write("**Evidence:**", v.evidence)
                    st.write("**Confidence:**", v.confidence)
                    st.write("**STRIDE:**", ", ".join(v.stride))

        st.divider()

        if st.button("📄 Generate Professional PDF Report"):
            pdf_path = PDFReporter.generate(scan_results)
            st.success("📄 Report generated")
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Report",
                    f,
                    file_name=pdf_path,
                    mime="application/pdf"
                  )
