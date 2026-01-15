import streamlit as st
from engine.scanner import scan_terraform
from engine.reporter import generate_bilingual_report

st.set_page_config(page_title="ULN SecInspector", layout="centered")
st.title("🛡️ ULN SecInspector")
st.write("أداة فحص أمني مبسطة – لا تحتاج خبرة أمن سيبراني")

uploaded = st.file_uploader("📂 ارفع ملف Terraform (.tf)", type=["tf"])

if uploaded:
    content = uploaded.read().decode("utf-8")

    if st.button("🔍 افحص الآن"):
        findings = scan_terraform(content)

        if not findings:
            st.success("✅ لم يتم العثور على ثغرات خطيرة")
        else:
            for f in findings:
                ar, en = generate_bilingual_report(f)

                st.subheader("📌 الشرح (عربي)")
                st.info(ar)

                st.subheader("📄 Report (English)")
                st.code(en)
