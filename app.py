import streamlit as st
import requests
from urllib.parse import urlparse

st.set_page_config(page_title="ULN Inspector", layout="centered")

st.title("🛡️ ULN Inspector (Private)")
st.write("أداة تحليل تقني مبسطة لاكتشاف نقاط ضعف شائعة — بدون تعقيد")

# --- Input ---
target = st.text_input("🔗 أدخل رابط الموقع (مع https://)")

if st.button("🔍 افحص الموقع"):
    if not target.startswith("http"):
        st.error("الرجاء إدخال رابط صحيح يبدأ بـ http أو https")
    else:
        try:
            r = requests.get(target, timeout=10)
            st.success("تم الاتصال بالموقع بنجاح")

            st.subheader("📄 معلومات عامة")
            st.write("Status Code:", r.status_code)
            st.write("Server:", r.headers.get("Server", "غير معروف"))

            st.subheader("🔐 فحص أمني مبسط")

            issues = []

            if "X-Frame-Options" not in r.headers:
                issues.append("غياب X-Frame-Options (خطر Clickjacking)")

            if "Content-Security-Policy" not in r.headers:
                issues.append("غياب Content-Security-Policy")

            if "X-Content-Type-Options" not in r.headers:
                issues.append("غياب X-Content-Type-Options")

            if issues:
                for i in issues:
                    st.warning(i)
            else:
                st.success("لم يتم العثور على مشاكل واضحة في الرؤوس")

            st.subheader("🧠 تقييم منطقي")
            if len(issues) >= 2:
                st.info("الموقع يحتاج مراجعة أمنية أساسية")
            else:
                st.info("الوضع العام جيد، لا مؤشرات خطيرة واضحة")

        except Exception as e:
            st.error(f"خطأ أثناء الفحص: {e}")
