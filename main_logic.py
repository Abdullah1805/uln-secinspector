def generate_bilingual_report(f):
    ar = f"""
العنوان: {f['title']}
الخطورة: {f['severity']}
الوصف: {f['impact']}
الدليل: {f['evidence']}
التوصية: {f['recommendation']}
"""

    en = f"""
Title: {translate(f['title'])}
Severity: {f['severity']}
Description: A security misconfiguration was identified.
Impact: {f['impact']}
Evidence: {f['evidence']}
Recommendation: {translate(f['recommendation'])}
"""

    return ar, en

def translate(text):
    mapping = {
        "Security Group مفتوح للعالم": "Security Group Open to the Internet",
        "S3 Bucket عام بدون حماية": "Public S3 Bucket",
        "IAM Policy بصلاحيات كاملة": "Overly Permissive IAM Policy",
        "تقييد الصلاحيات حسب الحاجة.": "Apply the principle of least privilege."
    }
    return mapping.get(text, text)
