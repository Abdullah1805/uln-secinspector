# ============================================================
# STRIDE Threat Modeling
# ============================================================

class STRIDEAnalyzer:

    MAP = {
        "XSS": ["Spoofing", "Information Disclosure"],
        "SQLi": ["Tampering", "Information Disclosure"],
        "SSTI": ["Elevation of Privilege"],
        "IDOR": ["Information Disclosure"],
        "RCE": ["Elevation of Privilege"]
    }

    @staticmethod
    def classify(title: str):
        for key, categories in STRIDEAnalyzer.MAP.items():
            if key.lower() in title.lower():
                return categories
        return ["Tampering"]
