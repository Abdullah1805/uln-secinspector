class ImpactEngine:
    def escalate(self, f, evidence):
        if f["type"] == "PARAM_TEST":
            return {
                "impact": "Time-based SQL Injection",
                "confidence": "High"
            }
        return None
