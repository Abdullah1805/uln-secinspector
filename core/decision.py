# ============================================================
# Decision Engine — Scientific Confidence Scoring
# ============================================================

class DecisionEngine:
    """
    Conservative decision logic.
    لا نثق بإشارة واحدة، بل ندمج عدة إشارات مستقلة.
    """

    WEIGHTS = {
        "dom_change": 0.30,
        "event_trigger": 0.35,
        "timing_shift": 0.20,
        "status_anomaly": 0.15,
    }

    @staticmethod
    def score(features: dict) -> float:
        """
        Bayesian‑inspired weighted score.
        """
        score = 0.0
        for key, weight in DecisionEngine.WEIGHTS.items():
            score += features.get(key, 0.0) * weight
        return round(min(score, 1.0), 3)

    @staticmethod
    def confidence(bayes: float, ml: float) -> float:
        """
        Fusion: Weighted mean (NOT optimistic max).
        """
        return round((0.6 * bayes + 0.4 * ml), 3)
