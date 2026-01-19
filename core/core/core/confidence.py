# ============================================================
# Confidence Fusion Utilities
# ============================================================

class ConfidenceFusion:
    """
    Harmonic Mean لتقليل الانحياز
    """

    @staticmethod
    def harmonic_mean(values, weights):
        numerator = sum(weights)
        denominator = sum(
            (w / (v + 1e-6)) for v, w in zip(values, weights)
        )
        return round(numerator / denominator, 3)
