# ============================================================
# ML Noise Reduction Engine
# ============================================================

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class MLNoiseReducer:
    """
    الهدف:
    - تقليل False Positives
    - ليس استبدال المنطق التحليلي
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=9,
            random_state=42
        )
        self._bootstrap_train()

    def _bootstrap_train(self):
        """
        تدريب أولي صناعي (Research‑grade bootstrap).
        في الأنظمة العملاقة يُستبدل ببيانات تاريخية حقيقية.
        """
        X, y = [], []

        for _ in range(500):
            dom = np.random.rand()
            event = np.random.rand()
            timing = np.random.rand()
            status = np.random.rand()

            score = (
                dom * 0.30 +
                event * 0.35 +
                timing * 0.20 +
                status * 0.15
            )

            X.append([dom, event, timing, status])
            y.append(1 if score > 0.55 else 0)

        X = self.scaler.fit_transform(X)
        self.model.fit(X, y)

    def probability(self, features: dict) -> float:
        vec = [
            features.get("dom_change", 0.0),
            features.get("event_trigger", 0.0),
            features.get("timing_shift", 0.0),
            features.get("status_anomaly", 0.0),
        ]
        vec = self.scaler.transform([vec])
        return round(self.model.predict_proba(vec)[0][1], 3)
