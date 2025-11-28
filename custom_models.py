# custom_models.py
import numpy as np

class CalibratedPipeline:
    """
    Wraps a fitted sklearn Pipeline (with predict_proba)
    and an IsotonicRegression calibrator.
    """
    def __init__(self, base, calibrator):
        self.base = base
        self.calibrator = calibrator

    def predict_proba(self, X):
        p = self.base.predict_proba(X)[:, 1]
        p_cal = self.calibrator.predict(p)
        p_cal = np.clip(p_cal, 0.0, 1.0)
        return np.vstack([1 - p_cal, p_cal]).T

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] > threshold).astype(int)
