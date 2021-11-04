import numpy as np


class ThresholdCalculator():
    """
    A class to calculate threshold for anomaly detection.
    Currently only one method supported.
    """

    def __init__(self, quantile=0.99, scale=1.05):
        self._quantile = quantile
        self._scale = scale

    def compute_threshold(self, anomaly_scores):
        return np.quantile(anomaly_scores, self._quantile) * self._scale

    def get_quantile(self):
        return self._quantile
