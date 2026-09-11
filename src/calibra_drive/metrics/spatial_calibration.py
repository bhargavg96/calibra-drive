import numpy as np
from typing import Dict, List
from .calibration import CalibrationMetrics

class SpatialCalibrationAnalyzer:
    def __init__(self, calibration_metrics: CalibrationMetrics, distance_bins: List[float], categories: List[str]):
        self.metrics = calibration_metrics
        self.distance_bins = distance_bins
        self.categories = categories

    def analyze_by_distance(self, predicted_probs: np.ndarray, ground_truth: np.ndarray, distances: np.ndarray) -> Dict[str, float]:
        results = {}
        for i in range(len(self.distance_bins) - 1):
            low, high = self.distance_bins[i], self.distance_bins[i+1]
            mask = (distances >= low) & (distances < high)
            if np.any(mask):
                results[f"{low}-{high}m"] = self.metrics.ece(predicted_probs[mask], ground_truth[mask])
        return results

    def analyze_by_category(self, predicted_probs: np.ndarray, ground_truth: np.ndarray, category_mask: Dict[str, np.ndarray]) -> Dict[str, float]:
        results = {}
        for cat in self.categories:
            if cat in category_mask and np.any(category_mask[cat]):
                results[cat] = self.metrics.ece(predicted_probs[category_mask[cat]], ground_truth[category_mask[cat]])
        return results

    def analyze_by_horizon(self, preds_per_step: np.ndarray, gts_per_step: np.ndarray, horizons: List[float]) -> Dict[str, float]:
        results = {}
        T = min(preds_per_step.shape[0], len(horizons))
        for t in range(T):
            results[f"{horizons[t]}s"] = self.metrics.ece(preds_per_step[t], gts_per_step[t])
        return results

    def full_analysis(self, preds, gts, distances, category_mask, horizons) -> Dict[str, Dict]:
        return {
            'distance': self.analyze_by_distance(preds, gts, distances),
            'category': self.analyze_by_category(preds, gts, category_mask),
            'horizon': self.analyze_by_horizon(preds, gts, horizons)
        }
