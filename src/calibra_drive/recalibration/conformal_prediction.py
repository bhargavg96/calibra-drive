import numpy as np

class ConformalPredictor:
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self._threshold = 0.5

    def fit(self, predictions: np.ndarray, ground_truth: np.ndarray):
        preds = predictions.flatten()
        gts = ground_truth.flatten()
        
        scores = np.abs(preds - gts)
        n = len(scores)
        q = np.ceil((n + 1) * (1 - self.alpha)) / n
        q = min(max(q, 0.0), 1.0)
        self._threshold = float(np.quantile(scores, q))
        return self

    def predict_set(self, predictions: np.ndarray) -> np.ndarray:
        lower = np.clip(predictions - self._threshold, 0, 1)
        upper = np.clip(predictions + self._threshold, 0, 1)
        return np.stack([lower, upper], axis=-1)

    @property
    def coverage_target(self) -> float:
        return 1 - self.alpha

    @property
    def threshold(self) -> float:
        return self._threshold

    def evaluate_coverage(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        pred_sets = self.predict_set(predictions)
        lower, upper = pred_sets[..., 0], pred_sets[..., 1]
        covered = (ground_truth >= lower) & (ground_truth <= upper)
        return float(np.mean(covered))
