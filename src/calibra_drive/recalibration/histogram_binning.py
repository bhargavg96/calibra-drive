import numpy as np

class HistogramBinning:
    def __init__(self, num_bins=15):
        self.num_bins = num_bins
        self.bins = np.linspace(0, 1, num_bins + 1)
        self.corrections = np.zeros(num_bins)

    def fit(self, predicted_probs: np.ndarray, ground_truth: np.ndarray):
        preds = predicted_probs.flatten()
        gts = ground_truth.flatten()
        
        indices = np.digitize(preds, self.bins[:-1]) - 1
        for i in range(self.num_bins):
            mask = indices == i
            if np.any(mask):
                self.corrections[i] = np.mean(gts[mask])
            else:
                self.corrections[i] = (self.bins[i] + self.bins[i+1]) / 2
        return self

    def transform(self, predicted_probs: np.ndarray) -> np.ndarray:
        shape = predicted_probs.shape
        preds = predicted_probs.flatten()
        indices = np.digitize(preds, self.bins[:-1]) - 1
        indices = np.clip(indices, 0, self.num_bins - 1)
        return self.corrections[indices].reshape(shape)
