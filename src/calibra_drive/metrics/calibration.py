import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Dict, Tuple

class CalibrationMetrics:
    def __init__(self, num_bins: int = 15, bin_strategy: str = 'equal_width'):
        self.num_bins = num_bins
        self.bin_strategy = bin_strategy

    def compute_all(self, predicted_probs: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
        preds = predicted_probs.flatten()
        gts = ground_truth.flatten()
        return {
            'ece': self.ece(preds, gts),
            'mce': self.mce(preds, gts),
            'brier': self.brier_score(preds, gts),
            'auroc': self.auroc(preds, gts)
        }

    def _get_bins(self, preds: np.ndarray, gts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.bin_strategy == 'equal_width':
            bins = np.linspace(0, 1, self.num_bins + 1)
            indices = np.digitize(preds, bins[:-1]) - 1
        else:
            bins = np.quantile(preds, np.linspace(0, 1, self.num_bins + 1))
            bins[-1] = 1.0 # Ensure max is covered
            indices = np.digitize(preds, bins[:-1]) - 1
            
        bin_accs = np.zeros(self.num_bins)
        bin_confs = np.zeros(self.num_bins)
        bin_counts = np.zeros(self.num_bins)
        
        for i in range(self.num_bins):
            mask = indices == i
            count = np.sum(mask)
            if count > 0:
                bin_accs[i] = np.mean(gts[mask])
                bin_confs[i] = np.mean(preds[mask])
            bin_counts[i] = count
            
        return bin_accs, bin_confs, bin_counts

    def ece(self, predicted_probs: np.ndarray, ground_truth: np.ndarray) -> float:
        preds = predicted_probs.flatten()
        gts = ground_truth.flatten()
        bin_accs, bin_confs, bin_counts = self._get_bins(preds, gts)
        n = len(preds)
        if n == 0:
            return 0.0
        return float(np.sum((bin_counts / n) * np.abs(bin_accs - bin_confs)))

    def mce(self, predicted_probs: np.ndarray, ground_truth: np.ndarray) -> float:
        preds = predicted_probs.flatten()
        gts = ground_truth.flatten()
        bin_accs, bin_confs, bin_counts = self._get_bins(preds, gts)
        valid = bin_counts > 0
        if not np.any(valid):
            return 0.0
        return float(np.max(np.abs(bin_accs[valid] - bin_confs[valid])))

    def brier_score(self, predicted_probs: np.ndarray, ground_truth: np.ndarray) -> float:
        preds = predicted_probs.flatten()
        gts = ground_truth.flatten()
        return float(np.mean((preds - gts) ** 2))

    def auroc(self, predicted_probs: np.ndarray, ground_truth: np.ndarray) -> float:
        preds = predicted_probs.flatten()
        gts = ground_truth.flatten()
        if len(np.unique(gts)) < 2:
            return 0.5
        return float(roc_auc_score(gts, preds))

    def reliability_diagram_data(self, predicted_probs: np.ndarray, ground_truth: np.ndarray) -> Dict[str, np.ndarray]:
        preds = predicted_probs.flatten()
        gts = ground_truth.flatten()
        bin_accs, bin_confs, bin_counts = self._get_bins(preds, gts)
        return {
            'bin_centers': bin_confs,
            'bin_accuracies': bin_accs,
            'bin_counts': bin_counts
        }

    def sparsification_error(self, predicted_probs: np.ndarray, ground_truth: np.ndarray, uncertainty: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Sort by uncertainty descending
        order = np.argsort(uncertainty.flatten())[::-1]
        preds = predicted_probs.flatten()[order]
        gts = ground_truth.flatten()[order]
        
        n = len(preds)
        errors = np.abs(preds - gts)
        fractions = np.linspace(0, 1, 100)
        sparsification = []
        for f in fractions:
            keep = int((1 - f) * n)
            if keep == 0:
                sparsification.append(0.0)
            else:
                sparsification.append(np.mean(errors[int(f*n):]))
        return fractions, np.array(sparsification)
