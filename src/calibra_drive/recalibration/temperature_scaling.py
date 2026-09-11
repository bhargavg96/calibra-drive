import numpy as np
from scipy.optimize import minimize

class TemperatureScaling:
    def __init__(self, optimizer='lbfgs', max_iter=100, lr=0.01):
        self._temperature = 1.0
        self.max_iter = max_iter

    def _nll(self, T: float, logits: np.ndarray, labels: np.ndarray) -> float:
        scaled_logits = logits / T
        probs = 1 / (1 + np.exp(-scaled_logits))
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        return -np.sum(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))

    def fit(self, logits: np.ndarray, labels: np.ndarray):
        logits = logits.flatten()
        labels = labels.flatten()
        
        res = minimize(
            self._nll, 
            x0=[1.5], 
            args=(logits, labels), 
            bounds=[(0.01, 100.0)], 
            method='L-BFGS-B',
            options={'maxiter': self.max_iter}
        )
        self._temperature = float(res.x[0])
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-(logits / self._temperature)))

    @property
    def temperature(self) -> float:
        return self._temperature
