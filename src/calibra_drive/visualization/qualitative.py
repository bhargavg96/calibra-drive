import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class QualitativeVisualizer:
    def plot_occupancy_with_uncertainty(self, occupancy_probs: np.ndarray, ground_truth: np.ndarray, save_path: Path = None):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        ax1.imshow(ground_truth.max(axis=-1), cmap='gray')
        ax1.set_title('Ground Truth')
        ax2.imshow(occupancy_probs.max(axis=-1), cmap='viridis')
        ax2.set_title('Predicted Probabilities')
        uncertainty = occupancy_probs * (1 - occupancy_probs)
        ax3.imshow(uncertainty.max(axis=-1), cmap='hot')
        ax3.set_title('Uncertainty (Var)')
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
        return fig

    def plot_prediction_samples(self, samples: np.ndarray, ground_truth: np.ndarray, save_path: Path = None):
        n = min(samples.shape[0], 5)
        fig, axes = plt.subplots(1, n+1, figsize=(3*(n+1), 3))
        axes[0].imshow(ground_truth.max(axis=-1), cmap='gray')
        axes[0].set_title('Ground Truth')
        for i in range(n):
            axes[i+1].imshow(samples[i].max(axis=-1), cmap='viridis')
            axes[i+1].set_title(f'Sample {i+1}')
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
        return fig
