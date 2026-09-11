import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List

class ReliabilityPlotter:
    def __init__(self, figsize=(8,6), style='paper'):
        self.figsize = figsize
        sns.set_theme(style="whitegrid")

    def plot_reliability_diagram(self, diagram_data: dict, model_name: str, save_path: Path = None):
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        ax.plot(diagram_data['bin_centers'], diagram_data['bin_accuracies'], 'o-', label=model_name)
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.legend()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
        return fig

    def plot_multi_model_comparison(self, diagram_data_dict: Dict[str, dict], save_path: Path = None):
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        for name, data in diagram_data_dict.items():
            ax.plot(data['bin_centers'], data['bin_accuracies'], 'o-', label=name)
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.legend()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
        return fig

    def plot_ece_vs_horizon(self, ece_per_horizon: Dict[str, List[float]], horizons: List[float], save_path: Path = None):
        fig, ax = plt.subplots(figsize=self.figsize)
        for name, eces in ece_per_horizon.items():
            ax.plot(horizons, eces, 'o-', label=name)
        ax.set_xlabel('Horizon (s)')
        ax.set_ylabel('ECE')
        ax.legend()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
        return fig

    def plot_ece_by_distance(self, ece_per_distance: Dict[str, List[float]], distance_bins: List[float], save_path: Path = None):
        fig, ax = plt.subplots(figsize=self.figsize)
        for name, eces in ece_per_distance.items():
            x = [(distance_bins[i] + distance_bins[i+1])/2 for i in range(len(distance_bins)-1)]
            ax.plot(x, eces, 'o-', label=name)
        ax.set_xlabel('Distance from Ego (m)')
        ax.set_ylabel('ECE')
        ax.legend()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
        return fig

    def plot_before_after_recalibration(self, before: dict, after: dict, method_name: str, save_path: Path = None):
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        ax.plot(before['bin_centers'], before['bin_accuracies'], 'o-', label='Before')
        ax.plot(after['bin_centers'], after['bin_accuracies'], 's-', label=f'After ({method_name})')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.legend()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
        return fig
