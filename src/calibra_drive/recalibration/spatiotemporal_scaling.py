"""Spatio-Temporal Temperature Scaling for Driving World Models.

Extends classical global Platt/temperature scaling to structured 4D driving grids:
1. Spatial Temperature Scaling T(d): Learns distance-dependent temperature scaling
   to account for sensory range decay and far-field overconfidence.
2. Horizon Temperature Scaling T(t): Learns timestep-dependent temperature scaling
   to counteract autoregressive drift and compounding token errors.
3. Joint Spatio-Temporal Scaling T(d, t): Joint parametric recalibration.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _binary_nll(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-7) -> float:
    p = np.clip(probs, eps, 1.0 - eps)
    return float(-np.mean(labels * np.log(p) + (1.0 - labels) * np.log(1.0 - p)))


class SpatioTemporalTemperatureScaling:
    """Parametric recalibration conditioned on spatial distance and prediction horizon."""

    def __init__(
        self,
        mode: str = "joint",  # "global", "spatial", "temporal", "joint"
        num_distance_bins: int = 4,
        distance_edges: Tuple[float, ...] = (0.0, 15.0, 35.0, 60.0, 100.0),
    ):
        self.mode = mode
        self.num_distance_bins = num_distance_bins
        self.distance_edges = np.array(distance_edges)
        self.is_fitted = False

        # Parameters
        self.global_temp: float = 1.0
        self.spatial_temps: np.ndarray = np.ones(len(distance_edges) - 1)
        self.temporal_temps: Optional[np.ndarray] = None
        self.joint_params: Dict[str, float] = {"t0": 1.0, "alpha_dist": 0.01, "beta_time": 0.15}

    def _get_distance_bin_indices(self, distances: np.ndarray) -> np.ndarray:
        """Assign distances (in meters) to bin indices."""
        bins = np.digitize(distances, self.distance_edges) - 1
        return np.clip(bins, 0, len(self.distance_edges) - 2)

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        distances: Optional[np.ndarray] = None,
        timesteps: Optional[np.ndarray] = None,
    ) -> "SpatioTemporalTemperatureScaling":
        """Fit temperature parameters via NLL minimization.

        Args:
            logits: Flattened log-odds (1D array)
            labels: Binary ground truth labels (1D array)
            distances: Radial distance from ego in meters (1D array matching logits)
            timesteps: Future timestep in seconds (1D array matching logits)
        """
        # 1. Global temperature scaling
        def global_obj(t_arr):
            t = t_arr[0]
            scaled_probs = _sigmoid(logits / t)
            return _binary_nll(scaled_probs, labels)

        res_g = minimize(global_obj, [1.5], bounds=[(0.1, 10.0)], method="L-BFGS-B")
        self.global_temp = float(res_g.x[0])

        # 2. Spatial temperature scaling T(d)
        if distances is not None and self.mode in ("spatial", "joint"):
            bin_idxs = self._get_distance_bin_indices(distances)
            num_bins = len(self.distance_edges) - 1
            temps = np.zeros(num_bins)

            for b in range(num_bins):
                mask = bin_idxs == b
                if mask.sum() > 50:
                    b_logits = logits[mask]
                    b_labels = labels[mask]
                    def bin_obj(t_arr):
                        return _binary_nll(_sigmoid(b_logits / t_arr[0]), b_labels)
                    res_b = minimize(bin_obj, [self.global_temp], bounds=[(0.1, 10.0)], method="L-BFGS-B")
                    temps[b] = float(res_b.x[0])
                else:
                    temps[b] = self.global_temp
            self.spatial_temps = temps

        # 3. Temporal temperature scaling T(t)
        if timesteps is not None and self.mode in ("temporal", "joint"):
            unique_times = np.sort(np.unique(timesteps))
            t_temps = []
            for t_val in unique_times:
                mask = np.isclose(timesteps, t_val, atol=0.05)
                if mask.sum() > 50:
                    t_logits = logits[mask]
                    t_labels = labels[mask]
                    def time_obj(t_arr):
                        return _binary_nll(_sigmoid(t_logits / t_arr[0]), t_labels)
                    res_t = minimize(time_obj, [self.global_temp], bounds=[(0.1, 10.0)], method="L-BFGS-B")
                    t_temps.append(float(res_t.x[0]))
                else:
                    t_temps.append(self.global_temp)
            self.temporal_temps = np.array(t_temps)

        # 4. Joint parametric optimization: T(d, t) = t0 + alpha*d + beta*t
        if distances is not None and timesteps is not None and self.mode == "joint":
            def joint_obj(p):
                t0, a_dist, b_time = p
                T_grid = np.clip(t0 + a_dist * distances + b_time * timesteps, 0.2, 10.0)
                scaled = _sigmoid(logits / T_grid)
                return _binary_nll(scaled, labels)

            res_j = minimize(
                joint_obj,
                [self.global_temp, 0.01, 0.15],
                bounds=[(0.2, 8.0), (0.0, 0.2), (0.0, 1.5)],
                method="L-BFGS-B",
            )
            self.joint_params = {
                "t0": float(res_j.x[0]),
                "alpha_dist": float(res_j.x[1]),
                "beta_time": float(res_j.x[2]),
            }

        self.is_fitted = True
        return self

    def transform(
        self,
        logits: np.ndarray,
        distances: Optional[np.ndarray] = None,
        timesteps: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply recalibration to logits."""
        if not self.is_fitted:
            return _sigmoid(logits)

        if self.mode == "global" or distances is None:
            return _sigmoid(logits / self.global_temp)

        if self.mode == "spatial":
            bin_idxs = self._get_distance_bin_indices(distances)
            T_vals = self.spatial_temps[bin_idxs]
            return _sigmoid(logits / T_vals)

        if self.mode == "temporal" and timesteps is not None and self.temporal_temps is not None:
            # Map timesteps to nearest discrete temp
            unique_times = np.linspace(timesteps.min(), timesteps.max(), len(self.temporal_temps))
            time_idxs = np.clip(np.searchsorted(unique_times, timesteps), 0, len(self.temporal_temps) - 1)
            T_vals = self.temporal_temps[time_idxs]
            return _sigmoid(logits / T_vals)

        if self.mode == "joint" and distances is not None and timesteps is not None:
            p = self.joint_params
            T_vals = np.clip(p["t0"] + p["alpha_dist"] * distances + p["beta_time"] * timesteps, 0.2, 10.0)
            return _sigmoid(logits / T_vals)

        return _sigmoid(logits / self.global_temp)
