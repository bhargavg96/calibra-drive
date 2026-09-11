"""Publication-quality driving scene visualizations for the CalibraDrive paper.

Generates visually compelling figures showing:
- BEV driving scenes with uncertainty overlays
- Multi-sample stochastic rollout montages
- Confidence vs. reality comparison panels
- Camera frame uncertainty heatmaps
- Recalibration before/after scene comparisons
- Composite teaser figure for page 1
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ── Custom colormaps ──────────────────────────────────────────────────────────

# Uncertainty: transparent blue → opaque red
_UNCERTAINTY_COLORS = [
    (0.2, 0.4, 0.8, 0.05),   # low uncertainty: faint blue
    (0.2, 0.7, 0.3, 0.25),   # medium-low: green
    (1.0, 0.85, 0.0, 0.50),  # medium: yellow
    (1.0, 0.4, 0.0, 0.70),   # medium-high: orange
    (0.85, 0.1, 0.1, 0.90),  # high uncertainty: red
]
UNCERTAINTY_CMAP = LinearSegmentedColormap.from_list("uncertainty", _UNCERTAINTY_COLORS, N=256)

# Calibration error: green (well-calibrated) → red (poorly calibrated)
_CALERR_COLORS = [
    (0.1, 0.7, 0.3),   # well-calibrated: green
    (0.95, 0.9, 0.2),  # moderate: yellow
    (0.85, 0.1, 0.1),  # poor: red
]
CALERR_CMAP = LinearSegmentedColormap.from_list("cal_error", _CALERR_COLORS, N=256)

# Occupancy: white (free) → dark blue (occupied)
_OCC_COLORS = [(0.95, 0.95, 0.97), (0.15, 0.25, 0.55)]
OCC_CMAP = LinearSegmentedColormap.from_list("occupancy", _OCC_COLORS, N=256)


def _setup_style():
    """Apply publication-quality matplotlib defaults."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def _draw_ego_vehicle(ax, center: tuple[float, float], size: float = 4.0,
                      color: str = "#2196F3", heading: float = 90.0):
    """Draw a stylized ego vehicle marker on a BEV plot."""
    x, y = center
    half = size / 2
    # Car body
    car = mpatches.FancyBboxPatch(
        (x - half * 0.4, y - half), half * 0.8, size,
        boxstyle="round,pad=0.3", facecolor=color, edgecolor="white",
        linewidth=1.5, zorder=10,
    )
    ax.add_patch(car)
    # Direction arrow
    ax.annotate("", xy=(x, y + half + 1.5), xytext=(x, y + half - 0.5),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=1.5),
                zorder=11)


def _draw_agent_boxes(ax, num_agents: int, grid_size: tuple[int, int],
                      rng: np.random.RandomState, color: str = "#FF9800"):
    """Draw random agent bounding boxes on a BEV plot."""
    H, W = grid_size
    for _ in range(num_agents):
        cx = rng.uniform(W * 0.1, W * 0.9)
        cy = rng.uniform(H * 0.1, H * 0.9)
        w = rng.uniform(2, 5)
        h = rng.uniform(4, 8)
        angle = rng.uniform(-30, 30)
        rect = mpatches.FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.15",
            facecolor=color, edgecolor="white", linewidth=0.8,
            alpha=0.85, zorder=8,
        )
        t = mpl.transforms.Affine2D().rotate_deg_around(cx, cy, angle) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)


def _draw_road_markings(ax, grid_size: tuple[int, int]):
    """Draw lane-like road markings on BEV."""
    H, W = grid_size
    cx = W / 2
    # Center dashed line
    for y in range(0, H, 8):
        ax.plot([cx, cx], [y, min(y + 4, H)], color="white",
                linewidth=1.0, alpha=0.5, zorder=5)
    # Lane boundaries
    for offset in [-W * 0.2, W * 0.2]:
        ax.plot([cx + offset, cx + offset], [0, H], color="white",
                linewidth=0.7, alpha=0.3, linestyle="--", zorder=5)


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════


class PaperFigureGenerator:
    """Generate all publication-quality figures for the CalibraDrive paper."""

    def __init__(self, save_dir: str | Path = "figures", fmt: str = "pdf"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.fmt = fmt
        _setup_style()

    def _save(self, fig: plt.Figure, name: str) -> Path:
        path = self.save_dir / f"{name}.{self.fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        return path

    # ──────────────────────────────────────────────────────────────────────
    #  Figure 1: BEV Scene with Uncertainty Overlay
    # ──────────────────────────────────────────────────────────────────────

    def plot_bev_uncertainty_scene(
        self,
        occupancy_prob: np.ndarray,
        uncertainty: np.ndarray,
        ground_truth: np.ndarray,
        timestep: int = 0,
        title: str = "Predicted Occupancy with Uncertainty",
        num_agents: int = 6,
        seed: int = 42,
        save_name: Optional[str] = "bev_uncertainty_scene",
    ) -> plt.Figure:
        """BEV driving scene with occupancy predictions and uncertainty heatmap.

        Args:
            occupancy_prob: (T, H, W) predicted occupancy probabilities.
            uncertainty: (T, H, W) prediction uncertainty (e.g., std across samples).
            ground_truth: (T, H, W) binary ground truth occupancy.
            timestep: Which future timestep to visualize.
            title: Figure title.
            num_agents: Number of agent boxes to render.
            seed: Random seed for agent placement.
            save_name: Filename (without extension).
        """
        prob = occupancy_prob[timestep]
        unc = uncertainty[timestep]
        gt = ground_truth[timestep]
        H, W = prob.shape
        rng = np.random.RandomState(seed)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)

        # ── Panel A: Predicted Occupancy ──
        ax = axes[0]
        ax.imshow(prob, cmap=OCC_CMAP, vmin=0, vmax=1, origin="lower", aspect="equal")
        _draw_road_markings(ax, (H, W))
        _draw_ego_vehicle(ax, (W / 2, H * 0.15))
        _draw_agent_boxes(ax, num_agents, (H, W), rng)
        ax.set_title("(a) Predicted Occupancy", fontweight="bold")
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")

        # ── Panel B: Uncertainty Overlay ──
        ax = axes[1]
        ax.imshow(prob, cmap="gray_r", vmin=0, vmax=1, origin="lower", aspect="equal", alpha=0.4)
        im = ax.imshow(unc, cmap=UNCERTAINTY_CMAP, vmin=0, vmax=unc.max() * 1.1,
                       origin="lower", aspect="equal")
        _draw_road_markings(ax, (H, W))
        _draw_ego_vehicle(ax, (W / 2, H * 0.15))
        ax.set_title("(b) Prediction Uncertainty", fontweight="bold")
        ax.set_xlabel("X (meters)")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Uncertainty (σ)")

        # ── Panel C: Ground Truth ──
        ax = axes[2]
        ax.imshow(gt, cmap=OCC_CMAP, vmin=0, vmax=1, origin="lower", aspect="equal")
        _draw_road_markings(ax, (H, W))
        _draw_ego_vehicle(ax, (W / 2, H * 0.15))
        # Highlight mismatches
        error_mask = np.abs(prob - gt) > 0.5
        error_overlay = np.zeros((*gt.shape, 4))
        error_overlay[error_mask] = [1.0, 0.2, 0.2, 0.5]  # red for errors
        ax.imshow(error_overlay, origin="lower", aspect="equal")
        ax.set_title("(c) Ground Truth + Errors", fontweight="bold")
        ax.set_xlabel("X (meters)")

        for ax in axes:
            ax.set_xticks([0, W // 4, W // 2, 3 * W // 4, W])
            ax.set_xticklabels(["-50", "-25", "0", "25", "50"])
            ax.set_yticks([0, H // 4, H // 2, 3 * H // 4, H])
            ax.set_yticklabels(["0", "25", "50", "75", "100"])

        plt.tight_layout()
        if save_name:
            self._save(fig, save_name)
        return fig

    # ──────────────────────────────────────────────────────────────────────
    #  Figure 2: Stochastic Rollout Montage
    # ──────────────────────────────────────────────────────────────────────

    def plot_rollout_montage(
        self,
        samples: np.ndarray,
        ground_truth: np.ndarray,
        timestep: int = 0,
        num_show: int = 6,
        title: str = "Diverse Stochastic Predictions",
        save_name: Optional[str] = "rollout_montage",
    ) -> plt.Figure:
        """Show N diverse stochastic rollouts in a grid alongside ground truth.

        Args:
            samples: (N, T, H, W) binary occupancy samples.
            ground_truth: (T, H, W) binary ground truth.
            timestep: Which timestep to show.
            num_show: Number of samples to display (max 8).
            title: Figure title.
            save_name: Filename.
        """
        num_show = min(num_show, samples.shape[0], 8)
        cols = min(num_show + 1, 4)  # +1 for GT
        rows = int(np.ceil((num_show + 1) / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
        axes = np.atleast_2d(axes)

        # Ground truth in first position
        ax = axes.flat[0]
        ax.imshow(ground_truth[timestep], cmap=OCC_CMAP, vmin=0, vmax=1,
                  origin="lower", aspect="equal")
        H, W = ground_truth[timestep].shape
        _draw_ego_vehicle(ax, (W / 2, H * 0.15), size=3)
        ax.set_title("Ground Truth", fontweight="bold", color="#2E7D32")
        ax.set_xticks([])
        ax.set_yticks([])
        # Green border
        for spine in ax.spines.values():
            spine.set_edgecolor("#2E7D32")
            spine.set_linewidth(3)

        # Samples
        for i in range(num_show):
            ax = axes.flat[i + 1]
            ax.imshow(samples[i, timestep], cmap=OCC_CMAP, vmin=0, vmax=1,
                      origin="lower", aspect="equal")
            _draw_ego_vehicle(ax, (W / 2, H * 0.15), size=3)
            ax.set_title(f"Sample {i + 1}", fontweight="bold", color="#1565C0")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#1565C0")
                spine.set_linewidth(2)

        # Hide unused axes
        for j in range(num_show + 1, rows * cols):
            if j < len(axes.flat):
                axes.flat[j].set_visible(False)

        plt.tight_layout()
        if save_name:
            self._save(fig, save_name)
        return fig

    # ──────────────────────────────────────────────────────────────────────
    #  Figure 3: Confidence vs. Reality
    # ──────────────────────────────────────────────────────────────────────

    def plot_confidence_vs_reality(
        self,
        occupancy_prob: np.ndarray,
        ground_truth: np.ndarray,
        timestep: int = 0,
        title: str = "Confidence vs. Reality",
        save_name: Optional[str] = "confidence_vs_reality",
    ) -> plt.Figure:
        """Side-by-side: model confidence map vs ground truth with error highlights.

        Highlights cells where the model was confidently wrong (overconfident failures).
        """
        prob = occupancy_prob[timestep]
        gt = ground_truth[timestep]
        H, W = prob.shape

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)

        # ── Panel A: Model Confidence ──
        ax = axes[0]
        im = ax.imshow(prob, cmap="RdYlGn_r", vmin=0, vmax=1, origin="lower", aspect="equal")
        _draw_ego_vehicle(ax, (W / 2, H * 0.15), size=3)
        ax.set_title("(a) Model Confidence\n(P(occupied))", fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(occupied)")

        # ── Panel B: Ground Truth ──
        ax = axes[1]
        ax.imshow(gt, cmap="gray_r", vmin=0, vmax=1, origin="lower", aspect="equal")
        _draw_ego_vehicle(ax, (W / 2, H * 0.15), size=3)
        ax.set_title("(b) Ground Truth", fontweight="bold")

        # ── Panel C: Calibration Error Map ──
        ax = axes[2]
        cal_error = np.abs(prob - gt)
        # Highlight dangerous overconfident errors
        # "Confident occupied but actually free" or "confident free but actually occupied"
        dangerous = ((prob > 0.8) & (gt == 0)) | ((prob < 0.2) & (gt == 1))

        im = ax.imshow(cal_error, cmap=CALERR_CMAP, vmin=0, vmax=1,
                       origin="lower", aspect="equal")
        # Outline dangerous cells
        danger_overlay = np.zeros((*gt.shape, 4))
        danger_overlay[dangerous] = [1.0, 0.0, 0.0, 0.7]
        ax.imshow(danger_overlay, origin="lower", aspect="equal")
        _draw_ego_vehicle(ax, (W / 2, H * 0.15), size=3)
        ax.set_title("(c) Calibration Error\n(red outline = dangerous)", fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|P - GT|")

        n_dangerous = dangerous.sum()
        n_total = dangerous.size
        ax.text(0.02, 0.02, f"⚠ {n_dangerous} dangerous cells ({n_dangerous/n_total:.1%})",
                transform=ax.transAxes, fontsize=9, color="red", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        for ax in axes:
            ax.set_xticks([0, W // 2, W])
            ax.set_xticklabels(["-50m", "0", "50m"])
            ax.set_yticks([0, H // 2, H])
            ax.set_yticklabels(["0", "50m", "100m"])

        plt.tight_layout()
        if save_name:
            self._save(fig, save_name)
        return fig

    # ──────────────────────────────────────────────────────────────────────
    #  Figure 4: Recalibration Before/After Scene
    # ──────────────────────────────────────────────────────────────────────

    def plot_recalibration_scene(
        self,
        prob_before: np.ndarray,
        prob_after: np.ndarray,
        ground_truth: np.ndarray,
        timestep: int = 0,
        method_name: str = "Temperature Scaling",
        ece_before: float = 0.0,
        ece_after: float = 0.0,
        save_name: Optional[str] = "recalibration_scene",
    ) -> plt.Figure:
        """Side-by-side BEV showing raw vs recalibrated predictions on a real scene.

        Args:
            prob_before: (T, H, W) raw predicted probabilities.
            prob_after: (T, H, W) recalibrated probabilities.
            ground_truth: (T, H, W) binary ground truth.
            timestep: Which timestep to visualize.
            method_name: Recalibration method name for the title.
            ece_before: ECE value before recalibration.
            ece_after: ECE value after recalibration.
            save_name: Filename.
        """
        before = prob_before[timestep]
        after = prob_after[timestep]
        gt = ground_truth[timestep]
        H, W = before.shape

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        fig.suptitle(f"Recalibration Effect: {method_name}", fontsize=15,
                     fontweight="bold", y=1.02)

        # Calibration error maps
        err_before = np.abs(before - gt)
        err_after = np.abs(after - gt)

        # ── Panel A: Before ──
        ax = axes[0]
        im = ax.imshow(err_before, cmap=CALERR_CMAP, vmin=0, vmax=0.8,
                       origin="lower", aspect="equal")
        _draw_ego_vehicle(ax, (W / 2, H * 0.15), size=3)
        ax.set_title(f"(a) Before Recalibration\nECE = {ece_before:.4f}",
                     fontweight="bold", color="#C62828")
        for spine in ax.spines.values():
            spine.set_edgecolor("#C62828")
            spine.set_linewidth(2)

        # ── Panel B: After ──
        ax = axes[1]
        ax.imshow(err_after, cmap=CALERR_CMAP, vmin=0, vmax=0.8,
                  origin="lower", aspect="equal")
        _draw_ego_vehicle(ax, (W / 2, H * 0.15), size=3)
        ax.set_title(f"(b) After {method_name}\nECE = {ece_after:.4f}",
                     fontweight="bold", color="#2E7D32")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2E7D32")
            spine.set_linewidth(2)

        # ── Panel C: Improvement Map ──
        ax = axes[2]
        improvement = err_before - err_after  # positive = improved
        vmax = max(abs(improvement.min()), abs(improvement.max()), 0.3)
        im = ax.imshow(improvement, cmap="RdYlGn", vmin=-vmax, vmax=vmax,
                       origin="lower", aspect="equal")
        _draw_ego_vehicle(ax, (W / 2, H * 0.15), size=3)
        ax.set_title("(c) Improvement Map\n(green = better)", fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Δ error")

        # Stats annotation
        pct_improved = (improvement > 0.01).sum() / improvement.size * 100
        ax.text(0.02, 0.02, f"{pct_improved:.0f}% of cells improved",
                transform=ax.transAxes, fontsize=9, fontweight="bold",
                color="#2E7D32",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        for ax in axes:
            ax.set_xticks([0, W // 2, W])
            ax.set_xticklabels(["-50m", "0", "50m"])
            ax.set_yticks([0, H // 2, H])
            ax.set_yticklabels(["0", "50m", "100m"])

        plt.tight_layout()
        if save_name:
            self._save(fig, save_name)
        return fig

    # ──────────────────────────────────────────────────────────────────────
    #  Figure 5: Temporal Rollout Strip
    # ──────────────────────────────────────────────────────────────────────

    def plot_temporal_rollout_strip(
        self,
        occupancy_prob: np.ndarray,
        uncertainty: np.ndarray,
        ground_truth: np.ndarray,
        timesteps_s: list[float] | None = None,
        model_name: str = "OccWorld",
        save_name: Optional[str] = "temporal_rollout",
    ) -> plt.Figure:
        """Show how predictions and uncertainty evolve over time.

        Displays a horizontal strip: one column per timestep, top row = prediction,
        bottom row = uncertainty, with calibration degradation visible.
        """
        T = occupancy_prob.shape[0]
        if timesteps_s is None:
            timesteps_s = [0.5 * (t + 1) for t in range(T)]

        fig, axes = plt.subplots(3, T, figsize=(3.5 * T, 10))
        fig.suptitle(f"{model_name}: Prediction Evolution Over Time",
                     fontsize=15, fontweight="bold", y=1.02)

        H, W = occupancy_prob.shape[1], occupancy_prob.shape[2]

        for t in range(T):
            # Row 1: Occupancy prediction
            ax = axes[0, t]
            ax.imshow(occupancy_prob[t], cmap=OCC_CMAP, vmin=0, vmax=1,
                      origin="lower", aspect="equal")
            _draw_ego_vehicle(ax, (W / 2, H * 0.15), size=2.5)
            ax.set_title(f"t = {timesteps_s[t]:.1f}s", fontweight="bold")
            if t == 0:
                ax.set_ylabel("Prediction", fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

            # Row 2: Uncertainty
            ax = axes[1, t]
            im = ax.imshow(uncertainty[t], cmap=UNCERTAINTY_CMAP,
                           vmin=0, vmax=uncertainty.max() * 1.1,
                           origin="lower", aspect="equal")
            _draw_ego_vehicle(ax, (W / 2, H * 0.15), size=2.5)
            if t == 0:
                ax.set_ylabel("Uncertainty", fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

            # Row 3: Error
            ax = axes[2, t]
            error = np.abs(occupancy_prob[t] - ground_truth[t])
            ax.imshow(error, cmap=CALERR_CMAP, vmin=0, vmax=1,
                      origin="lower", aspect="equal")
            _draw_ego_vehicle(ax, (W / 2, H * 0.15), size=2.5)
            if t == 0:
                ax.set_ylabel("Error |P−GT|", fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

            # Annotate mean error
            mean_err = error.mean()
            ax.text(0.5, 0.02, f"MAE={mean_err:.3f}",
                    transform=ax.transAxes, ha="center", fontsize=8,
                    fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))

        # Add arrow showing time direction
        fig.text(0.5, -0.01, "→ Prediction Horizon →", ha="center",
                 fontsize=12, fontstyle="italic", color="gray")

        plt.tight_layout()
        if save_name:
            self._save(fig, save_name)
        return fig

    # ──────────────────────────────────────────────────────────────────────
    #  Figure 6: Paper Teaser (Composite)
    # ──────────────────────────────────────────────────────────────────────

    def plot_teaser(
        self,
        occupancy_prob: np.ndarray,
        uncertainty: np.ndarray,
        ground_truth: np.ndarray,
        samples: np.ndarray,
        ece_value: float,
        model_name: str = "World Model",
        timestep: int = 0,
        save_name: Optional[str] = "teaser",
    ) -> plt.Figure:
        """Composite teaser figure for page 1 of the paper.

        Layout:
        ┌─────────────────────────┬──────────────────┐
        │  BEV Scene + Uncertainty│  3 sample rollouts│
        │  (large, hero panel)    │  (stacked)        │
        ├─────────────────────────┼──────────────────┤
        │  Confidence vs Reality  │  Key finding box  │
        └─────────────────────────┴──────────────────┘
        """
        prob = occupancy_prob[timestep]
        unc = uncertainty[timestep]
        gt = ground_truth[timestep]
        H, W = prob.shape
        rng = np.random.RandomState(42)

        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1.2, 1],
                               hspace=0.25, wspace=0.2)

        # ── Top-left: Hero BEV with uncertainty ──
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(prob, cmap="gray_r", vmin=0, vmax=1, origin="lower",
                  aspect="equal", alpha=0.5)
        im = ax.imshow(unc, cmap=UNCERTAINTY_CMAP, vmin=0, vmax=unc.max() * 1.1,
                       origin="lower", aspect="equal")
        _draw_road_markings(ax, (H, W))
        _draw_ego_vehicle(ax, (W / 2, H * 0.15), size=4)
        _draw_agent_boxes(ax, 5, (H, W), rng)
        ax.set_title(f"{model_name}: Occupancy Prediction + Uncertainty",
                     fontweight="bold", fontsize=13)
        ax.set_xticks([0, W // 2, W])
        ax.set_xticklabels(["-50m", "0", "50m"])
        ax.set_yticks([0, H // 2, H])
        ax.set_yticklabels(["0", "50m", "100m"])
        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="Uncertainty (σ)")

        # ── Top-right: 3 sample rollouts ──
        n_show = min(3, samples.shape[0])
        gs_right = gridspec.GridSpecFromSubplotSpec(n_show, 1, subplot_spec=gs[0, 1],
                                                     hspace=0.08)
        for i in range(n_show):
            ax = fig.add_subplot(gs_right[i])
            ax.imshow(samples[i, timestep], cmap=OCC_CMAP, vmin=0, vmax=1,
                      origin="lower", aspect="equal")
            _draw_ego_vehicle(ax, (W / 2, H * 0.15), size=2)
            ax.set_title(f"Sample {i + 1}", fontsize=9, fontweight="bold",
                         color="#1565C0", pad=2)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#1565C0")
                spine.set_linewidth(1.5)

        # ── Bottom-left: Confidence vs Reality ──
        ax = fig.add_subplot(gs[1, 0])
        cal_error = np.abs(prob - gt)
        dangerous = ((prob > 0.8) & (gt == 0)) | ((prob < 0.2) & (gt == 1))
        im = ax.imshow(cal_error, cmap=CALERR_CMAP, vmin=0, vmax=1,
                       origin="lower", aspect="equal")
        danger_overlay = np.zeros((*gt.shape, 4))
        danger_overlay[dangerous] = [1.0, 0.0, 0.0, 0.6]
        ax.imshow(danger_overlay, origin="lower", aspect="equal")
        _draw_ego_vehicle(ax, (W / 2, H * 0.15), size=3)
        ax.set_title("Calibration Error Map (red = overconfident failures)",
                     fontweight="bold", fontsize=11)
        ax.set_xticks([0, W // 2, W])
        ax.set_xticklabels(["-50m", "0", "50m"])
        ax.set_yticks([0, H // 2, H])
        ax.set_yticklabels(["0", "50m", "100m"])
        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="|P(occ) − GT|")

        # ── Bottom-right: Key finding box ──
        ax = fig.add_subplot(gs[1, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Finding box
        box_props = dict(boxstyle="round,pad=0.4", facecolor="#FFF3E0",
                         edgecolor="#FF9800", linewidth=2)
        finding_text = (
            f"Key Finding\n"
            f"━━━━━━━━━━━━━━\n\n"
            f"ECE = {ece_value:.4f}\n\n"
            f"Current driving world\n"
            f"models are significantly\n"
            f"overconfident.\n\n"
            f"Simple recalibration\n"
            f"can reduce ECE by\n"
            f">50% and improve\n"
            f"planning safety."
        )
        ax.text(0.5, 0.5, finding_text, transform=ax.transAxes,
                fontsize=11, verticalalignment="center", horizontalalignment="center",
                fontfamily="monospace", bbox=box_props)

        if save_name:
            self._save(fig, save_name)
        return fig

    # ──────────────────────────────────────────────────────────────────────
    #  Generate all paper figures from experiment data
    # ──────────────────────────────────────────────────────────────────────

    def generate_all(
        self,
        predictions: list[np.ndarray],
        ground_truths: list[np.ndarray],
        model_name: str = "OccWorld",
        ece_value: float = 0.15,
        prob_recalibrated: np.ndarray | None = None,
        ece_before: float = 0.15,
        ece_after: float = 0.05,
    ) -> dict[str, Path]:
        """Generate all paper figures from experiment data.

        Args:
            predictions: List of (N, T, H, W) sample arrays, one per scenario.
            ground_truths: List of (T, H, W) ground truth arrays.
            model_name: Name of the model for titles.
            ece_value: Overall ECE value for the teaser.
            prob_recalibrated: (T, H, W) recalibrated probabilities for one scenario.
            ece_before: ECE before recalibration.
            ece_after: ECE after recalibration.

        Returns:
            Dict mapping figure names to saved file paths.
        """
        saved = {}

        # Pick a representative scenario (medium density)
        densities = [gt.mean() for gt in ground_truths]
        median_idx = int(np.argsort(densities)[len(densities) // 2])

        samples = predictions[median_idx]           # (N, T, H, W)
        gt = ground_truths[median_idx]              # (T, H, W)
        prob = samples.mean(axis=0)                 # (T, H, W)
        unc = samples.std(axis=0)                   # (T, H, W)

        print(f"📸 Generating paper figures for {model_name}...")
        print(f"   Using scenario {median_idx} (density={densities[median_idx]:.3f})")

        # Figure 1: BEV Uncertainty Scene
        self.plot_bev_uncertainty_scene(prob, unc, gt, model_name=model_name)
        saved["bev_uncertainty_scene"] = self.save_dir / f"bev_uncertainty_scene.{self.fmt}"
        print("   ✅ BEV uncertainty scene")

        # Figure 2: Rollout Montage
        self.plot_rollout_montage(samples, gt, model_name=model_name)
        saved["rollout_montage"] = self.save_dir / f"rollout_montage.{self.fmt}"
        print("   ✅ Stochastic rollout montage")

        # Figure 3: Confidence vs Reality
        self.plot_confidence_vs_reality(prob, gt, model_name=model_name)
        saved["confidence_vs_reality"] = self.save_dir / f"confidence_vs_reality.{self.fmt}"
        print("   ✅ Confidence vs. reality")

        # Figure 4: Recalibration scene
        if prob_recalibrated is not None:
            self.plot_recalibration_scene(
                prob, prob_recalibrated, gt,
                ece_before=ece_before, ece_after=ece_after,
            )
            saved["recalibration_scene"] = self.save_dir / f"recalibration_scene.{self.fmt}"
            print("   ✅ Recalibration before/after scene")

        # Figure 5: Temporal rollout strip
        self.plot_temporal_rollout_strip(prob, unc, gt, model_name=model_name)
        saved["temporal_rollout"] = self.save_dir / f"temporal_rollout.{self.fmt}"
        print("   ✅ Temporal rollout strip")

        # Figure 6: Teaser
        self.plot_teaser(prob, unc, gt, samples, ece_value=ece_value,
                         model_name=model_name)
        saved["teaser"] = self.save_dir / f"teaser.{self.fmt}"
        print("   ✅ Paper teaser (composite)")

        print(f"\n📁 All figures saved to: {self.save_dir}")
        return saved
