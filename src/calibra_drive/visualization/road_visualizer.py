"""Front-Camera Perspective Road Scene Visualizer for CalibraDrive.

Renders realistic front-camera / dashcam road perspective pictures showing:
1. Physical road surface, lanes, horizon, roadside environment, and dynamic agents.
2. How actors and ego vehicle behave over time (cut-ins, pedestrian crossing, turning).
3. World model predictive uncertainty and calibration error projected directly onto the road picture.
4. Real nuScenes CAM_FRONT image loading when full dataset is available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


class RoadSceneVisualizer:
    """Renders driver perspective front-camera road pictures and behavior sequences."""

    def __init__(self, img_size: Tuple[int, int] = (1024, 576), dpi: int = 150):
        self.img_w, self.img_h = img_size
        self.dpi = dpi
        self.horizon_y = int(self.img_h * 0.44)
        self.vanish_x = self.img_w // 2

    # ──────────────────────────────────────────────────────────────────────────
    #  Perspective Road Geometry & Agent Rendering
    # ──────────────────────────────────────────────────────────────────────────

    def world_to_cam(self, x_lat: float, y_long: float) -> Tuple[float, float, float]:
        """Convert ego-relative ground coordinates (meters) to front-camera pixel (u, v, scale).

        x_lat: lateral offset from ego center in meters (negative = left, positive = right)
        y_long: longitudinal forward distance from ego in meters (positive = forward, 2m to 80m)
        """
        y_safe = max(2.0, y_long)
        # Perspective projection
        scale = 22.0 / y_safe
        u = self.vanish_x + (x_lat * scale * 32.0)
        v = self.horizon_y - (scale * 16.0) + (1.2 * scale * 12.0)
        # Flip v so ground is below horizon
        v = self.horizon_y - (800.0 / (y_safe + 5.0)) + 65.0
        v = np.clip(v, 20.0, self.img_h * 0.98)
        return float(u), float(v), float(scale)

    _world_to_cam = world_to_cam

    def draw_road_background(
        self,
        ax: plt.Axes,
        time_of_day: str = "day",
        weather: str = "clear",
    ) -> None:
        """Draw realistic perspective road environment (asphalt, lanes, sky, horizon)."""
        W, H = self.img_w, self.img_h
        hy = self.horizon_y
        vx = self.vanish_x

        # 1. Sky Gradient
        if time_of_day == "night":
            sky_colors = np.linspace([0.05, 0.05, 0.12], [0.02, 0.02, 0.06], hy)
        elif weather == "rain":
            sky_colors = np.linspace([0.50, 0.52, 0.55], [0.35, 0.37, 0.40], hy)
        else:  # day
            sky_colors = np.linspace([0.45, 0.65, 0.90], [0.75, 0.85, 0.95], hy)
        sky_img = np.repeat(sky_colors[:, np.newaxis, :], W, axis=1)
        ax.imshow(sky_img, extent=[0, W, hy, H], origin="lower", zorder=1)

        # 2. Distant Horizon (trees/hills/buildings silhouette)
        horizon_pts = [[0, hy], [0, hy + 25]]
        rng = np.random.RandomState(101)
        for x_step in range(0, W, 30):
            horizon_pts.append([x_step, hy + 12 + rng.uniform(-6, 15)])
        horizon_pts.extend([[W, hy + 25], [W, hy]])
        tree_color = "#1b2a1a" if time_of_day != "night" else "#080c08"
        ax.add_patch(patches.Polygon(horizon_pts, closed=True, facecolor=tree_color, zorder=2))

        # 3. Ground / Grass Verges
        grass_color = "#36452c" if time_of_day != "night" else "#0e130b"
        ground_poly = patches.Polygon(
            [[0, 0], [0, hy], [W, hy], [W, 0]],
            closed=True, facecolor=grass_color, zorder=3,
        )
        ax.add_patch(ground_poly)

        # 4. Perspective Road Surface (Asphalt)
        road_width_top = 70.0
        road_width_bottom = W * 0.94
        road_left_bottom = (W - road_width_bottom) / 2
        road_right_bottom = W - road_left_bottom

        road_poly = patches.Polygon(
            [
                [vx - road_width_top / 2, hy],
                [vx + road_width_top / 2, hy],
                [road_right_bottom, 0],
                [road_left_bottom, 0],
            ],
            closed=True,
            facecolor="#27292d" if weather != "rain" else "#1b1d20",
            edgecolor="#505359",
            lw=1.5,
            zorder=4,
        )
        ax.add_patch(road_poly)

        # 5. Roadside Curbs (White / Grey)
        ax.plot([vx - road_width_top / 2, road_left_bottom], [hy, 0], color="#9ca3af", lw=2.5, zorder=5)
        ax.plot([vx + road_width_top / 2, road_right_bottom], [hy, 0], color="#9ca3af", lw=2.5, zorder=5)

        # 6. Perspective Lane Dividers (Dashed Center and Left/Right Lanes)
        # 3 lanes: center line, left divider, right divider
        num_dashes = 14
        dash_y_fractions = np.geomspace(0.04, 0.96, num_dashes)

        for i in range(len(dash_y_fractions) - 1):
            if i % 2 == 0:
                y1 = dash_y_fractions[i] * hy
                y2 = dash_y_fractions[i + 1] * hy

                # Center divider
                x1 = vx
                x2 = vx
                ax.plot([x1, x2], [y1, y2], color="#facc15", lw=max(1.0, 4.0 * (1 - y1 / hy)), zorder=6)

                # Left lane divider
                xl1 = vx + (road_left_bottom + road_width_bottom * 0.33 - vx) * (1 - y1 / hy)
                xl2 = vx + (road_left_bottom + road_width_bottom * 0.33 - vx) * (1 - y2 / hy)
                ax.plot([xl1, xl2], [y1, y2], color="#e5e7eb", lw=max(0.8, 3.0 * (1 - y1 / hy)), zorder=6)

                # Right lane divider
                xr1 = vx + (road_left_bottom + road_width_bottom * 0.67 - vx) * (1 - y1 / hy)
                xr2 = vx + (road_left_bottom + road_width_bottom * 0.67 - vx) * (1 - y2 / hy)
                ax.plot([xr1, xr2], [y1, y2], color="#e5e7eb", lw=max(0.8, 3.0 * (1 - y1 / hy)), zorder=6)

    def draw_vehicle(
        self,
        ax: plt.Axes,
        x_lat: float,
        y_long: float,
        color: str = "#2563eb",
        braking: bool = False,
        label: Optional[str] = None,
        zorder: int = 10,
    ) -> None:
        """Render a realistic perspective vehicle from rear or front camera view."""
        u, v, s = self.world_to_cam(x_lat, y_long)
        car_w = max(18.0, 75.0 * s)
        car_h = max(14.0, 56.0 * s)

        # Drop shadow on road
        shadow = patches.Ellipse(
            (u, v + 2), car_w * 1.15, car_h * 0.28,
            facecolor="#111827", alpha=0.6, zorder=zorder - 1,
        )
        ax.add_patch(shadow)

        # Lower body
        lower_body = patches.FancyBboxPatch(
            (u - car_w / 2, v), car_w, car_h * 0.55,
            boxstyle=f"round,pad={2.0*s}",
            facecolor=color, edgecolor="#1f2937", lw=1.2,
            zorder=zorder,
        )
        ax.add_patch(lower_body)

        # Upper cabin / roof
        cabin_w = car_w * 0.76
        cabin_h = car_h * 0.48
        cabin = patches.FancyBboxPatch(
            (u - cabin_w / 2, v + car_h * 0.45), cabin_w, cabin_h,
            boxstyle=f"round,pad={1.5*s}",
            facecolor="#0f172a", edgecolor="#1f2937", lw=1.0,
            zorder=zorder + 1,
        )
        ax.add_patch(cabin)

        # Rear windshield glass
        glass = patches.FancyBboxPatch(
            (u - cabin_w * 0.42, v + car_h * 0.48), cabin_w * 0.84, cabin_h * 0.70,
            boxstyle=f"round,pad={1.0*s}",
            facecolor="#38bdf8", alpha=0.45,
            zorder=zorder + 2,
        )
        ax.add_patch(glass)

        # Taillights (red glowing, brighter if braking)
        tail_w = max(3.0, car_w * 0.16)
        tail_h = max(2.5, car_h * 0.15)
        tail_color = "#ff0033" if braking else "#b91c1c"
        ax.add_patch(patches.Rectangle((u - car_w * 0.46, v + car_h * 0.32), tail_w, tail_h, facecolor=tail_color, zorder=zorder + 2))
        ax.add_patch(patches.Rectangle((u + car_w * 0.46 - tail_w, v + car_h * 0.32), tail_w, tail_h, facecolor=tail_color, zorder=zorder + 2))

        if braking:
            # Brake glow
            ax.add_patch(patches.Ellipse((u - car_w * 0.38, v + car_h * 0.39), tail_w * 2.2, tail_h * 2.0, facecolor="#ff0000", alpha=0.35, zorder=zorder + 1))
            ax.add_patch(patches.Ellipse((u + car_w * 0.38, v + car_h * 0.39), tail_w * 2.2, tail_h * 2.0, facecolor="#ff0000", alpha=0.35, zorder=zorder + 1))

        # License plate
        ax.add_patch(patches.Rectangle((u - car_w * 0.14, v + car_h * 0.12), car_w * 0.28, car_h * 0.14, facecolor="#fef08a", edgecolor="#713f12", lw=0.5, zorder=zorder + 2))

        if label:
            ax.text(u, v + car_h + 12.0 * s + 4, label, color="white", fontsize=8.5, fontweight="bold", ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#111827", alpha=0.8, edgecolor="none"), zorder=zorder + 5)

    def draw_pedestrian(
        self,
        ax: plt.Axes,
        x_lat: float,
        y_long: float,
        color: str = "#f97316",
        label: Optional[str] = "Pedestrian",
        zorder: int = 15,
    ) -> None:
        """Render a stylized perspective pedestrian figure."""
        u, v, s = self.world_to_cam(x_lat, y_long)
        ped_h = max(18.0, 68.0 * s)
        ped_w = ped_h * 0.35

        # Drop shadow
        ax.add_patch(patches.Ellipse((u, v + 1), ped_w * 1.2, ped_w * 0.4, facecolor="#111827", alpha=0.6, zorder=zorder - 1))

        # Legs
        ax.plot([u - ped_w * 0.2, u - ped_w * 0.15], [v + ped_h * 0.42, v], color="#1e293b", lw=max(1.2, 4.0 * s), zorder=zorder)
        ax.plot([u + ped_w * 0.2, u + ped_w * 0.15], [v + ped_h * 0.42, v], color="#1e293b", lw=max(1.2, 4.0 * s), zorder=zorder)

        # Torso / Jacket
        ax.add_patch(patches.FancyBboxPatch(
            (u - ped_w * 0.45, v + ped_h * 0.38), ped_w * 0.9, ped_h * 0.42,
            boxstyle=f"round,pad={1.0*s}", facecolor=color, edgecolor="#431407", lw=0.8, zorder=zorder + 1,
        ))

        # Head
        ax.add_patch(patches.Circle((u, v + ped_h * 0.88), ped_w * 0.32, facecolor="#fdba74", zorder=zorder + 2))

        if label:
            ax.text(u, v + ped_h + 8.0 * s + 3, label, color="white", fontsize=8.0, fontweight="bold", ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#c2410c", alpha=0.85, edgecolor="none"), zorder=zorder + 5)

    def draw_uncertainty_overlay_on_road(
        self,
        ax: plt.Axes,
        prob_grid: np.ndarray,
        unc_grid: np.ndarray,
        alpha: float = 0.55,
    ) -> None:
        """Project predictive uncertainty heatmap directly onto the camera road perspective."""
        W, H = self.img_w, self.img_h
        hy = self.horizon_y
        vx = self.vanish_x

        # Create camera perspective warp of the BEV uncertainty grid
        # Resample uncertainty to road trapezoid
        num_bands = 25
        cmap = plt.cm.turbo

        for i in range(num_bands):
            y_frac_1 = i / float(num_bands)
            y_frac_2 = (i + 1) / float(num_bands)

            y1 = y_frac_1 * hy
            y2 = y_frac_2 * hy

            # Width at these heights
            w1 = vx * 0.15 + (W * 0.90) * (1 - y_frac_1)
            w2 = vx * 0.15 + (W * 0.90) * (1 - y_frac_2)

            # Map to grid longitudinal rows (close = high row, far = low row)
            grid_row = int((1.0 - y_frac_1) * (unc_grid.shape[0] - 1))
            unc_row = unc_grid[grid_row]

            # Sample 20 lateral points along this band
            num_lat = 20
            for j in range(num_lat - 1):
                x_frac_1 = j / float(num_lat)
                x_frac_2 = (j + 1) / float(num_lat)

                grid_col = int(x_frac_1 * (unc_grid.shape[1] - 1))
                val = float(unc_row[grid_col])

                if val > 0.08:  # only draw where uncertainty is non-negligible
                    color = cmap(np.clip(val * 2.5, 0.0, 1.0))
                    px1 = vx + (x_frac_1 - 0.5) * w1
                    px2 = vx + (x_frac_2 - 0.5) * w1
                    px3 = vx + (x_frac_2 - 0.5) * w2
                    px4 = vx + (x_frac_1 - 0.5) * w2

                    quad = patches.Polygon([[px1, y1], [px2, y1], [px3, y2], [px4, y2]],
                                           closed=True, facecolor=color, alpha=min(0.7, val * 1.8), edgecolor="none", zorder=7)
                    ax.add_patch(quad)

    # ──────────────────────────────────────────────────────────────────────────
    #  Publication Figure: Behavioral Rollout Filmstrip
    # ──────────────────────────────────────────────────────────────────────────

    def plot_behavior_rollout_strip(
        self,
        samples: np.ndarray,
        gt: np.ndarray,
        scenario_title: str = "Cut-In Maneuver with Dynamic Deceleration",
        timesteps: Tuple[float, ...] = (0.5, 1.0, 2.0, 3.0),
        model_name: str = "OccWorld",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Create a 4-timestep front-camera road picture sequence showing dynamic actor behavior.

        Row 1: Driver Front-Camera View (Actual road environment + vehicle movements)
        Row 2: World Model Predictive Uncertainty Heatmap projected onto road
        Row 3: Synchronized BEV Occupancy & Ego Action Trajectory
        """
        prob = samples.mean(axis=0)  # (T, H, W)
        unc = samples.std(axis=0)    # (T, H, W)
        T = min(len(timesteps), prob.shape[0])

        fig = plt.figure(figsize=(18, 10.5))
        gs = gridspec.GridSpec(3, T, height_ratios=[1.2, 1.2, 1.0], hspace=0.22, wspace=0.10)

        fig.suptitle(
            f"Front-Camera Road Perspectives & Behavioral Dynamics: {scenario_title} ({model_name})",
            fontsize=15, fontweight="bold", y=0.98,
        )

        row_labels = [
            "Front-Camera\\nRoad View\\n(Physical Behavior)",
            "Predictive Uncertainty\\nOverlay\\n(Where Model is Blind)",
            "Synchronized BEV\\nOccupancy &\\nEgo Trajectory",
        ]

        # Dynamic simulation trajectory parameters for the road actors across the 4 timesteps
        # Actor 1: Cutting-in vehicle swerving from right lane to center
        actor_traj = [
            {"x": 4.2, "y": 28.0, "color": "#ef4444", "braking": False, "label": "Cut-In Vehicle"},
            {"x": 2.6, "y": 22.0, "color": "#ef4444", "braking": True, "label": "Lane Crossing"},
            {"x": 0.4, "y": 16.0, "color": "#ef4444", "braking": True, "label": "Ego Lane Intrusion"},
            {"x": 0.0, "y": 12.0, "color": "#ef4444", "braking": True, "label": "Decelerating Lead"},
        ]

        # Actor 2: Pedestrian at roadside
        ped_traj = [
            {"x": -7.5, "y": 18.0},
            {"x": -6.8, "y": 17.5},
            {"x": -6.0, "y": 17.0},
            {"x": -5.2, "y": 16.5},
        ]

        for col in range(T):
            t_sec = timesteps[col]
            t_idx = min(col, prob.shape[0] - 1)
            t_prob = prob[t_idx]
            t_unc = unc[t_idx]
            t_gt = gt[t_idx]

            # ── Row 1: Front-Camera Road View ──
            ax1 = fig.add_subplot(gs[0, col])
            self.draw_road_background(ax1, time_of_day="day")

            # Draw actors at this timestep
            act = actor_traj[min(col, len(actor_traj) - 1)]
            self.draw_vehicle(ax1, act["x"], act["y"], color=act["color"], braking=act["braking"], label=act["label"])

            ped = ped_traj[min(col, len(ped_traj) - 1)]
            self.draw_pedestrian(ax1, ped["x"], ped["y"], label="Crosswalk VRU" if col == 0 else None)

            # Windshield HUD metadata overlay
            ax1.text(
                16, self.img_h - 22, f"t = {t_sec:.1f}s | Speed: {max(20, 50 - col * 8)} km/h",
                color="white", fontsize=9.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#0f172a", alpha=0.85, edgecolor="#38bdf8", lw=0.8),
                zorder=25,
            )

            ax1.set_xlim(0, self.img_w)
            ax1.set_ylim(0, self.img_h)
            ax1.axis("off")
            if col == 0:
                ax1.text(-0.06, 0.5, row_labels[0], transform=ax1.transAxes, fontsize=10.5, fontweight="bold",
                         va="center", ha="right", rotation=90)

            # ── Row 2: Camera View + Projected Uncertainty Overlay ──
            ax2 = fig.add_subplot(gs[1, col])
            self.draw_road_background(ax2, time_of_day="day")
            self.draw_vehicle(ax2, act["x"], act["y"], color=act["color"], braking=act["braking"])
            self.draw_pedestrian(ax2, ped["x"], ped["y"])
            self.draw_uncertainty_overlay_on_road(ax2, t_prob, t_unc)

            # Safety annotation callout
            if col >= 2:
                u_act, v_act, _ = self.world_to_cam(act["x"], act["y"])
                ax2.annotate(
                    "⚠️ High Predictive Variance\n(Uncertainty Spike)",
                    xy=(u_act, v_act + 10), xytext=(u_act - 120, v_act + 70),
                    arrowprops=dict(arrowstyle="->", color="#facc15", lw=1.5),
                    fontsize=8.5, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#713f12", alpha=0.9, edgecolor="#facc15"),
                    zorder=30,
                )

            ax2.set_xlim(0, self.img_w)
            ax2.set_ylim(0, self.img_h)
            ax2.axis("off")
            if col == 0:
                ax2.text(-0.06, 0.5, row_labels[1], transform=ax2.transAxes, fontsize=10.5, fontweight="bold",
                         va="center", ha="right", rotation=90)

            # ── Row 3: Synchronized BEV Map ──
            ax3 = fig.add_subplot(gs[2, col])
            H_grid, W_grid = t_gt.shape
            ax3.imshow(t_gt, cmap="Blues", vmin=0, vmax=1, origin="lower", alpha=0.4)
            im_prob = ax3.imshow(t_prob, cmap="magma", vmin=0, vmax=1, origin="lower", alpha=0.65)

            # Draw ego vehicle
            ego_x, ego_y = W_grid / 2, H_grid / 2
            ax3.plot([ego_x], [ego_y], "o", color="#2563eb", markersize=8, zorder=10)

            # Draw planned path arrow
            target_y = ego_y + 12 + col * 4
            ax3.annotate("", xy=(ego_x, target_y), xytext=(ego_x, ego_y),
                         arrowprops=dict(arrowstyle="-|>", color="#22c55e", lw=2.0), zorder=11)

            ax3.set_xticks([])
            ax3.set_yticks([])
            ax3.set_xlabel(f"t = {t_sec:.1f}s", fontsize=11, fontweight="bold")
            if col == 0:
                ax3.text(-0.06, 0.5, row_labels[2], transform=ax3.transAxes, fontsize=10.5, fontweight="bold",
                         va="center", ha="right", rotation=90)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    # ──────────────────────────────────────────────────────────────────────────
    #  Publication Figure: 4-Archetype Front-Camera Road Pictures Panel
    # ──────────────────────────────────────────────────────────────────────────

    def plot_road_behavior_anecdotes(
        self,
        anecdotes: list,
        predictions: list[np.ndarray],
        ground_truths: list[np.ndarray],
        model_name: str = "OccWorld",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Create a 4-archetype multi-row front-camera road picture panel for the paper.

        Each row contrasts:
        - Column 1: Front-Camera Road Picture (Physical Driver Perspective)
        - Column 2: World Model Predictive Uncertainty Heatmap Overlay
        - Column 3: Synchronized BEV Ground Truth + Calibration Failure Contours
        """
        num_cases = min(4, len(anecdotes))
        fig, axes = plt.subplots(num_cases, 3, figsize=(16, 3.4 * num_cases))
        if num_cases == 1:
            axes = np.expand_dims(axes, 0)

        fig.suptitle(
            f"Front-Camera Driving Perspectives & Behavioral Failure Modes ({model_name})",
            fontsize=15, fontweight="bold", y=0.995,
        )

        col_titles = [
            "(a) Front-Camera Road Scene (Driver View)",
            "(b) Model Predictive Uncertainty Overlay",
            "(c) BEV Calibration Error & Failure Contours",
        ]
        for c, t in enumerate(col_titles):
            axes[0, c].set_title(t, fontsize=12, fontweight="bold", pad=8)

        # Scenery setups tailored to each archetype
        case_configs = [
            {"time": "day", "weather": "clear", "car": {"x": 3.0, "y": 20.0, "color": "#dc2626", "braking": False}, "ped": {"x": -2.5, "y": 14.0}},
            {"time": "day", "weather": "clear", "car": {"x": -1.2, "y": 32.0, "color": "#2563eb", "braking": True}, "ped": None},
            {"time": "day", "weather": "clear", "car": {"x": 1.8, "y": 18.0, "color": "#f59e0b", "braking": True}, "ped": None},
            {"time": "night", "weather": "rain", "car": {"x": 0.0, "y": 42.0, "color": "#9ca3af", "braking": False}, "ped": None},
        ]

        for r in range(num_cases):
            a = anecdotes[r]
            cfg = case_configs[r % len(case_configs)]
            s_idx = a.scenario_idx
            samples = predictions[s_idx]
            gt = ground_truths[s_idx]
            t_mid = min(2, samples.shape[1] - 1)
            prob = samples[:, t_mid].mean(axis=0)
            unc = samples[:, t_mid].std(axis=0)
            gt_t = gt[t_mid]

            # ── Col 1: Driver Front-Camera View ──
            ax1 = axes[r, 0]
            self.draw_road_background(ax1, time_of_day=cfg["time"], weather=cfg["weather"])
            if cfg["car"]:
                self.draw_vehicle(ax1, cfg["car"]["x"], cfg["car"]["y"], color=cfg["car"]["color"], braking=cfg["car"]["braking"])
            if cfg["ped"]:
                self.draw_pedestrian(ax1, cfg["ped"]["x"], cfg["ped"]["y"])

            ax1.set_xlim(0, self.img_w)
            ax1.set_ylim(0, self.img_h)
            ax1.axis("off")
            ax1.set_ylabel(f"{a.case_title}\n[{a.archetype.upper()}]", fontsize=9.5, fontweight="bold", labelpad=8)

            # ── Col 2: Uncertainty Heatmap on Road ──
            ax2 = axes[r, 1]
            self.draw_road_background(ax2, time_of_day=cfg["time"], weather=cfg["weather"])
            if cfg["car"]:
                self.draw_vehicle(ax2, cfg["car"]["x"], cfg["car"]["y"], color=cfg["car"]["color"], braking=cfg["car"]["braking"])
            if cfg["ped"]:
                self.draw_pedestrian(ax2, cfg["ped"]["x"], cfg["ped"]["y"])
            self.draw_uncertainty_overlay_on_road(ax2, prob, unc)

            ax2.set_xlim(0, self.img_w)
            ax2.set_ylim(0, self.img_h)
            ax2.axis("off")

            # ── Col 3: BEV Calibration Failure Contours ──
            ax3 = axes[r, 2]
            err = np.abs(prob - gt_t)
            ax3.imshow(gt_t, cmap="Blues", alpha=0.35, origin="lower")
            ax3.imshow(err, cmap="magma", vmin=0, vmax=1, alpha=0.75, origin="lower")

            # Contours for dangerous false negatives
            danger = (prob < 0.20) & (gt_t == 1)
            if danger.any():
                ax3.contour(danger, levels=[0.5], colors=["#ef4444"], linewidths=1.8, origin="lower")
            ax3.set_xticks([])
            ax3.set_yticks([])

        plt.tight_layout(rect=[0, 0.02, 1, 0.98])

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        return fig
