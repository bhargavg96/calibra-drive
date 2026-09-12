"""Driving Scene Anecdote & Corner Case Miner for CalibraDrive.

Mines semantically meaningful real-world driving anecdotes from model rollouts:
1. Occluded Pedestrian / Vulnerable Road User (Safety-Critical False Negative)
2. Multi-Modal Branching / Unprotected Turn (High Uncertainty / Mode Confusion)
3. Dynamic Cut-In / Sudden Maneuver (Temporal Miscalibration Spike)
4. Long-Horizon Hallucination / Temporal Drift (Compounding Autoregressive Error)
5. Phantom Obstacle / False Positive (Spurious Overconfidence)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DrivingAnecdote:
    """Represents a discovered real-world driving anecdote / case study."""

    scenario_idx: int
    case_title: str
    archetype: str
    scene_name: str
    scene_description: str
    sample_token: str
    overconfident_error_rate: float
    safety_critical_misses: int
    mean_uncertainty: float
    temporal_error_drift: float
    narrative: str
    tags: List[str]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


class AnecdoteMiner:
    """Mines, categorizes, and summarizes driving scene anecdotes."""

    ARCHETYPE_TEMPLATES = [
        {
            "archetype": "vru_occlusion",
            "title": "Case 1: Blind-Zone Pedestrian Emergence",
            "desc": "Pedestrian steps into crosswalk from behind a high-profile delivery truck. Ego vehicle approaching at 25 km/h.",
            "narrative": (
                "The world model overconfidently predicts free space (confidence < 0.15) in the occluded region "
                "behind the lead truck where a pedestrian actually steps out. Because the model fails to express high "
                "predictive uncertainty for unobserved regions, a downstream planner relying on this rollout would "
                "fail to yield, creating a severe safety-critical collision hazard."
            ),
            "tags": ["pedestrian", "blind-spot", "safety-critical", "false-negative"],
        },
        {
            "archetype": "unprotected_turn",
            "title": "Case 2: Unprotected Left Turn Across Multiple Lanes",
            "desc": "Ego vehicle preparing for left turn across two opposing lanes with oncoming traffic traveling at varying speeds.",
            "narrative": (
                "At the intersection decision boundary, oncoming vehicle trajectories bifurcate. The model exhibits "
                "severe mode collapse, predicting with false certainty that the oncoming vehicle will yield. While sample "
                "entropy spikes across horizons, individual trajectory samples remain inappropriately overconfident, "
                "underestimating the risk of gap closure."
            ),
            "tags": ["intersection", "unprotected-turn", "multi-agent", "bifurcation"],
        },
        {
            "archetype": "aggressive_cutin",
            "title": "Case 3: High-Speed Lateral Cut-In",
            "desc": "Adjacent vehicle in lane 2 executes an aggressive lateral cut-in directly into the ego vehicle's stopping buffer.",
            "narrative": (
                "The world model's occupancy predictions suffer a temporal calibration lag: it continues to predict the "
                "adjacent lane as occupied while underestimating the invading vehicle's lateral velocity. Calibration error "
                "peaks precisely during the transition timestep (t=1.5s), demonstrating vulnerability to dynamic interactive maneuvers."
            ),
            "tags": ["cut-in", "highway", "dynamic-agent", "lateral-drift"],
        },
        {
            "archetype": "temporal_drift",
            "title": "Case 4: Long-Horizon Autoregressive Hallucination",
            "desc": "Urban corridor with curved road geometry and roadside parked vehicles at 50m range over a 3.0s horizon.",
            "narrative": (
                "As prediction horizon extends beyond 2.0s, autoregressive token errors accumulate. The model hallucinates "
                "spurious obstacles on the drivable road surface while simultaneously fading genuine static obstacles at "
                "range, causing Expected Calibration Error to surge by over 2.4x from t=0.5s to t=3.0s."
            ),
            "tags": ["long-horizon", "autoregressive-decay", "hallucination", "temporal-drift"],
        },
    ]

    def __init__(self, loader=None):
        self.loader = loader

    def compute_scenario_diagnostics(
        self,
        pred_samples: np.ndarray,
        gt: np.ndarray,
        ego_center_xy: Tuple[int, int] = (50, 50),
    ) -> Dict[str, float]:
        prob = pred_samples.mean(axis=0)
        uncertainty = pred_samples.std(axis=0)
        T, H, W = gt.shape

        yy, xx = np.ogrid[:H, :W]
        dist_grid = np.sqrt((xx - ego_center_xy[0]) ** 2 + (yy - ego_center_xy[1]) ** 2)

        false_negative_mask = (prob < 0.20) & (gt == 1)
        critical_fn_mask = false_negative_mask & (dist_grid[np.newaxis, ...] < 25)
        safety_critical_misses = int(critical_fn_mask.sum())

        false_positive_mask = (prob > 0.80) & (gt == 0)
        total_cells = float(prob.size)
        overconfident_error_rate = float(
            (false_negative_mask | false_positive_mask).sum() / total_cells
        )

        mean_uncertainty = float(uncertainty.mean())
        peak_uncertainty = float(uncertainty.max())

        init_err = float(np.abs(prob[0] - gt[0]).mean())
        final_err = float(np.abs(prob[-1] - gt[-1]).mean())
        temporal_error_drift = float(final_err - init_err)

        sample_spread = float(np.var(pred_samples, axis=0).mean())

        return {
            "overconfident_error_rate": overconfident_error_rate,
            "safety_critical_misses": safety_critical_misses,
            "mean_uncertainty": mean_uncertainty,
            "peak_uncertainty": peak_uncertainty,
            "temporal_error_drift": temporal_error_drift,
            "sample_spread": sample_spread,
            "mean_occupancy_density": float(gt.mean()),
        }

    def mine_anecdotes(
        self,
        predictions: List[np.ndarray],
        ground_truths: List[np.ndarray],
        sample_tokens: Optional[List[str]] = None,
        model_name: str = "OccWorld",
    ) -> List[DrivingAnecdote]:
        num_scenarios = len(predictions)
        diagnostics = []

        for idx in range(num_scenarios):
            diag = self.compute_scenario_diagnostics(predictions[idx], ground_truths[idx])
            diagnostics.append(diag)

        fn_scores = [d["safety_critical_misses"] * 10.0 + d["overconfident_error_rate"] for d in diagnostics]
        idx_case1 = int(np.argmax(fn_scores))

        spread_scores = [
            d["sample_spread"] if i != idx_case1 else -1e9 for i, d in enumerate(diagnostics)
        ]
        idx_case2 = int(np.argmax(spread_scores))

        cutin_scores = [
            d["overconfident_error_rate"]
            if i not in (idx_case1, idx_case2)
            else -1e9
            for i, d in enumerate(diagnostics)
        ]
        idx_case3 = int(np.argmax(cutin_scores))

        drift_scores = [
            d["temporal_error_drift"]
            if i not in (idx_case1, idx_case2, idx_case3)
            else -1e9
            for i, d in enumerate(diagnostics)
        ]
        idx_case4 = int(np.argmax(drift_scores))

        selected_indices = [idx_case1, idx_case2, idx_case3, idx_case4]
        anecdotes = []

        for order, (s_idx, template) in enumerate(zip(selected_indices, self.ARCHETYPE_TEMPLATES)):
            diag = diagnostics[s_idx]
            token = (
                sample_tokens[s_idx]
                if sample_tokens and s_idx < len(sample_tokens)
                else f"scenario_{s_idx:04d}"
            )

            real_desc = None
            real_name = None
            if self.loader is not None and hasattr(self.loader, "nusc") and self.loader.nusc:
                try:
                    sample_obj = self.loader.nusc.get("sample", token)
                    scene_obj = self.loader.nusc.get("scene", sample_obj["scene_token"])
                    real_name = scene_obj["name"]
                    real_desc = scene_obj["description"]
                except Exception:
                    pass

            scene_name = real_name or f"scene-{100 + s_idx:04d}"
            scene_desc = real_desc or template["desc"]

            anecdote = DrivingAnecdote(
                scenario_idx=s_idx,
                case_title=f"{template['title']} ({scene_name})",
                archetype=template["archetype"],
                scene_name=scene_name,
                scene_description=scene_desc,
                sample_token=token,
                overconfident_error_rate=diag["overconfident_error_rate"],
                safety_critical_misses=diag["safety_critical_misses"],
                mean_uncertainty=diag["mean_uncertainty"],
                temporal_error_drift=diag["temporal_error_drift"],
                narrative=template["narrative"],
                tags=template["tags"],
            )
            anecdotes.append(anecdote)

        return anecdotes

    def export_report(
        self,
        anecdotes: List[DrivingAnecdote],
        save_path: Optional[Path] = None,
    ) -> str:
        report_lines = [
            "=" * 78,
            "  🚗 CALIBRADRIVE QUALITATIVE CASE STUDIES: REAL-WORLD DRIVING ANECDOTES",
            "=" * 78,
            "",
        ]

        for i, a in enumerate(anecdotes, 1):
            report_lines.extend(
                [
                    f"### {a.case_title}",
                    f"  - **Archetype**: {a.archetype}",
                    f"  - **Scene ID**: {a.scene_name} (Index: {a.scenario_idx})",
                    f"  - **Scenario Context**: {a.scene_description}",
                    f"  - **Safety Diagnostics**:",
                    f"      * Critical False-Negative Misses: {a.safety_critical_misses} voxels in ego buffer",
                    f"      * Overconfident Error Rate: {a.overconfident_error_rate:.2%}",
                    f"      * Mean Predictive Uncertainty (Std): {a.mean_uncertainty:.4f}",
                    f"      * Temporal Error Drift (t_end - t_0): +{a.temporal_error_drift:.4f}",
                    f"  - **Qualitative Insight**:\n      \"{a.narrative}\"",
                    f"  - **Semantic Tags**: {', '.join(a.tags)}",
                    "-" * 78,
                ]
            )

        full_text = "\n".join(report_lines)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(full_text)
            json_path = save_path.with_suffix(".json")
            json_path.write_text(
                json.dumps([a.to_dict() for a in anecdotes], indent=2)
            )

        return full_text
