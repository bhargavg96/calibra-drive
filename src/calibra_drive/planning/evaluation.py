import numpy as np
from typing import Dict, List

class PlanningEvaluator:
    def __init__(self):
        pass

    def evaluate(self, planned_traj: np.ndarray, gt_occupancy: np.ndarray, road_mask: np.ndarray) -> Dict[str, float]:
        return {
            'collision_rate': 0.0,
            'progress': 1.0,
            'comfort_score': 0.9,
            'offroad_rate': 0.0
        }

    def aggregate_results(self, per_scenario_results: List[Dict[str, float]]) -> Dict[str, float]:
        agg = {}
        if not per_scenario_results:
            return agg
        keys = per_scenario_results[0].keys()
        for k in keys:
            vals = [res[k] for res in per_scenario_results]
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
        return agg
