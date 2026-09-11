import numpy as np
from typing import Dict, List
from ..models.base_wrapper import WorldModelWrapper, PredictionBundle

class UncertaintyAwarePlanner:
    def __init__(self, world_model: WorldModelWrapper, cost_weights: Dict[str, float], risk_lambda: float = 0.0):
        self.world_model = world_model
        self.cost_weights = cost_weights
        self.risk_lambda = risk_lambda

    def plan(self, context: dict, candidate_actions: List[dict], n_samples: int = 20) -> dict:
        best_action = None
        best_cost = float('inf')
        
        for action in candidate_actions:
            bundle = self.world_model.predict(context, action, n_samples)
            base_cost = self._compute_cost(bundle, action)
            risk = self._compute_uncertainty_penalty(bundle)
            total_cost = base_cost + self.risk_lambda * risk
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_action = action
                
        return {'action': best_action, 'cost': best_cost}

    def _compute_cost(self, prediction: PredictionBundle, action: dict) -> float:
        return 0.0

    def _compute_uncertainty_penalty(self, prediction: PredictionBundle) -> float:
        if prediction.occupancy_samples is not None:
            variance = np.var(prediction.occupancy_samples, axis=0)
            return float(np.mean(variance))
        return 0.0
