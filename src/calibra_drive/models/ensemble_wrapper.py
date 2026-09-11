import numpy as np
from typing import List
from .base_wrapper import WorldModelWrapper, PredictionBundle

class EnsembleWrapper(WorldModelWrapper):
    def __init__(self, models: List[WorldModelWrapper]):
        if not models:
            raise ValueError("Models list cannot be empty")
        self.models = models

    def predict(self, context: dict, action: dict, n_samples: int = 20) -> PredictionBundle:
        samples_per_model = max(1, n_samples // len(self.models))
        occ_samples = []
        for model in self.models:
            bundle = model.predict(context, action, samples_per_model)
            if bundle.occupancy_samples is not None:
                occ_samples.append(bundle.occupancy_samples)
        
        occ_concat = np.concatenate(occ_samples, axis=0) if occ_samples else None
        return PredictionBundle(
            occupancy_samples=occ_concat,
            trajectory_samples=None,
            timestamps=bundle.timestamps,
            metadata={}
        )

    @property
    def name(self) -> str:
        names = ", ".join(m.name for m in self.models)
        return f"Ensemble({names})"

    @property
    def output_type(self) -> str:
        return self.models[0].output_type
