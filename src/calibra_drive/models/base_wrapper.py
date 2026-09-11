from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

@dataclass
class PredictionBundle:
    occupancy_samples: Optional[np.ndarray]
    trajectory_samples: Optional[np.ndarray]
    timestamps: np.ndarray
    metadata: Dict[str, Any]

class WorldModelWrapper(ABC):
    @abstractmethod
    def predict(self, context: dict, action: dict, n_samples: int = 20) -> PredictionBundle:
        pass

    def get_occupancy_probs(self, bundle: PredictionBundle) -> np.ndarray:
        if bundle.occupancy_samples is None:
            raise ValueError("No occupancy samples available")
        return np.mean(bundle.occupancy_samples, axis=0)

    def get_trajectory_distribution(self, bundle: PredictionBundle) -> Dict[str, np.ndarray]:
        if bundle.trajectory_samples is None:
            raise ValueError("No trajectory samples available")
        return {
            'mean': np.mean(bundle.trajectory_samples, axis=0),
            'std': np.std(bundle.trajectory_samples, axis=0)
        }

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def output_type(self) -> str:
        pass
