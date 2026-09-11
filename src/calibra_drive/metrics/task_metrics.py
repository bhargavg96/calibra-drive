import numpy as np

class TaskMetrics:
    @staticmethod
    def iou(pred: np.ndarray, gt: np.ndarray) -> float:
        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        return float(intersection / (union + 1e-6))

    @staticmethod
    def miou(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> float:
        ious = []
        for c in range(num_classes):
            p_c = (pred == c)
            g_c = (gt == c)
            intersection = np.logical_and(p_c, g_c).sum()
            union = np.logical_or(p_c, g_c).sum()
            if union > 0:
                ious.append(intersection / union)
        return float(np.mean(ious)) if ious else 0.0

    @staticmethod
    def ade(pred_traj: np.ndarray, gt_traj: np.ndarray) -> float:
        return float(np.mean(np.linalg.norm(pred_traj - gt_traj, axis=-1)))

    @staticmethod
    def fde(pred_traj: np.ndarray, gt_traj: np.ndarray) -> float:
        return float(np.linalg.norm(pred_traj[-1] - gt_traj[-1]))

    @staticmethod
    def collision_rate(ego_traj: np.ndarray, occupancy_grid: np.ndarray) -> float:
        # Dummy implementation
        return 0.0
