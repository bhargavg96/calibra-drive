import numpy as np
from calibra_drive.visualization.road_visualizer import RoadSceneVisualizer

def test_road_visualizer_camera_projection():
    viz = RoadSceneVisualizer()
    u, v, s = viz.world_to_cam(0.0, 20.0)
    assert 0 < u < viz.img_w
    assert 0 < v < viz.img_h
    assert s > 0

def test_road_visualizer_render_rollout_strip(tmp_path):
    viz = RoadSceneVisualizer()
    rng = np.random.RandomState(42)
    preds = rng.beta(0.5, 0.5, size=(4, 4, 30, 30))
    gts = (rng.rand(4, 30, 30) > 0.85).astype(np.uint8)

    save_path = tmp_path / "test_rollout.png"
    fig = viz.plot_behavior_rollout_strip(preds, gts, save_path=save_path)
    assert save_path.exists()
    assert save_path.stat().st_size > 1000
