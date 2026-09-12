import numpy as np
from calibra_drive.recalibration import SpatioTemporalTemperatureScaling
from calibra_drive.metrics import SampleEfficiencyEvaluator, DomainShiftEvaluator

def test_spatiotemporal_scaling_distance_monotonicity():
    rng = np.random.RandomState(42)
    logits = rng.uniform(-2.0, 2.0, 400)
    labels = (rng.rand(400) > 0.5).astype(int)
    dists = np.linspace(2.0, 80.0, 400)
    times = np.linspace(0.5, 3.0, 400)

    scaler = SpatioTemporalTemperatureScaling(mode="joint")
    scaler.fit(logits, labels, distances=dists, timesteps=times)
    scaled_probs = scaler.transform(logits, distances=dists, timesteps=times)

    assert len(scaled_probs) == 400
    assert np.all(scaled_probs >= 0.0) and np.all(scaled_probs <= 1.0)
    assert scaler.joint_params["t0"] > 0

def test_sample_efficiency_evaluator():
    rng = np.random.RandomState(42)
    N, T, H, W = 10, 2, 10, 10
    preds = [rng.beta(0.5, 0.5, size=(N, T, H, W)) for _ in range(4)]
    gts = [(rng.rand(T, H, W) > 0.85).astype(np.uint8) for _ in range(4)]

    evaluator = SampleEfficiencyEvaluator(num_bins=5)
    results = evaluator.evaluate(preds, gts, sample_counts=(1, 5, 10))

    assert "N=1" in results and "N=10" in results
    assert results["N=1"]["relative_latency"] == 1.0
    assert results["N=10"]["relative_latency"] == 10.0

def test_domain_shift_evaluator():
    rng = np.random.RandomState(42)
    N, T, H, W = 5, 2, 10, 10
    preds = [rng.beta(0.5, 0.5, size=(N, T, H, W)) for _ in range(6)]
    gts = [(rng.rand(T, H, W) > 0.85).astype(np.uint8) for _ in range(6)]

    evaluator = DomainShiftEvaluator(num_bins=5)
    results = evaluator.evaluate(preds, gts)

    assert "Day / Clear" in results
    assert "ece" in results["Day / Clear"]
