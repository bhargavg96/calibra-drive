import numpy as np
from calibra_drive.metrics import CalibrationMetrics

def test_perfect_calibration():
    metrics = CalibrationMetrics(num_bins=10)
    preds = np.linspace(0.1, 0.9, 9)
    gts = (preds > 0.5).astype(float)
    # This won't be perfectly 0 in finite bins without careful setup, 
    # but let's test a simple exact match
    exact_preds = np.array([0.0, 1.0])
    exact_gts = np.array([0.0, 1.0])
    assert metrics.ece(exact_preds, exact_gts) < 1e-5

def test_overconfident():
    metrics = CalibrationMetrics()
    preds = np.array([0.9]*10)
    gts = np.array([1]*5 + [0]*5)
    assert metrics.ece(preds, gts) > 0.3

def test_brier_score_perfect():
    metrics = CalibrationMetrics()
    preds = np.array([0.0, 1.0])
    gts = np.array([0.0, 1.0])
    assert metrics.brier_score(preds, gts) == 0.0

def test_brier_score_random():
    metrics = CalibrationMetrics()
    preds = np.array([0.5, 0.5])
    gts = np.array([1.0, 0.0])
    assert metrics.brier_score(preds, gts) == 0.25

def test_reliability_diagram_bins():
    metrics = CalibrationMetrics(num_bins=5)
    preds = np.random.rand(100)
    gts = np.random.randint(0, 2, 100)
    res = metrics.reliability_diagram_data(preds, gts)
    assert len(res['bin_centers']) == 5

def test_ece_symmetry():
    metrics = CalibrationMetrics()
    preds = np.array([0.1, 0.9])
    gts = np.array([0, 1])
    ece1 = metrics.ece(preds, gts)
    ece2 = metrics.ece(1 - preds, 1 - gts)
    assert abs(ece1 - ece2) < 1e-5
