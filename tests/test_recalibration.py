import numpy as np
from calibra_drive.recalibration import TemperatureScaling, ConformalPredictor, HistogramBinning
from calibra_drive.metrics import CalibrationMetrics

def test_temperature_scaling_identity():
    ts = TemperatureScaling()
    logits = np.array([0.0, 1.0, -1.0])
    preds = ts.transform(logits)
    expected = 1 / (1 + np.exp(-logits))
    np.testing.assert_allclose(preds, expected)

def test_temperature_scaling_softens():
    ts = TemperatureScaling()
    ts._temperature = 2.0
    logits = np.array([2.0, -2.0])
    preds = ts.transform(logits)
    expected = 1 / (1 + np.exp(-1.0))
    assert preds[0] < 1 / (1 + np.exp(-2.0))

def test_conformal_coverage():
    cp = ConformalPredictor(alpha=0.1)
    preds = np.linspace(0, 1, 100)
    gts = preds + np.random.normal(0, 0.05, 100)
    cp.fit(preds, gts)
    cov = cp.evaluate_coverage(preds, gts)
    assert cov >= 0.85 # rough check

def test_histogram_binning_calibrates():
    hb = HistogramBinning(num_bins=5)
    preds = np.array([0.1, 0.1, 0.9, 0.9])
    gts = np.array([0, 1, 1, 1])
    hb.fit(preds, gts)
    transformed = hb.transform(preds)
    # For bin 0 (preds around 0.1), acc is 0.5
    assert abs(transformed[0] - 0.5) < 1e-5
