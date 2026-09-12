import numpy as np
from calibra_drive.metrics.anecdote_miner import AnecdoteMiner, DrivingAnecdote

def test_anecdote_miner_discovery():
    rng = np.random.RandomState(42)
    N, T, H, W = 5, 4, 30, 30
    preds = [rng.beta(0.5, 0.5, size=(N, T, H, W)) for _ in range(6)]
    gts = [(rng.rand(T, H, W) > 0.85).astype(np.uint8) for _ in range(6)]

    miner = AnecdoteMiner()
    anecdotes = miner.mine_anecdotes(preds, gts, model_name="OccWorld")

    assert len(anecdotes) == 4
    for a in anecdotes:
        assert isinstance(a, DrivingAnecdote)
        assert a.case_title
        assert a.archetype in ["vru_occlusion", "unprotected_turn", "aggressive_cutin", "temporal_drift"]
        assert a.overconfident_error_rate >= 0.0

def test_anecdote_miner_report():
    rng = np.random.RandomState(42)
    N, T, H, W = 5, 4, 20, 20
    preds = [rng.beta(0.5, 0.5, size=(N, T, H, W)) for _ in range(4)]
    gts = [(rng.rand(T, H, W) > 0.85).astype(np.uint8) for _ in range(4)]

    miner = AnecdoteMiner()
    anecdotes = miner.mine_anecdotes(preds, gts)
    report = miner.export_report(anecdotes)

    assert "CALIBRADRIVE QUALITATIVE CASE STUDIES" in report
    assert "Case 1: Blind-Zone Pedestrian Emergence" in report
