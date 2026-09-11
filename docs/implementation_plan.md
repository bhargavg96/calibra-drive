# Implementation Plan: Uncertainty-Aware Evaluation of Driving World Models

**Paper Title**: *"How Well-Calibrated Are Driving World Models? A Benchmark for Predictive Uncertainty in Action-Conditioned Traffic Simulation"*  
**Target Venue**: WACV Workshop on World Action Models for Driving (4–8 page paper)  
**Last Updated**: September 2026

---

## Table of Contents

1. [Overview](#1-overview)
2. [Project Structure](#2-project-structure)
3. [Component Details](#3-component-details)
4. [Experimental Design](#4-experimental-design)
5. [Hypotheses](#5-hypotheses)
6. [Verification Plan](#6-verification-plan)
7. [Timeline & Milestones](#7-timeline--milestones)
8. [Requirements](#8-requirements)
9. [Open Questions](#9-open-questions)

---

## 1. Overview

### 1.1 Motivation

Current driving world models (GAIA-2, Vista, OccWorld, GenAD) generate impressive future predictions of traffic scenes, but **none systematically quantify whether their predictive uncertainty is calibrated**. A model that predicts "80% chance this cell is occupied" should be correct 80% of the time — but this fundamental property has never been verified for driving world models.

This gap is safety-critical: downstream planners that rely on overconfident or underconfident world model predictions will either brake unnecessarily (underconfident) or miss hazards (overconfident).

### 1.2 Contribution

Our paper will be the **first systematic calibration benchmark for driving world models**, providing:

1. A calibration evaluation framework applicable to any driving world model
2. Calibration analysis across multiple open-source models (OccWorld, Vista, GenAD)
3. Novel driving-specific calibration metrics (spatial, temporal, agent-type stratification)
4. Simple recalibration methods that significantly improve safety
5. Downstream evidence: calibrated predictions → safer planning

---

## 2. Project Structure

```
world-models-driving/
│
├── docs/                           # Documentation
│   ├── research_review.md          # Literature review & gap analysis
│   └── implementation_plan.md      # This document
│
├── configs/                        # Configuration files (YAML)
│   ├── datasets/
│   │   ├── nuscenes.yaml           # nuScenes dataset paths & splits
│   │   └── waymo.yaml              # Waymo dataset config (optional)
│   ├── models/
│   │   ├── occworld.yaml           # OccWorld model config
│   │   ├── vista.yaml              # Vista model config
│   │   └── genad.yaml              # GenAD model config
│   └── experiments/
│       ├── calibration.yaml        # Calibration experiment settings
│       ├── recalibration.yaml      # Recalibration experiment settings
│       └── planning.yaml           # Planning experiment settings
│
├── src/                            # Source code
│   ├── data/                       # Data loading & preprocessing
│   │   ├── __init__.py
│   │   ├── nuscenes_loader.py      # nuScenes data pipeline
│   │   ├── waymo_loader.py         # Waymo data pipeline (optional)
│   │   └── scenario_sampler.py     # Difficulty-stratified sampling
│   │
│   ├── models/                     # Model wrappers (unified interface)
│   │   ├── __init__.py
│   │   ├── base_wrapper.py         # Abstract base class
│   │   ├── occworld_wrapper.py     # OccWorld wrapper
│   │   ├── vista_wrapper.py        # Vista wrapper
│   │   ├── genad_wrapper.py        # GenAD wrapper
│   │   └── ensemble_wrapper.py     # Deep ensemble baseline
│   │
│   ├── metrics/                    # Evaluation metrics
│   │   ├── __init__.py
│   │   ├── calibration.py          # Core calibration metrics (ECE, MCE, Brier)
│   │   ├── spatial_calibration.py  # Driving-specific spatial analysis
│   │   └── task_metrics.py         # Standard metrics (IoU, ADE, FDE)
│   │
│   ├── recalibration/              # Post-hoc recalibration methods
│   │   ├── __init__.py
│   │   ├── temperature_scaling.py  # Platt / temperature scaling
│   │   ├── conformal_prediction.py # Distribution-free coverage guarantees
│   │   └── histogram_binning.py    # Non-parametric baseline
│   │
│   ├── planning/                   # Downstream planning experiment
│   │   ├── __init__.py
│   │   ├── uncertainty_planner.py  # MPC planner with uncertainty penalty
│   │   └── evaluation.py          # Closed-loop planning metrics
│   │
│   └── visualization/             # Plotting & qualitative results
│       ├── __init__.py
│       ├── reliability_plots.py    # Reliability diagrams, calibration curves
│       └── qualitative.py          # Occupancy + uncertainty overlays
│
├── scripts/                        # Executable scripts
│   ├── run_inference.py            # Step 1: Generate stochastic rollouts
│   ├── run_calibration.py          # Step 2: Compute calibration metrics
│   ├── run_recalibration.py        # Step 3: Apply recalibration
│   └── run_planning.py            # Step 4: Planning evaluation
│
├── tests/                          # Unit & integration tests
│   ├── test_calibration.py
│   ├── test_synthetic_calibration.py
│   └── test_sanity.py
│
├── paper/                          # LaTeX source for the workshop paper
│   ├── main.tex
│   ├── references.bib
│   └── figures/
│
├── notebooks/                      # Exploration & analysis notebooks
│   └── 01_exploratory_analysis.ipynb
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## 3. Component Details

### 3.1 Data Pipeline

#### `src/data/nuscenes_loader.py`
**Purpose**: Load nuScenes data and extract context–future evaluation pairs.

**Key functionality**:
- Load nuScenes mini (for development) and trainval (for final evaluation) splits
- Extract per-frame sensor data: multi-camera images, LiDAR point clouds
- Extract ego-vehicle action signals from CAN bus: velocity, steering angle, acceleration
- Build evaluation pairs: `(context_frames, ego_action, ground_truth_future)`
- Ground truth formats:
  - **Occupancy grids**: Voxelized LiDAR → binary 3D occupancy (for occupancy models)
  - **Agent trajectories**: Annotated agent bounding box tracks (for trajectory models)

**Dependencies**: `nuscenes-devkit`, `numpy`, `torch`

#### `src/data/scenario_sampler.py`
**Purpose**: Sample evaluation scenarios stratified by difficulty.

**Stratification criteria**:
| Difficulty | Criteria | Target Count |
|-----------|----------|-------------|
| **Easy** | Highway/straight road, ≤2 nearby agents, no lane change | 500 |
| **Medium** | Urban intersection, 3–8 agents, moderate curvature | 500 |
| **Hard** | Dense traffic (8+ agents), lane change/merge, near-miss events | 500 |

**Sampling method**: Use nuScenes scene metadata + heuristics (ego curvature, agent count, minimum TTC) to classify and sample.

---

### 3.2 Model Wrappers

#### `src/models/base_wrapper.py`
**Purpose**: Define a unified interface so all models can be evaluated identically.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

@dataclass
class PredictionBundle:
    """Container for a set of stochastic predictions."""
    occupancy_samples: np.ndarray    # (N, T, H, W, D) or (N, T, H, W)
    trajectory_samples: np.ndarray   # (N, T, num_agents, 2) — x, y
    metadata: dict                   # model-specific extra info

class WorldModelWrapper(ABC):
    """Abstract base class for driving world model wrappers."""

    @abstractmethod
    def predict(self, context, action, n_samples: int = 20) -> PredictionBundle:
        """Generate n_samples stochastic rollouts given context and ego action."""
        pass

    def get_occupancy_probs(self, bundle: PredictionBundle) -> np.ndarray:
        """Aggregate N binary occupancy samples → per-cell probability."""
        return bundle.occupancy_samples.mean(axis=0)  # (T, H, W, D)

    def get_trajectory_distribution(self, bundle: PredictionBundle) -> dict:
        """Compute mean and covariance of trajectory samples."""
        mean = bundle.trajectory_samples.mean(axis=0)
        std = bundle.trajectory_samples.std(axis=0)
        return {"mean": mean, "std": std}
```

#### `src/models/occworld_wrapper.py`
- **Model**: OccWorld (ECCV 2024)
- **Output**: 3D semantic occupancy grids on nuScenes
- **Stochastic generation**: Apply MC Dropout during inference, or sample from the GPT-like autoregressive head with temperature > 0
- **N samples**: Run N forward passes to get diverse occupancy predictions

#### `src/models/vista_wrapper.py`
- **Model**: Vista (NeurIPS 2024)
- **Output**: Future video frames
- **Stochastic generation**: Built-in latent diffusion sampling
- **Post-processing**: Extract pseudo-occupancy via depth estimation (UniDepthV2) from generated frames

#### `src/models/genad_wrapper.py`
- **Model**: GenAD (ECCV 2024)
- **Output**: Future trajectories from VAE latent space
- **Stochastic generation**: Sample from the learned latent distribution (reparameterization trick)
- **Natural uncertainty**: VAE provides principled uncertainty decomposition

#### `src/models/ensemble_wrapper.py`
- **Method**: Deep ensemble of M=5 independently trained checkpoints
- **Purpose**: Gold-standard uncertainty baseline
- **Implementation**: Load M checkpoints, run each, aggregate predictions

---

### 3.3 Calibration Metrics

#### `src/metrics/calibration.py`
Core calibration measurement tools.

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| **ECE** (Expected Calibration Error) | Σ_b (n_b/N) \|acc(b) - conf(b)\| | Weighted avg gap between confidence and accuracy across bins |
| **MCE** (Maximum Calibration Error) | max_b \|acc(b) - conf(b)\| | Worst-case calibration error across bins |
| **Brier Score** | (1/N) Σ (p_i - y_i)² | Joint accuracy + calibration score (lower = better) |
| **AUROC** | Area under ROC curve | How well uncertainty separates correct from incorrect predictions |
| **Sparsification Error** | See implementation | Does removing high-uncertainty predictions improve accuracy? |
| **Reliability Diagram** | Visual: predicted prob vs actual freq | Diagonal = perfect calibration; below = overconfident |

**Binning strategy**: 15 equal-width bins from 0 to 1, following Guo et al. (2017) "On Calibration of Modern Neural Networks."

**Implementation notes**:
- For occupancy models: treat each voxel as an independent binary prediction
- For trajectory models: convert position errors to probability via predicted Gaussian
- Compute calibration separately per prediction timestep

#### `src/metrics/spatial_calibration.py`
Novel driving-specific calibration analysis.

**Distance-stratified calibration**:
- Bin predictions by distance from ego: [0–10m], [10–30m], [30–50m], [50m+]
- Hypothesis: models are better-calibrated nearby, overconfident at distance

**Agent-vs-road calibration**:
- Separate calibration for dynamic objects (vehicles, pedestrians) vs. static elements (road, buildings)
- Hypothesis: dynamic objects have higher uncertainty and potentially worse calibration

**Temporal calibration decay**:
- Measure ECE at each prediction horizon: t = {0.5s, 1s, 2s, 3s, 5s}
- Hypothesis: calibration degrades monotonically with horizon
- Plot: ECE vs. prediction horizon for each model

**Scenario-difficulty calibration**:
- Measure ECE stratified by easy/medium/hard scenarios
- Hypothesis: models are worst-calibrated on hard (safety-critical) scenarios

---

### 3.4 Recalibration Methods

#### `src/recalibration/temperature_scaling.py`
**Method**: Learn a single scalar temperature T on a held-out calibration set.

```
calibrated_logit = raw_logit / T
calibrated_prob = sigmoid(calibrated_logit)
```

**Optimization**: Minimize NLL on calibration set via L-BFGS.

**Advantages**: Single parameter, no overfitting risk, well-understood.

#### `src/recalibration/conformal_prediction.py`
**Method**: Distribution-free prediction sets with coverage guarantees.

**Algorithm**:
1. On calibration set: compute nonconformity scores s_i for each prediction
2. Find quantile q̂ = ⌈(n+1)(1-α)⌉-th smallest score
3. At test time: prediction set C(x) = {y : s(x,y) ≤ q̂}

**For occupancy**: Produce a set of cells guaranteed to contain all truly-occupied cells with probability ≥ 1-α.

**Safety-compelling output**: "With 95% confidence, all occupied cells are within this set" — directly useful for planning.

#### `src/recalibration/histogram_binning.py`
**Method**: Non-parametric, bin-level recalibration.
- For each confidence bin, replace predicted probability with empirical accuracy in that bin.
- Simple baseline, but can be noisy with limited calibration data.

---

### 3.5 Downstream Planning Experiment

#### `src/planning/uncertainty_planner.py`
**Method**: Simple model-predictive control (MPC) with uncertainty awareness.

**Baseline planner** (uncertainty-unaware):
```
best_action = argmin_a  E[cost(world_model(context, a))]
```

**Our planner** (uncertainty-aware):
```
best_action = argmin_a  E[cost(world_model(context, a))] + λ · Var[cost(world_model(context, a))]
```

Where λ is a risk-aversion parameter. Higher λ = more conservative driving.

**Cost function**: Weighted sum of:
- Collision cost (binary: any overlap with predicted occupied cells)
- Progress cost (negative distance traveled along route)
- Comfort cost (jerk + lateral acceleration)
- Off-road cost (binary: ego outside drivable area)

#### `src/planning/evaluation.py`
**Closed-loop evaluation metrics**:

| Metric | Description |
|--------|------------|
| Collision rate (%) | Fraction of scenarios with collision |
| Progress (m) | Average distance traveled along route |
| Comfort score | 1 / (1 + avg_jerk + avg_lat_accel) |
| Off-road rate (%) | Fraction of scenarios with off-road excursion |
| Risk-adjusted score | Progress × (1 - collision_rate) |

---

## 4. Experimental Design

### 4.1 Dataset

**Primary**: nuScenes (1000 scenes, 28k annotated keyframes)
- Training split: used by models for their original training
- Validation split: our evaluation set
- We further split validation: 70% for evaluation, 30% for calibration set (for recalibration methods)

**Secondary** (stretch goal): Waymo Open Dataset for cross-dataset generalization analysis.

### 4.2 Models Under Evaluation

| Model | Output | Uncertainty Source | Priority |
|-------|--------|-------------------|----------|
| OccWorld | 3D occupancy | MC Dropout / autoregressive sampling | **P0** (must-have) |
| Vista | Video frames | Latent diffusion sampling | **P0** (must-have) |
| GenAD | Trajectories | VAE latent sampling | **P1** (should-have) |
| Deep Ensemble | Any | Ensemble disagreement | **P1** (baseline) |

### 4.2.1 Model Availability Status (Verified September 2026)

| Model | Repo | Weights? | Dataset | Stochastic Sampling | License |
|-------|------|----------|---------|---------------------|---------|
| **OccWorld** | [`wzzheng/OccWorld`](https://github.com/wzzheng/OccWorld) | ✅ Yes (Tsinghua Cloud) | nuScenes (Occ3D) | Autoregressive token sampling possible; default is argmax | Academic |
| **Vista** | [`OpenDriveLab/Vista`](https://github.com/OpenDriveLab/Vista) | ✅ Yes (Hugging Face) | OpenDV-20K pretrain; nuScenes, Waymo eval | ✅ Native diffusion sampling | Apache-2.0 |
| **Drive-WM** | [`BraveGroup/Drive-WM`](https://github.com/BraveGroup/Drive-WM) | ❌ "Coming soon" | nuScenes (6-cam) | Diffusion-based (if weights available) | Apache-2.0 |
| **Drive-OccWorld** | [`yuyang-cloud/Drive-OccWorld`](https://github.com/yuyang-cloud/Drive-OccWorld) | ❌ Code only, no weights | nuScenes (Occ3D) | Deterministic; multi-future via action branching | Apache-2.0 |
| **DriveDreamer4D** | [`GigaAI-research/DriveDreamer4D`](https://github.com/GigaAI-research/DriveDreamer4D) | ✅ Yes (Google Drive) | Waymo | ✅ Diffusion-based NTGM | Apache-2.0 |

**Conclusion**: OccWorld + Vista are our **confirmed primary models** (both have public weights and nuScenes support). DriveDreamer4D is a potential third model but uses Waymo only.

### 4.2.2 Critical Prior Work to Address

A recent paper **"C3: World Models That Know When They Don't Know"** (Zhiting Mei et al., Princeton, arXiv:2512.05927) is the closest prior work:
- Proposes calibrated uncertainty for controllable video world models using strictly proper scoring rules
- Computes uncertainty in latent space to avoid pixel-level computational issues
- **Our differentiation**: We provide a **systematic benchmark across multiple models and architectures** (video vs. occupancy vs. trajectory), introduce **driving-specific spatial/temporal calibration analysis**, and demonstrate **downstream planning impact**. C3 proposes a method; we provide a benchmark.

Additionally relevant: **RELIOCC** (post-hoc occupancy calibration) confirms that OccWorld-style models exhibit dramatic overconfidence over long rollout horizons — supporting our H1 hypothesis.

### 4.3 Experiment Matrix

| Experiment | Models | Metrics | Figures |
|-----------|--------|---------|---------|
| **E1: Raw calibration** | All | ECE, MCE, Brier, AUROC | Reliability diagrams (1 per model) |
| **E2: Spatial analysis** | All | Distance-stratified ECE | ECE vs. distance plot |
| **E3: Temporal analysis** | All | Horizon-stratified ECE | ECE vs. time horizon plot |
| **E4: Difficulty analysis** | All | Scenario-stratified ECE | ECE by easy/medium/hard bar chart |
| **E5: Recalibration** | All | ECE before/after, coverage | Before/after reliability diagrams |
| **E6: Planning impact** | Best + worst calibrated | Collision rate, progress | Planning metrics table |

### 4.4 Number of Stochastic Samples

- Default: N = 20 samples per scenario (balance between compute and estimate quality)
- Ablation: N ∈ {5, 10, 20, 50} to show convergence of calibration estimates

---

## 5. Hypotheses

| ID | Hypothesis | Expected Finding |
|----|-----------|------------------|
| **H1** | Current driving world models are systematically overconfident | ECE > 0.10, reliability curves below the diagonal |
| **H2** | Calibration degrades with prediction horizon | ECE at 5s > ECE at 1s for all models |
| **H3** | Calibration is worse for distant predictions | ECE at 50m+ > ECE at 0–10m |
| **H4** | Dynamic objects are less well-calibrated than static scene | ECE(agents) > ECE(road) |
| **H5** | Simple temperature scaling significantly reduces ECE | ECE_recalibrated < 0.5 × ECE_raw |
| **H6** | Conformal prediction achieves stated coverage | Empirical coverage ≥ (1-α) for all tested α |
| **H7** | Calibrated uncertainty improves planning safety | Collision rate with calibrated planner < baseline |
| **H8** | Hard scenarios have worst calibration | ECE(hard) > ECE(medium) > ECE(easy) |

---

## 6. Verification Plan

### 6.1 Automated Tests

```bash
# Unit tests: calibration metrics produce correct results on known inputs
pytest tests/test_calibration.py -v

# Integration test: a perfectly-calibrated synthetic model should have ECE ≈ 0
pytest tests/test_synthetic_calibration.py -v

# Sanity check: uniform random predictions → Brier score ≈ 0.25
pytest tests/test_sanity.py -v

# Full pipeline smoke test on 10 scenarios
python scripts/run_inference.py --config configs/experiments/calibration.yaml --num_scenarios 10
python scripts/run_calibration.py --predictions outputs/smoke_test/
```

### 6.2 Result Validation

- **Reliability diagrams**: Overconfident models should show curves below the diagonal
- **Temperature scaling**: Should move reliability curve closer to diagonal; ECE should decrease
- **Conformal prediction**: Empirical coverage should match or exceed stated coverage (95%, 90%)
- **Planning**: Uncertainty-aware planner should achieve lower collision rate than baseline without significant loss in progress

### 6.3 Sanity Checks

- A random predictor should have ECE ≈ 0.0 (it predicts ~50% and is right ~50%)
- A perfect predictor should have ECE = 0.0 and Brier = 0.0
- Adding more ensemble members should improve calibration (or at least not hurt)
- Temperature T should be > 1.0 if models are overconfident (softens probabilities)

---

## 7. Timeline & Milestones

> **Note**: Timeline depends on submission deadline. Below assumes ~8 weeks of development.

| Week | Milestone | Deliverables |
|------|-----------|-------------|
| **1** | Environment setup + data pipeline | nuScenes loader, scenario sampler, dev environment |
| **2** | Model wrappers (OccWorld) | Working OccWorld inference with stochastic rollouts |
| **3** | Model wrappers (Vista + GenAD) | All model wrappers producing predictions |
| **4** | Calibration metrics + Experiments E1–E4 | Core calibration results, reliability diagrams |
| **5** | Recalibration + Experiment E5 | Temperature scaling, conformal prediction results |
| **6** | Planning experiment E6 | Downstream planning impact numbers |
| **7** | Paper writing | First draft of workshop paper |
| **8** | Polish + submission | Final paper, supplementary material, code release |

---

## 8. Requirements

### 8.1 Software Dependencies

```
# Core
python>=3.9
torch>=2.0
numpy
scipy
scikit-learn

# Data
nuscenes-devkit
pyquaternion

# Visualization
matplotlib
seaborn
plotly

# Models (installed per-model)
# OccWorld: github.com/wzzheng/OccWorld
# Vista: github.com/OpenDriveLab/Vista
# GenAD: github.com/wzzheng/GenAD

# Paper
# LaTeX (texlive or overleaf)
```

### 8.2 Compute Requirements

| Task | Estimated GPU-Hours | Hardware |
|------|-------------------|----------|
| OccWorld inference (1500 scenarios × 20 samples) | ~20 hrs | 1× A100 (40GB) |
| Vista inference (1500 scenarios × 20 samples) | ~30 hrs | 1× A100 (40GB) |
| GenAD inference (1500 scenarios × 20 samples) | ~15 hrs | 1× A100 (40GB) |
| Calibration computation | ~2 hrs | CPU |
| Recalibration optimization | ~1 hr | CPU |
| Planning evaluation | ~5 hrs | 1× GPU |
| **Total** | **~73 hrs** | |

### 8.3 Data Requirements

| Dataset | Size | Required? |
|---------|------|-----------|
| nuScenes mini | ~4 GB | Yes (development) |
| nuScenes trainval | ~300 GB | Yes (evaluation) |
| nuScenes-devkit | pip install | Yes |
| Waymo Open Dataset | ~1 TB | No (stretch goal) |

---

## 9. Open Questions

### For User Decision

1. **Paper length**: 4-page or 8-page workshop paper? (Affects experimental scope — 4-page = E1+E2+E5; 8-page = all experiments)

2. **Dataset access**: Do you already have nuScenes downloaded? If not, the mini split (~4 GB) is sufficient for development; full trainval (~300 GB) needed for final results.

3. **Compute access**: Do you have access to GPUs? Options:
   - Local GPU (24GB+ VRAM recommended)
   - Cloud: Google Colab Pro, AWS/GCP, or university cluster
   - Estimated total: ~73 GPU-hours on A100

4. **Timeline**: When is the workshop submission deadline? This determines whether we pursue all 6 experiments or a focused subset.

5. **Collaboration**: Working solo or with co-authors? This affects task parallelization.

6. **Code release**: Do you plan to open-source the benchmark code alongside the paper? (Recommended — increases citations)

### Technical Decisions (Can Decide Later)

7. **Number of calibration bins**: Default 15 (standard), but could experiment with {10, 15, 20, 25}.

8. **Conformal prediction variant**: Standard split-conformal vs. more advanced (e.g., adaptive conformal). Start with standard.

9. **Planning environment**: Simple replay-based simulation (easier) vs. full closed-loop in CARLA (more realistic but harder). Recommend replay-based for workshop scope.
