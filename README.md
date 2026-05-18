# Brownian-4D Authentication

Underwater acoustic physical layer authentication using a composite Brownian motion detector with 4D channel features.

## Overview

This method authenticates underwater acoustic transmitters by modeling the statistical properties of their Channel Impulse Response (CIR). It combines two anomaly detection mechanisms:

1. **Absolute-position detector** — fits a KDE density model + convex hull envelope on legitimate feature points. Points with low density or outside the hull are flagged.

2. **Increment-process detector** — models the Brownian motion increments (Δf_t = f_t - f_{t-1}) as a multivariate Gaussian. Trajectories with unlikely increment patterns are flagged.

The final score is the average of both normalized scores.

### 4D Features

Each CIR snapshot (2880 complex samples) is summarized into 4 features:

| # | Feature | Description |
|---|---------|-------------|
| 1 | Amplitude mean | Mean of |h(t)| |
| 2 | Amplitude variance | Var of |h(t)| |
| 3 | CFO mean | Mean of Δφ(t) (carrier frequency offset) |
| 4 | CFO variance | Var of Δφ(t) |

## Project Structure

```
brownian-4d-auth/
├── brownian_4d/
│   ├── __init__.py
│   ├── detector.py      # BrownianGeometricDetector, BrownianIncrementDetector, CompositeBrownianDetector
│   ├── features.py      # FeatureExtractor4D
│   ├── data.py          # CIR data loading
│   └── experiments.py   # Experiment definitions (A/B/C groups)
├── run.py               # Main entry point
├── requirements.txt
└── README.md
```

## Dataset

Uses the [ShallowWater_50km](https://doi.org/10.5281/zenodo.3694682) CIR dataset from a March 2019 Mediterranean Sea experiment.

Place the dataset directory as a sibling of this project folder:
```
parent/
├── brownian-4d-auth/
└── channel-impulse-responses-mar-2019-long-range-experiment-mediterranean-sea/
    └── ShallowWater_50km/
        ├── tx20m_8dB_06_38.mat
        ├── tx35m_12dB_11_38.mat
        └── ...
```

Each `.mat` file contains a variable `Vch` of shape `(n_rows, 2880)` complex.


## Usage

```bash
pip install -r requirements.txt
python run.py
```

### Output

For each experiment, the script reports:
- **Hard decision (k=2)**: threshold = mean + 2σ of legitimate scores
- **Hard decision (3-class)**: Normal / Uncertain / Anomalous
- **Soft decision**: ROC-AUC, optimal F1 threshold, confidence statistics

## Experiment Design

| Group | Description | Experiments |
|-------|-------------|-------------|
| A | Depth discrimination (same gain, different depth) | A1–A4 |
| B | Cross-time robustness (12dB, different timestamps) | B1–B5 |
| C | Cross-time robustness (8dB, different timestamps) | C1–C4 |

## License

MIT
