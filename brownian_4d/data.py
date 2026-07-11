"""Data loading utilities for ShallowWater_50km CIR dataset.

Expected directory layout:
  channel-impulse-responses-mar-2019-long-range-experiment-mediterranean-sea/
    ShallowWater_50km/
      tx20m_8dB_06_38.mat
      tx20m_8dB_06_44.mat
      ...
      tx35m_12dB_11_38.mat
      ...

Each .mat file contains a variable 'Vch' of shape (n_rows, 2880) complex.
"""

import os
import numpy as np
from scipy.io import loadmat

from .features import FeatureExtractor4D

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "channel-impulse-responses-mar-2019-long-range-experiment-mediterranean-sea",
    "ShallowWater_50km",
)


def load_snapshots(data_dir, prefix, db_val, time_filter=None):
    """Load CIR snapshots matching the given prefix and gain.

    Args:
        data_dir: Path to ShallowWater_50km directory.
        prefix: Filename prefix, e.g. 'tx35m'.
        db_val: Gain value in dB, e.g. 8 or 12.
        time_filter: None (all), str 'HH:MM', or tuple (min_minute, max_minute).

    Returns:
        snapshots: list of (2880,) complex arrays.
        snapshot_ids: list of string identifiers for group-aware splitting.
    """
    snapshots = []
    snapshot_ids = []
    files = sorted(
        f for f in os.listdir(data_dir) if f.startswith(prefix) and f.endswith(".mat")
    )
    for fname in files:
        parts = fname.split("_")
        db = int(parts[1].replace("dB", ""))
        if db != db_val:
            continue
        hh = int(parts[2])
        mm = int(parts[3].replace(".mat", ""))
        time_str = f"{hh:02d}:{mm:02d}"

        if time_filter is not None:
            if isinstance(time_filter, str):
                if time_str != time_filter:
                    continue
            elif isinstance(time_filter, tuple):
                time_min = hh * 60 + mm
                if not (time_filter[0] <= time_min <= time_filter[1]):
                    continue

        fpath = os.path.join(data_dir, fname)
        mat = loadmat(fpath)
        Vch = mat["Vch"]
        for i, row in enumerate(Vch):
            snapshots.append(row)
            snapshot_ids.append(f"{fname}_row{i}")

    return snapshots, snapshot_ids


def load_experiment_data(exp_config, data_dir=None):
    """Load data for one experiment, returning 4D features and raw snapshots.

    Args:
        exp_config: Dict with 'group_a' and 'group_b' keys.
        data_dir: Override data directory (default: DATA_DIR).

    Returns:
        Dict with keys: X_4d, y_4d, snaps_a, snaps_b, label_a, label_b,
                        n_snapshots_a, n_snapshots_b, or None if data is insufficient.
    """
    if data_dir is None:
        data_dir = DATA_DIR

    prefix_a, db_a, time_a = exp_config["group_a"]
    prefix_b, db_b, time_b = exp_config["group_b"]

    snaps_a, ids_a = load_snapshots(data_dir, prefix_a, db_a, time_a)
    snaps_b, ids_b = load_snapshots(data_dir, prefix_b, db_b, time_b)

    if len(snaps_a) == 0 or len(snaps_b) == 0:
        return None

    X_4d_a = FeatureExtractor4D.extract_batch(snaps_a)
    X_4d_b = FeatureExtractor4D.extract_batch(snaps_b)
    X_4d = np.concatenate([X_4d_a, X_4d_b])
    y_4d = np.concatenate([np.ones(len(X_4d_a)), np.zeros(len(X_4d_b))])

    label_a = f"{prefix_a}_{db_a}dB" + (f"({time_a})" if time_a else "")
    label_b = f"{prefix_b}_{db_b}dB" + (f"({time_b})" if time_b else "")

    return {
        "X_4d": X_4d,
        "y_4d": y_4d,
        "snaps_a": snaps_a,
        "snaps_b": snaps_b,
        "ids_a": ids_a,
        "ids_b": ids_b,
        "label_a": label_a,
        "label_b": label_b,
        "n_snapshots_a": len(snaps_a),
        "n_snapshots_b": len(snaps_b),
    }
