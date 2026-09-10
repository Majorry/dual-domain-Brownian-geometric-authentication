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


# Default trajectory length in CIR snapshots.
TRAJECTORY_WINDOW = 16


def _acquisition_of(snapshot_id):
    """Recover the source .mat filename from a snapshot id ('<fname>_row<i>')."""
    return snapshot_id.rsplit("_row", 1)[0]


def build_trajectories(X, ids, window=TRAJECTORY_WINDOW):
    """Group consecutive snapshots into trajectories.

    load_snapshots emits rows grouped by acquisition file, and in file order, so a
    run of consecutive ids sharing one .mat file is one acquisition. Each
    acquisition is cut into non-overlapping windows of `window` consecutive
    snapshots, so a trajectory holds successive CIR observations from a single
    receiver position -- which is what the Brownian increment model assumes.

    Windows shorter than 2 snapshots are dropped (there are no increments to
    model) and counted, so a coverage gap is visible rather than silent.

    Args:
        X: (n, d) feature matrix, in the order produced by load_snapshots.
        ids: Length-n list of snapshot ids, one per row of X.
        window: Target trajectory length in snapshots.

    Returns:
        (trajectories, n_dropped), where trajectories is a list of (m, d) arrays
        with 2 <= m <= window.
    """
    trajectories = []
    n_dropped = 0
    n = len(ids)
    start = 0
    while start < n:
        fname = _acquisition_of(ids[start])
        end = start
        while end < n and _acquisition_of(ids[end]) == fname:
            end += 1

        block = X[start:end]
        for s in range(0, len(block), window):
            chunk = block[s : s + window]
            if len(chunk) >= 2:
                trajectories.append(chunk)
            else:
                n_dropped += len(chunk)
        start = end

    return trajectories, n_dropped


def load_experiment_data(exp_config, data_dir=None):
    """Load data for one experiment, returning 4D features and trajectories.

    Args:
        exp_config: Dict with 'group_a' and 'group_b' keys.
        data_dir: Override data directory (default: DATA_DIR).

    Returns:
        Dict with keys: X_4d, y_4d, trajs_a, trajs_b, snaps_a, snaps_b, label_a,
                        label_b, n_snapshots_a, n_snapshots_b, n_dropped, or None
                        if data is insufficient.
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

    trajs_a, dropped_a = build_trajectories(X_4d_a, ids_a)
    trajs_b, dropped_b = build_trajectories(X_4d_b, ids_b)
    if len(trajs_a) < 2 or len(trajs_b) < 1:
        return None

    label_a = f"{prefix_a}_{db_a}dB" + (f"({time_a})" if time_a else "")
    label_b = f"{prefix_b}_{db_b}dB" + (f"({time_b})" if time_b else "")

    return {
        "X_4d": X_4d,
        "y_4d": y_4d,
        "trajs_a": trajs_a,
        "trajs_b": trajs_b,
        "n_dropped": dropped_a + dropped_b,
        "snaps_a": snaps_a,
        "snaps_b": snaps_b,
        "ids_a": ids_a,
        "ids_b": ids_b,
        "label_a": label_a,
        "label_b": label_b,
        "n_snapshots_a": len(snaps_a),
        "n_snapshots_b": len(snaps_b),
    }
