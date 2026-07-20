"""Brownian motion-based anomaly detectors for underwater acoustic authentication.

Three detectors:
  1. BrownianGeometricDetector  — absolute position: KDE density + convex hull
  2. BrownianIncrementDetector  — increment process: multivariate Gaussian on Δf_t
  3. CompositeBrownianDetector  — combines both
"""

import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.spatial import ConvexHull, Delaunay


class BrownianGeometricDetector:
    """Two-layer anomaly detector: KDE density threshold + convex hull geometric envelope.

    Scores each point in a trajectory by:
      - delta_kde:  1 if KDE log-density < threshold (low-density region)
      - delta_geom: 1 if point is outside the convex hull of nominal data
    Trajectory score S(T) = sum(delta_kde + delta_geom) / T,  range [0, 2].
    """

    def __init__(self, kde_bandwidth=1.0, tau_kde_quantile=0.05, tau1=0.3, tau2=0.7):
        self.kde = KernelDensity(kernel="gaussian", bandwidth=kde_bandwidth)
        self.tau_kde = None
        self.tau_kde_quantile = tau_kde_quantile
        self.tau1 = tau1
        self.tau2 = tau2
        self.hull = None
        self.delaunay = None

    def fit(self, nominal_trajectories):
        """Fit KDE and convex hull on nominal (legitimate) trajectory points."""
        all_points = np.vstack(nominal_trajectories)
        self.kde.fit(all_points)
        log_densities = self.kde.score_samples(all_points)
        self.tau_kde = np.percentile(log_densities, self.tau_kde_quantile * 100)
        self.hull = ConvexHull(all_points)
        self.delaunay = Delaunay(all_points[self.hull.vertices])

    def _in_convex_hull(self, points):
        return self.delaunay.find_simplex(points) >= 0

    def score_trajectory(self, trajectory):
        """Compute anomaly score S(T) = (delta_KDE + delta_Geom) / T."""
        log_densities = self.kde.score_samples(trajectory)
        delta_kde = (log_densities < self.tau_kde).astype(int)
        delta_geom = (~self._in_convex_hull(trajectory)).astype(int)
        return np.sum(delta_kde + delta_geom) / len(trajectory)

    def predict(self, trajectory):
        """Three-level decision: Normal / Uncertain / Anomalous."""
        score = self.score_trajectory(trajectory)
        if score <= self.tau1:
            return "Normal", score
        elif score < self.tau2:
            return "Uncertain", score
        else:
            return "Anomalous", score


class BrownianIncrementDetector:
    """Increment-process detector: fits a multivariate Gaussian on trajectory increments.

    For a trajectory f_1, f_2, ..., f_T, computes increments Δf_t = f_t - f_{t-1}
    and models them as N(μ, Σ). Anomaly score is the negative log-likelihood mean.
    """

    def __init__(self):
        self.mu = None
        self.sigma = None
        self.sigma_inv = None
        self.log_det = None

    def fit(self, nominal_trajectories):
        """Fit Gaussian on increments from nominal trajectories."""
        all_increments = []
        for traj in nominal_trajectories:
            if len(traj) >= 2:
                all_increments.append(traj[1:] - traj[:-1])
        if not all_increments:
            self.mu = None
            return
        all_increments = np.vstack(all_increments)
        self.mu = np.mean(all_increments, axis=0)
        centered = all_increments - self.mu
        self.sigma = np.cov(centered, rowvar=False)
        if self.sigma.ndim == 0:
            self.sigma = np.array([[self.sigma]])
        self.sigma += np.eye(len(self.mu)) * 1e-6  # regularize
        self.sigma_inv = np.linalg.inv(self.sigma)
        _, self.log_det = np.linalg.slogdet(self.sigma)

    def score(self, trajectory):
        """Negative log-likelihood mean of increments (higher = more anomalous)."""
        if self.mu is None or len(trajectory) < 2:
            return 0.0
        increments = trajectory[1:] - trajectory[:-1]
        d = len(self.mu)
        diffs = increments - self.mu
        mahal = np.sum(diffs @ self.sigma_inv * diffs, axis=1)
        nll = 0.5 * (d * np.log(2 * np.pi) + self.log_det + mahal)
        return float(np.mean(nll))


class CompositeBrownianDetector:
    """Composite detector combining absolute-position and increment-process scoring.

    Final score = mean(normalized_abs_score, normalized_inc_score).
    """

    def __init__(self, kde_bandwidth=1.0, tau_kde_quantile=0.05, tau1=0.3, tau2=0.7):
        self.abs_detector = BrownianGeometricDetector(
            kde_bandwidth, tau_kde_quantile, tau1, tau2
        )
        self.inc_detector = BrownianIncrementDetector()
        self.tau1 = tau1
        self.tau2 = tau2
        self._inc_max = 1.0

    def fit(self, nominal_trajectories):
        """Fit both detectors on nominal trajectories."""
        self.abs_detector.fit(nominal_trajectories)
        self.inc_detector.fit(nominal_trajectories)

    def fit_score_calibration(self, nominal_trajectories):
        """Calibrate increment score normalization using nominal data."""
        scores = [
            self.inc_detector.score(t)
            for t in nominal_trajectories
            if len(t) >= 2
        ]
        self._inc_max = np.percentile(scores, 99) if scores else 1.0

    def score_trajectory(self, trajectory):
        """Combined score: mean of normalized absolute + increment scores."""
        s_abs = self.abs_detector.score_trajectory(trajectory) / 2.0
        s_inc = self.inc_detector.score(trajectory)
        s_inc_norm = min(s_inc / self._inc_max, 1.0) if self._inc_max > 0 else 0.0
        return (s_abs + s_inc_norm) / 2.0

    def predict(self, trajectory):
        """Three-level decision: Normal / Uncertain / Anomalous."""
        score = self.score_trajectory(trajectory)
        if score <= self.tau1:
            return "Normal", score
        elif score < self.tau2:
            return "Uncertain", score
        else:
            return "Anomalous", score
