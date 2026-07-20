import numpy as np


class FeatureExtractor4D:
    """4D feature extraction from complex CSI vectors.

    Features:
        - Amplitude mean
        - Amplitude variance
        - CFO (Carrier Frequency Offset) mean  (diff of phase)
        - CFO variance
    """

    @staticmethod
    def extract(csi_complex_vector):
        amplitude = np.abs(csi_complex_vector)
        phase = np.angle(csi_complex_vector)
        cfo = np.diff(phase)
        return [np.mean(amplitude), np.var(amplitude), np.mean(cfo), np.var(cfo)]

    @staticmethod
    def extract_batch(snapshots):
        return np.array(
            [FeatureExtractor4D.extract(s) for s in snapshots], dtype=np.float32
        )
