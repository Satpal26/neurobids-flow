"""
NeuroBIDS-Flow — SSVEP Filter Bank CCA (FBCCA)
================================================
Filter Bank Canonical Correlation Analysis for SSVEP.
Improves on baseline CCA by using multiple frequency sub-bands
and weighting them by their sub-band index.

Reference:
    Chen et al. (2015) - Filter bank canonical correlation analysis
    for implementing a high-speed SSVEP-based BCI.

Usage:
    from neurobids_flow.ssvep.fbcca import FBCCA
    clf = FBCCA(stim_freqs=[6.0, 8.0, 10.0, 12.0], sfreq=256.0)
    labels = clf.predict(X)  # X: (n_epochs, n_channels, n_times)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt
from sklearn.cross_decomposition import CCA


# ── Default filter bank sub-bands (Hz) ────────────────────────────────────────
DEFAULT_SUBBANDS = [
    (6,  90),
    (14, 90),
    (22, 90),
    (30, 90),
    (38, 90),
]


class FBCCA:
    """
    Filter Bank Canonical Correlation Analysis (FBCCA) for SSVEP.

    Parameters
    ----------
    stim_freqs : list[float]
        Stimulus frequencies in Hz (e.g. [6.0, 8.0, 10.0, 12.0]).
    sfreq : float
        EEG sampling frequency in Hz.
    n_harmonics : int
        Number of harmonics to include in reference signals.
    subbands : list[tuple[float, float]] or None
        List of (low, high) Hz pairs defining sub-bands.
        Defaults to DEFAULT_SUBBANDS.
    filter_order : int
        Butterworth filter order for each sub-band.
    a : float
        Sub-band weight exponent: weight_k = k^(-a) + b.
    b : float
        Sub-band weight offset.
    n_components : int
        Number of CCA components (usually 1).
    """

    def __init__(
        self,
        stim_freqs: list[float],
        sfreq: float = 256.0,
        n_harmonics: int = 3,
        subbands: list[tuple[float, float]] | None = None,
        filter_order: int = 4,
        a: float = 1.25,
        b: float = 0.25,
        n_components: int = 1,
    ):
        self.stim_freqs = stim_freqs
        self.sfreq = sfreq
        self.n_harmonics = n_harmonics
        self.subbands = subbands if subbands is not None else DEFAULT_SUBBANDS
        self.filter_order = filter_order
        self.a = a
        self.b = b
        self.n_components = n_components

        # Sub-band weights: w_k = k^(-a) + b  (k starts at 1)
        self._weights = np.array(
            [(k + 1) ** (-self.a) + self.b for k in range(len(self.subbands))]
        )

    # ── Reference signal builder ───────────────────────────────────────────────
    def _build_reference(self, freq: float, n_times: int) -> np.ndarray:
        """Build sine/cosine reference matrix for one stimulus frequency."""
        t = np.arange(n_times) / self.sfreq
        refs = []
        for h in range(1, self.n_harmonics + 1):
            refs.append(np.sin(2 * np.pi * h * freq * t))
            refs.append(np.cos(2 * np.pi * h * freq * t))
        return np.stack(refs, axis=0)  # (2*n_harmonics, n_times)

    # ── Butterworth bandpass filter ────────────────────────────────────────────
    def _bandpass(self, X: np.ndarray, low: float, high: float) -> np.ndarray:
        """Apply bandpass filter to X (n_channels, n_times)."""
        nyq = self.sfreq / 2.0
        low_n = max(low / nyq, 1e-4)
        high_n = min(high / nyq, 0.9999)
        if low_n >= high_n:
            return X
        sos = butter(self.filter_order, [low_n, high_n], btype="band", output="sos")
        return sosfilt(sos, X, axis=-1)

    # ── CCA correlation for one epoch, one sub-band, one frequency ────────────
    def _cca_corr(
        self,
        epoch: np.ndarray,          # (n_channels, n_times)
        reference: np.ndarray,      # (2*n_harmonics, n_times)
    ) -> float:
        """Return max canonical correlation between epoch and reference."""
        X = epoch.T          # (n_times, n_channels)
        Y = reference.T      # (n_times, 2*n_harmonics)
        n_comp = min(self.n_components, X.shape[1], Y.shape[1])
        if n_comp < 1:
            return 0.0
        try:
            cca = CCA(n_components=n_comp, max_iter=1000)
            cca.fit(X, Y)
            X_c, Y_c = cca.transform(X, Y)
            corr = float(np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1])
            return abs(corr)
        except Exception:
            return 0.0

    # ── Score one epoch ────────────────────────────────────────────────────────
    def _score_epoch(self, epoch: np.ndarray) -> np.ndarray:
        """
        Compute FBCCA score for each stimulus frequency.

        Returns
        -------
        scores : np.ndarray, shape (n_freqs,)
            Weighted sum of per-subband CCA correlations.
        """
        n_times = epoch.shape[-1]
        scores = np.zeros(len(self.stim_freqs))

        for f_idx, freq in enumerate(self.stim_freqs):
            ref = self._build_reference(freq, n_times)
            weighted_corr = 0.0
            for sb_idx, (low, high) in enumerate(self.subbands):
                filtered = self._bandpass(epoch, low, high)
                r = self._cca_corr(filtered, ref)
                weighted_corr += self._weights[sb_idx] * (r ** 2)
            scores[f_idx] = weighted_corr

        return scores

    # ── Public API ─────────────────────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict stimulus frequency for each epoch.

        Parameters
        ----------
        X : np.ndarray, shape (n_epochs, n_channels, n_times)

        Returns
        -------
        labels : np.ndarray, shape (n_epochs,)
            Predicted frequency index (0-based) into stim_freqs.
        """
        return np.array([np.argmax(self._score_epoch(ep)) for ep in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return FBCCA score matrix (not true probabilities).

        Returns
        -------
        scores : np.ndarray, shape (n_epochs, n_freqs)
        """
        return np.stack([self._score_epoch(ep) for ep in X], axis=0)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Classification accuracy.

        Parameters
        ----------
        X : np.ndarray, shape (n_epochs, n_channels, n_times)
        y : np.ndarray, shape (n_epochs,) — integer class labels (0-based)
        """
        preds = self.predict(X)
        return float(np.mean(preds == y))

    def __repr__(self) -> str:
        return (
            f"FBCCA(freqs={self.stim_freqs}, sfreq={self.sfreq}, "
            f"n_harmonics={self.n_harmonics}, "
            f"n_subbands={len(self.subbands)})"
        )