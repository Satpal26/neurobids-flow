"""
NeuroBIDS-Flow — SSVEP CCA Classifier
=======================================
Canonical Correlation Analysis (CCA) for SSVEP frequency detection.

CCA finds the linear combination of EEG channels that maximally correlates
with a set of sinusoidal reference signals at each target frequency.
The predicted frequency is the one with the highest canonical correlation.

Reference:
    Lin et al. (2006). Frequency recognition based on canonical correlation
    analysis for SSVEP-based BCIs. IEEE Trans. Biomed. Eng., 53(12), 2610-2614.

Usage:
    from neurobids_flow.ssvep.cca import CCA
    clf = CCA(stim_freqs=[6.0, 8.0, 10.0, 12.0], sfreq=256.0)
    preds = clf.predict(X)   # X: (n_epochs, n_channels, n_times)
"""

from __future__ import annotations

import logging
import numpy as np
from sklearn.cross_decomposition import CCA as _SklearnCCA

log = logging.getLogger(__name__)


class CCA:
    """
    Training-free SSVEP classifier using Canonical Correlation Analysis.

    CCA requires NO training data — it computes correlation between
    EEG epochs and pre-defined sinusoidal reference signals.

    Parameters
    ----------
    stim_freqs : list[float]
        Target SSVEP stimulus frequencies in Hz (e.g. [6.0, 8.0, 10.0, 12.0]).
    sfreq : float
        EEG sampling frequency in Hz.
    n_harmonics : int
        Number of harmonics to include in reference signals (typical: 2-5).
    n_components : int
        Number of CCA components (usually 1).
    tmin : float
        Epoch start time in seconds.
    """

    def __init__(
        self,
        stim_freqs: list[float],
        sfreq: float = 256.0,
        n_harmonics: int = 3,
        n_components: int = 1,
        tmin: float = 0.0,
    ):
        self.stim_freqs = stim_freqs
        self.sfreq = sfreq
        self.n_harmonics = n_harmonics
        self.n_components = n_components
        self.tmin = tmin
        self._references: dict[float, np.ndarray] = {}

    # ── Reference signal builder ───────────────────────────────────────────────
    def _build_reference(self, freq: float, n_times: int) -> np.ndarray:
        """Build sine/cosine reference matrix for one stimulus frequency."""
        t = (np.arange(n_times) / self.sfreq) + self.tmin
        refs = []
        for h in range(1, self.n_harmonics + 1):
            refs.append(np.sin(2 * np.pi * h * freq * t))
            refs.append(np.cos(2 * np.pi * h * freq * t))
        return np.stack(refs, axis=0)  # (2*n_harmonics, n_times)

    def _build_all_references(self, n_times: int) -> None:
        """Pre-build references for all stimulus frequencies."""
        self._references = {
            freq: self._build_reference(freq, n_times)
            for freq in self.stim_freqs
        }

    # ── CCA correlation ────────────────────────────────────────────────────────
    def _cca_corr(self, epoch: np.ndarray, reference: np.ndarray) -> float:
        """Return max canonical correlation between epoch and reference."""
        X = epoch.T       # (n_times, n_channels)
        Y = reference.T   # (n_times, 2*n_harmonics)
        n_comp = min(self.n_components, X.shape[1], Y.shape[1])
        if n_comp < 1:
            return 0.0
        try:
            cca = _SklearnCCA(n_components=n_comp, max_iter=1000)
            cca.fit(X, Y)
            X_c, Y_c = cca.transform(X, Y)
            corr = float(np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1])
            return abs(corr)
        except Exception:
            return 0.0

    # ── Score one epoch ────────────────────────────────────────────────────────
    def _score_epoch(self, epoch: np.ndarray) -> np.ndarray:
        """Return CCA correlation score for each stimulus frequency."""
        return np.array([
            self._cca_corr(epoch, self._references[freq])
            for freq in self.stim_freqs
        ])

    # ── Public API ─────────────────────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict stimulus frequency index for each epoch.

        Parameters
        ----------
        X : np.ndarray, shape (n_epochs, n_channels, n_times)

        Returns
        -------
        labels : np.ndarray, shape (n_epochs,)
            Predicted class index (0-based) into stim_freqs.
        """
        self._build_all_references(X.shape[-1])
        return np.array([np.argmax(self._score_epoch(ep)) for ep in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return CCA correlation scores per frequency.

        Returns
        -------
        scores : np.ndarray, shape (n_epochs, n_freqs)
        """
        self._build_all_references(X.shape[-1])
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
            f"CCA(stim_freqs={self.stim_freqs}, sfreq={self.sfreq}, "
            f"n_harmonics={self.n_harmonics})"
        )


# Keep old name as alias for backward compatibility
CCAClassifier = CCA