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
    from neurobids_flow.ssvep.cca import CCAClassifier
    clf = CCAClassifier(freqs=[6, 8, 10, 12], sfreq=256, n_harmonics=2)
    clf.fit()   # builds reference signals — no training data needed
    pred = clf.predict(X)   # X: (n_epochs, n_channels, n_times)
"""

from __future__ import annotations

import logging
import numpy as np
from sklearn.cross_decomposition import CCA

log = logging.getLogger(__name__)


class CCAClassifier:
    """
    Training-free SSVEP classifier using Canonical Correlation Analysis.

    CCA requires NO training data — it computes correlation between
    EEG epochs and pre-defined sinusoidal reference signals.
    This makes it ideal for zero-shot SSVEP detection.

    Parameters
    ----------
    freqs : list of float
        Target SSVEP stimulus frequencies in Hz.
        e.g. [6, 8, 10, 12]
    sfreq : float
        EEG sampling frequency in Hz.
    n_harmonics : int
        Number of harmonics to include in reference signals.
        Higher = more discriminative but more sensitive to noise.
        Typical: 2-5.
    tmin : float
        Epoch start time in seconds (used to build reference time vector).
    """

    def __init__(
        self,
        freqs: list[float],
        sfreq: float,
        n_harmonics: int = 2,
        tmin: float = 0.0,
    ):
        self.freqs       = freqs
        self.sfreq       = sfreq
        self.n_harmonics = n_harmonics
        self.tmin        = tmin
        self.references_ = None   # built in fit()

        log.info(
            f"CCAClassifier | freqs={freqs} Hz | "
            f"sfreq={sfreq} | harmonics={n_harmonics}"
        )

    def fit(self, n_times: int = None, X: np.ndarray = None):
        """
        Build sinusoidal reference signals for each target frequency.
        No EEG training data needed — CCA is training-free.

        Parameters
        ----------
        n_times : int
            Number of time samples per epoch.
            Inferred from X if not provided.
        X : np.ndarray, shape (n_epochs, n_channels, n_times)
            Optional — used only to infer n_times.
        """
        if n_times is None:
            if X is not None:
                n_times = X.shape[2]
            else:
                raise ValueError("Provide either n_times or X.")

        self.n_times_ = n_times
        self.references_ = {}

        t = (np.arange(n_times) / self.sfreq) + self.tmin

        for freq in self.freqs:
            ref = []
            for h in range(1, self.n_harmonics + 1):
                ref.append(np.sin(2 * np.pi * h * freq * t))
                ref.append(np.cos(2 * np.pi * h * freq * t))
            # Shape: (2 * n_harmonics, n_times)
            self.references_[freq] = np.array(ref)

        log.info(
            f"  Built {len(self.freqs)} reference signals | "
            f"n_times={n_times} | harmonics={self.n_harmonics}"
        )
        return self

    def _compute_correlation(
        self,
        epoch: np.ndarray,
        reference: np.ndarray,
    ) -> float:
        """
        Compute canonical correlation between one epoch and one reference.

        Parameters
        ----------
        epoch : np.ndarray, shape (n_channels, n_times)
        reference : np.ndarray, shape (2*n_harmonics, n_times)

        Returns
        -------
        float : first canonical correlation coefficient
        """
        X = epoch.T       # (n_times, n_channels)
        Y = reference.T   # (n_times, 2*n_harmonics)

        n_components = min(X.shape[1], Y.shape[1], 1)
        cca = CCA(n_components=n_components, max_iter=1000)

        try:
            cca.fit(X, Y)
            X_c, Y_c = cca.transform(X, Y)
            # First canonical correlation
            corr = float(np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1])
            return abs(corr)
        except Exception:
            return 0.0

    def predict_single(self, epoch: np.ndarray) -> tuple[float, dict]:
        """
        Predict SSVEP frequency for a single epoch.

        Parameters
        ----------
        epoch : np.ndarray, shape (n_channels, n_times)

        Returns
        -------
        predicted_freq : float
        correlations   : dict mapping freq → correlation value
        """
        if self.references_ is None:
            raise RuntimeError("Call fit() before predict().")

        correlations = {}
        for freq in self.freqs:
            corr = self._compute_correlation(epoch, self.references_[freq])
            correlations[freq] = corr

        predicted_freq = max(correlations, key=correlations.get)
        return predicted_freq, correlations

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict SSVEP frequency for each epoch.

        Parameters
        ----------
        X : np.ndarray, shape (n_epochs, n_channels, n_times)

        Returns
        -------
        predictions : np.ndarray, shape (n_epochs,)
            Predicted frequency (Hz) for each epoch.
        """
        if self.references_ is None:
            self.fit(n_times=X.shape[2])

        predictions = []
        for epoch in X:
            pred_freq, _ = self.predict_single(epoch)
            predictions.append(pred_freq)

        return np.array(predictions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return correlation scores for each frequency (soft predictions).

        Parameters
        ----------
        X : np.ndarray, shape (n_epochs, n_channels, n_times)

        Returns
        -------
        scores : np.ndarray, shape (n_epochs, n_freqs)
            CCA correlation score per epoch per frequency.
        """
        if self.references_ is None:
            self.fit(n_times=X.shape[2])

        all_scores = []
        for epoch in X:
            _, corrs = self.predict_single(epoch)
            scores = [corrs[f] for f in self.freqs]
            all_scores.append(scores)

        return np.array(all_scores)

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        freq_to_label: dict = None,
    ) -> float:
        """
        Compute classification accuracy.

        Parameters
        ----------
        X : np.ndarray, shape (n_epochs, n_channels, n_times)
        y : np.ndarray, shape (n_epochs,)
            True labels — either frequencies (Hz) or integer indices.
        freq_to_label : dict, optional
            Maps frequency → integer label.
            If None, assumes y contains frequencies directly.

        Returns
        -------
        accuracy : float
        """
        predictions = self.predict(X)

        if freq_to_label is not None:
            label_to_freq = {v: k for k, v in freq_to_label.items()}
            y_freq = np.array([label_to_freq.get(label, -1) for label in y])
        else:
            y_freq = y.astype(float)

        accuracy = float(np.mean(predictions == y_freq))
        log.info(f"CCA accuracy: {accuracy:.3f} ({int(accuracy * len(y))}/{len(y)})")
        return accuracy