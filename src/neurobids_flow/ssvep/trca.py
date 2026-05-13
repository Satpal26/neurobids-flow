"""
NeuroBIDS-Flow — SSVEP Task-Related Component Analysis (TRCA)
=============================================================
TRCA/eTRCA implementation — exact port of Nakanishi et al. (2018) MATLAB code.

Reference:
    Nakanishi et al. (2018). Enhancing Detection of SSVEPs for a
    High-Speed Brain Speller Using Task-Related Component Analysis.
    IEEE Trans. Biomed. Eng., 65(1), 104-112.
    Original MATLAB: https://github.com/mnakanishi/12JFPM_SSVEP
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eig


class TRCA:
    """
    Task-Related Component Analysis (TRCA) for SSVEP.

    Parameters
    ----------
    stim_freqs : list[float]
        Stimulus frequencies in Hz.
    sfreq : float
        Sampling frequency in Hz.
    n_components : int
        Number of spatial filters per class (usually 1).
    ensemble : bool
        If True, use ensemble TRCA (eTRCA).
    """

    def __init__(
        self,
        stim_freqs: list[float],
        sfreq: float = 256.0,
        n_components: int = 1,
        ensemble: bool = True,
    ):
        self.stim_freqs = stim_freqs
        self.sfreq = sfreq
        self.n_components = n_components
        self.ensemble = ensemble

        self._filters: dict[int, np.ndarray] = {}
        self._templates: dict[int, np.ndarray] = {}
        self._fitted = False

    @staticmethod
    def _trca_filter(X_class: np.ndarray) -> np.ndarray:
        """
        Exact port of Nakanishi 2018 MATLAB trca() function.

        X_class : (n_trials, n_ch, n_times)
        Returns  : W (n_ch, 1)
        """
        n_trials, n_ch, n_times = X_class.shape

        # ── Inter-trial covariance S ──────────────────────────────────
        # S = sum_{i<j} (X_i @ X_j.T + X_j @ X_i.T)
        S = np.zeros((n_ch, n_ch))
        for i in range(n_trials - 1):
            xi = X_class[i]           # (n_ch, n_times)
            for j in range(i + 1, n_trials):
                xj = X_class[j]       # (n_ch, n_times)
                S += xi @ xj.T + xj @ xi.T

        # ── Total covariance Q ────────────────────────────────────────
        # Q = UX @ UX.T  where UX = [X_1 | X_2 | ... | X_n] (horizontal cat)
        UX = X_class.reshape(n_trials * n_ch, n_times)
        # Reshape to (n_ch, n_trials * n_times) — correct concat
        UX = np.concatenate([X_class[t] for t in range(n_trials)], axis=1)
        Q = UX @ UX.T   # (n_ch, n_ch)

        # ── Generalized eigenvalue: S w = lambda Q w ──────────────────
        try:
            eigenvalues, eigenvectors = eig(S, Q)
            eigenvalues = eigenvalues.real
            eigenvectors = eigenvectors.real
            # Take eigenvector with largest real eigenvalue
            idx = np.argsort(eigenvalues)[::-1]
            W = eigenvectors[:, idx[0:1]]   # (n_ch, 1)
        except Exception:
            W = np.ones((n_ch, 1)) / np.sqrt(n_ch)

        return W  # (n_ch, 1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TRCA":
        """
        Fit TRCA spatial filters.

        X : (n_epochs, n_channels, n_times)
        y : (n_epochs,) integer class labels 0-based
        """
        classes = np.unique(y)
        self._filters = {}
        self._templates = {}

        for cls in classes:
            X_cls = X[y == cls]        # (n_trials, n_ch, n_times)
            n_trials = X_cls.shape[0]

            if n_trials < 2:
                n_ch = X_cls.shape[1]
                W = np.ones((n_ch, 1)) / np.sqrt(n_ch)
            else:
                W = self._trca_filter(X_cls)

            # Template = mean across trials in channel space
            template = X_cls.mean(axis=0)  # (n_ch, n_times)

            self._filters[int(cls)] = W          # (n_ch, 1)
            self._templates[int(cls)] = template  # (n_ch, n_times)

        self._fitted = True
        return self

    def _score_epoch(self, epoch: np.ndarray) -> np.ndarray:
        """
        Score epoch against all class templates.

        For eTRCA: concatenate all W, project epoch and each template,
        compute sum of squared correlations across all components.

        Returns scores : (n_classes,)
        """
        classes = sorted(self._filters.keys())
        n_classes = len(classes)
        scores = np.zeros(n_classes)

        if self.ensemble:
            # eTRCA: (n_ch, n_classes) — one filter per class
            all_W = np.concatenate(
                [self._filters[c] for c in classes], axis=1
            )  # (n_ch, n_classes)

            # Project epoch: (n_classes, n_times)
            ep_proj = all_W.T @ epoch

            for i, cls in enumerate(classes):
                tp_proj = all_W.T @ self._templates[cls]  # (n_classes, n_times)

                # Sum of squared correlations across all components
                r_sum = 0.0
                for k in range(ep_proj.shape[0]):
                    ep_k = ep_proj[k]
                    tp_k = tp_proj[k]
                    std_e = ep_k.std()
                    std_t = tp_k.std()
                    if std_e > 1e-12 and std_t > 1e-12:
                        r = float(np.corrcoef(ep_k, tp_k)[0, 1])
                        r_sum += r ** 2
                scores[i] = r_sum

        else:
            for i, cls in enumerate(classes):
                W = self._filters[cls]               # (n_ch, 1)
                ep_proj = W.T @ epoch                # (1, n_times)
                tp_proj = W.T @ self._templates[cls] # (1, n_times)

                ep_k = ep_proj[0]
                tp_k = tp_proj[0]
                if ep_k.std() > 1e-12 and tp_k.std() > 1e-12:
                    r = float(np.corrcoef(ep_k, tp_k)[0, 1])
                    scores[i] = r ** 2
                else:
                    scores[i] = 0.0

        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class for each epoch. X: (n_epochs, n_ch, n_times)"""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        return np.array([np.argmax(self._score_epoch(ep)) for ep in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return scores. Returns (n_epochs, n_classes)"""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_proba().")
        return np.stack([self._score_epoch(ep) for ep in X], axis=0)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Classification accuracy."""
        return float(np.mean(self.predict(X) == y))

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return (
            f"TRCA(freqs={self.stim_freqs}, sfreq={self.sfreq}, "
            f"ensemble={self.ensemble}, {status})"
        )