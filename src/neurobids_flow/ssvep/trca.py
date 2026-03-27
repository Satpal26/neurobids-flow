"""
NeuroBIDS-Flow — SSVEP Task-Related Component Analysis (TRCA)
=============================================================
TRCA-based SSVEP recognition. Learns spatial filters that maximize
inter-trial covariance — significantly outperforms CCA with enough training data.

Reference:
    Nakanishi et al. (2018) - Enhancing Detection of SSVEPs for a
    High-Speed Brain Speller Using Task-Related Component Analysis.
    IEEE Trans. Biomed. Eng.

Usage:
    from neurobids_flow.ssvep.trca import TRCA
    clf = TRCA(stim_freqs=[6.0, 8.0, 10.0, 12.0], sfreq=256.0)
    clf.fit(X_train, y_train)
    labels = clf.predict(X_test)
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


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
        If True, use ensemble TRCA (eTRCA) — uses all class filters
        together for prediction. More robust with few trials.
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

        self._filters: dict[int, np.ndarray] = {}    # class_idx -> (n_ch, n_comp)
        self._templates: dict[int, np.ndarray] = {}  # class_idx -> (n_comp, n_times)
        self._fitted = False

    # ── TRCA spatial filter computation ───────────────────────────────────────
    @staticmethod
    def _compute_trca_filter(
        X_class: np.ndarray,   # (n_trials, n_channels, n_times)
        n_components: int,
    ) -> np.ndarray:
        """
        Compute TRCA spatial filter W for one class.

        Maximises inter-trial covariance:
            W = argmax W^T S W  subject to W^T Q W = I
        where:
            S = sum_{i!=j} X_i X_j^T  (inter-trial covariance)
            Q = X_all X_all^T          (overall covariance)

        Returns
        -------
        W : np.ndarray, shape (n_channels, n_components)
        """
        n_trials, n_ch, n_times = X_class.shape

        # Inter-trial covariance S
        X_sum = X_class.sum(axis=0)      # (n_ch, n_times)
        S = X_sum @ X_sum.T              # (n_ch, n_ch)
        for trial in X_class:
            S -= trial @ trial.T
        S /= n_trials * (n_trials - 1)

        # Total covariance Q
        X_all = X_class.reshape(n_trials * n_ch, n_times)
        Q = X_all.T @ X_all             # wrong shape — fix:
        X_cat = X_class.reshape(-1, n_times)   # (n_trials*n_ch, n_times)
        # Correct Q: sum of within-trial covariances
        Q = np.zeros((n_ch, n_ch))
        for trial in X_class:
            Q += trial @ trial.T
        Q /= n_trials

        # Regularise Q for numerical stability
        Q += np.eye(n_ch) * 1e-6

        # Generalised eigenvalue problem: S W = lambda Q W
        n_comp = min(n_components, n_ch)
        try:
            eigenvalues, eigenvectors = eigh(S, Q, subset_by_index=[n_ch - n_comp, n_ch - 1])
            # Take top n_comp (largest eigenvalues)
            idx = np.argsort(eigenvalues)[::-1]
            W = eigenvectors[:, idx[:n_comp]]
        except Exception:
            W = np.eye(n_ch)[:, :n_comp]

        return W  # (n_ch, n_comp)

    # ── Fit ───────────────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray) -> "TRCA":
        """
        Fit TRCA spatial filters from training data.

        Parameters
        ----------
        X : np.ndarray, shape (n_epochs, n_channels, n_times)
        y : np.ndarray, shape (n_epochs,) — integer class labels (0-based)
        """
        classes = np.unique(y)
        self._filters = {}
        self._templates = {}

        for cls in classes:
            X_cls = X[y == cls]  # (n_trials, n_ch, n_times)
            if X_cls.shape[0] < 2:
                # Not enough trials — use identity filter
                n_ch = X_cls.shape[1]
                n_comp = min(self.n_components, n_ch)
                W = np.eye(n_ch)[:, :n_comp]
            else:
                W = self._compute_trca_filter(X_cls, self.n_components)

            # Template: mean trial projected through filter
            template = (W.T @ X_cls.mean(axis=0))  # (n_comp, n_times)
            self._filters[int(cls)] = W
            self._templates[int(cls)] = template

        self._fitted = True
        return self

    # ── Score one epoch ────────────────────────────────────────────────────────
    def _score_epoch(self, epoch: np.ndarray) -> np.ndarray:
        """
        Compute correlation score between epoch and each class template.

        Returns
        -------
        scores : np.ndarray, shape (n_classes,)
        """
        classes = sorted(self._filters.keys())
        scores = np.zeros(len(classes))

        for i, cls in enumerate(classes):
            W = self._filters[cls]          # (n_ch, n_comp)
            template = self._templates[cls] # (n_comp, n_times)

            if self.ensemble:
                # eTRCA: use all class filters together
                all_W = np.concatenate(
                    [self._filters[c] for c in classes], axis=1
                )  # (n_ch, total_comp)
                ep_proj = all_W.T @ epoch       # (total_comp, n_times)
                tp_proj = all_W.T @ (W @ template)  # project template same way... simplified:
                # Standard eTRCA correlation
                ep_proj = all_W.T @ epoch
                tp_proj = all_W.T @ (W @ template)
            else:
                ep_proj = W.T @ epoch     # (n_comp, n_times)
                tp_proj = template        # (n_comp, n_times)

            # Correlation between projected epoch and template
            corrs = []
            for comp_ep, comp_tp in zip(ep_proj, tp_proj):
                if comp_ep.std() < 1e-12 or comp_tp.std() < 1e-12:
                    corrs.append(0.0)
                else:
                    r = float(np.corrcoef(comp_ep, comp_tp)[0, 1])
                    corrs.append(abs(r))
            scores[i] = np.mean(corrs)

        return scores

    # ── Public API ─────────────────────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class label for each epoch.

        Parameters
        ----------
        X : np.ndarray, shape (n_epochs, n_channels, n_times)

        Returns
        -------
        labels : np.ndarray, shape (n_epochs,) — predicted class index (0-based)
        """
        if not self._fitted:
            raise RuntimeError("TRCA.fit() must be called before predict().")
        return np.array([np.argmax(self._score_epoch(ep)) for ep in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return TRCA correlation scores per class.

        Returns
        -------
        scores : np.ndarray, shape (n_epochs, n_classes)
        """
        if not self._fitted:
            raise RuntimeError("TRCA.fit() must be called before predict_proba().")
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
        status = "fitted" if self._fitted else "not fitted"
        return (
            f"TRCA(freqs={self.stim_freqs}, sfreq={self.sfreq}, "
            f"n_components={self.n_components}, "
            f"ensemble={self.ensemble}, {status})"
        )