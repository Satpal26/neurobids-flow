"""
NeuroBIDS-Flow — SSVEP Visualizer
===================================
Generates standard SSVEP BCI result plots:
  - Accuracy bar chart (CCA vs FBCCA vs TRCA)
  - ITR comparison (bits/min)
  - Confusion matrix heatmap
  - PSD plot (power spectral density) showing SSVEP peaks
  - CV fold accuracy distribution

All plots save to PNG and optionally display interactively.

Usage:
    from neurobids_flow.ssvep.visualizer import SSVEPVisualizer
    from neurobids_flow.ssvep.evaluator import EvalResult

    viz = SSVEPVisualizer(stim_freqs=[6.0, 8.0, 10.0, 12.0], output_dir="./results")
    viz.plot_accuracy_itr(results)          # results: dict[str, EvalResult]
    viz.plot_confusion_matrix(result, "CCA")
    viz.plot_psd(X, sfreq=256.0)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .evaluator import EvalResult

log = logging.getLogger(__name__)


def _check_matplotlib() -> bool:
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        log.warning("matplotlib not installed — plots will be skipped. Install with: pip install matplotlib")
        return False


class SSVEPVisualizer:
    """
    SSVEP result visualizer.

    Parameters
    ----------
    stim_freqs : list[float]
        Stimulus frequencies in Hz.
    output_dir : str or Path
        Directory to save plots (created if missing).
    show : bool
        If True, display plots interactively (plt.show()).
    dpi : int
        Figure DPI for saved images.
    """

    def __init__(
        self,
        stim_freqs: list[float],
        output_dir: str | Path = "./results/ssvep_plots",
        show: bool = False,
        dpi: int = 150,
    ):
        self.stim_freqs = stim_freqs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.show = show
        self.dpi = dpi

    def _save(self, fig: Any, name: str) -> Path:  # noqa: F821
        path = self.output_dir / name
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        log.info("Saved: %s", path)
        return path

    # ── 1. Accuracy + ITR bar chart ────────────────────────────────────────────
    def plot_accuracy_itr(
        self,
        results: dict[str, "EvalResult"],
        filename: str = "accuracy_itr.png",
    ) -> Path | None:
        """Side-by-side bar chart: accuracy and ITR (bpm) per method."""
        if not _check_matplotlib():
            return None
        import matplotlib.pyplot as plt

        methods = list(results.keys())
        accs = [results[m].accuracy * 100 for m in methods]
        itrs = [results[m].itr_bpm for m in methods]
        n_classes = list(results.values())[0].n_classes
        chance = 100.0 / n_classes

        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"][:len(methods)]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle("SSVEPFlow — Classification Results", fontsize=13, fontweight="bold")

        # Accuracy
        bars = ax1.bar(methods, accs, color=colors, edgecolor="white", linewidth=1.2, zorder=3)
        ax1.axhline(chance, linestyle="--", color="grey", linewidth=1.2, label=f"Chance ({chance:.1f}%)")
        ax1.set_ylim(0, 110)
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_title("Classification Accuracy")
        ax1.legend(fontsize=9)
        ax1.grid(axis="y", alpha=0.3, zorder=0)
        for bar, val in zip(bars, accs):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

        # ITR
        bars2 = ax2.bar(methods, itrs, color=colors, edgecolor="white", linewidth=1.2, zorder=3)
        ax2.set_ylabel("ITR (bits/min)")
        ax2.set_title("Information Transfer Rate")
        ax2.grid(axis="y", alpha=0.3, zorder=0)
        for bar, val in zip(bars2, itrs):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        plt.tight_layout()
        if self.show:
            plt.show()
        out = self._save(fig, filename)
        plt.close(fig)
        return out

    # ── 2. Confusion matrix ────────────────────────────────────────────────────
    def plot_confusion_matrix(
        self,
        result: "EvalResult",
        method_name: str = "",
        filename: str | None = None,
    ) -> Path | None:
        """Heatmap of the confusion matrix for one method."""
        if not _check_matplotlib():
            return None
        import matplotlib.pyplot as plt

        cm = result.confusion_matrix.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, where=row_sums > 0)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label="Proportion")

        freq_labels = [f"{f} Hz" for f in self.stim_freqs]
        ax.set_xticks(range(len(freq_labels)))
        ax.set_yticks(range(len(freq_labels)))
        ax.set_xticklabels(freq_labels, rotation=45, ha="right")
        ax.set_yticklabels(freq_labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix — {method_name}\n(row-normalised)")

        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                val = cm_norm[i, j]
                text = f"{val:.2f}\n({int(cm[i,j])})"
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color)

        plt.tight_layout()
        if self.show:
            plt.show()
        fname = filename or f"confusion_{method_name.lower()}.png"
        out = self._save(fig, fname)
        plt.close(fig)
        return out

    # ── 3. PSD plot ────────────────────────────────────────────────────────────
    def plot_psd(
        self,
        X: np.ndarray,
        sfreq: float = 256.0,
        channel_idx: int = -1,
        filename: str = "psd_ssvep.png",
    ) -> Path | None:
        """
        Plot power spectral density showing SSVEP stimulus peaks.

        Parameters
        ----------
        X : np.ndarray, shape (n_epochs, n_channels, n_times)
        sfreq : float
        channel_idx : int
            Which channel to plot (default -1 = last, usually occipital).
        """
        if not _check_matplotlib():
            return None
        import matplotlib.pyplot as plt
        from scipy.signal import welch

        # Average across epochs, use one channel
        signal = X[:, channel_idx, :].mean(axis=0)
        freqs_psd, psd = welch(signal, fs=sfreq, nperseg=min(256, len(signal)))

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.semilogy(freqs_psd, psd, color="#4C72B0", linewidth=1.2, label="PSD")

        # Mark stimulus frequencies and harmonics
        palette = ["#DD8452", "#55A868", "#C44E52", "#8172B2"]
        for i, sf in enumerate(self.stim_freqs):
            c = palette[i % len(palette)]
            for h in range(1, 4):
                hf = sf * h
                if hf <= freqs_psd[-1]:
                    label = f"{sf} Hz" if h == 1 else None
                    ax.axvline(hf, color=c, linestyle="--", alpha=0.7 / h, linewidth=1.2, label=label)

        ax.set_xlim(0, min(50, freqs_psd[-1]))
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (µV²/Hz)")
        ax.set_title("Power Spectral Density — SSVEP Stimulus Peaks")
        ax.legend(fontsize=9, ncol=2)
        ax.grid(alpha=0.2)
        plt.tight_layout()
        if self.show:
            plt.show()
        out = self._save(fig, filename)
        plt.close(fig)
        return out

    # ── 4. CV fold distribution ────────────────────────────────────────────────
    def plot_cv_folds(
        self,
        results: dict[str, "EvalResult"],
        filename: str = "cv_folds.png",
    ) -> Path | None:
        """Box plot of CV fold accuracy distributions per method."""
        if not _check_matplotlib():
            return None
        import matplotlib.pyplot as plt

        methods = [m for m in results if results[m].cv_fold_accuracies]
        if not methods:
            log.info("No CV fold data — skipping cv_folds plot")
            return None

        data = [results[m].cv_fold_accuracies for m in methods]
        n_classes = list(results.values())[0].n_classes
        chance = 1.0 / n_classes

        fig, ax = plt.subplots(figsize=(7, 4))
        bp = ax.boxplot(data, labels=methods, patch_artist=True, notch=False)
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.axhline(chance, linestyle="--", color="grey", linewidth=1.2, label=f"Chance ({chance:.2f})")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Fold Accuracy")
        ax.set_title("Cross-Validation Fold Accuracy Distribution")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        if self.show:
            plt.show()
        out = self._save(fig, filename)
        plt.close(fig)
        return out

    # ── 5. All plots at once ───────────────────────────────────────────────────
    def plot_all(
        self,
        results: dict[str, "EvalResult"],
        X: np.ndarray | None = None,
        sfreq: float = 256.0,
    ) -> list[Path]:
        """Generate all standard plots. Returns list of saved file paths."""
        saved = []
        p = self.plot_accuracy_itr(results)
        if p:
            saved.append(p)
        for name, result in results.items():
            p = self.plot_confusion_matrix(result, method_name=name)
            if p:
                saved.append(p)
        if X is not None:
            p = self.plot_psd(X, sfreq=sfreq)
            if p:
                saved.append(p)
        p = self.plot_cv_folds(results)
        if p:
            saved.append(p)
        log.info("All plots saved to: %s", self.output_dir)
        return saved