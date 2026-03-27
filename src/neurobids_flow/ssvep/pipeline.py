"""
NeuroBIDS-Flow — SSVEPFlow End-to-End Pipeline
===============================================
Orchestrates the full SSVEP pipeline:
  1. Load BIDS-EEG output (from Target 1 NeuroBIDS-Flow)
  2. Preprocess (filter → epoch → baseline)
  3. Run CCA / FBCCA / TRCA classifiers
  4. Evaluate (accuracy, ITR, confusion matrix, CV)
  5. Print summary report

Usage:
    python src/neurobids_flow/ssvep/pipeline.py \\
        --bids-root ./bids_output \\
        --config configs/ssvep_config.yaml

    Or from Python:
        from neurobids_flow.ssvep.pipeline import SSVEPPipeline
        pipe = SSVEPPipeline.from_config("configs/ssvep_config.yaml")
        results = pipe.run()
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import SSVEPConfig, load_ssvep_config
from .preprocessor import SSVEPPreprocessor
from .cca import CCA as SSVEP_CCA
from .fbcca import FBCCA
from .trca import TRCA
from .evaluator import SSVEPEvaluator, EvalResult

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-7s | %(message)s",
)

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║         NeuroBIDS-Flow — SSVEPFlow Pipeline                     ║
║         CCA  ·  FBCCA  ·  TRCA  |  Passive BCI                 ║
╚══════════════════════════════════════════════════════════════════╝"""


class SSVEPPipeline:
    """
    End-to-end SSVEP BCI pipeline.

    Parameters
    ----------
    config : SSVEPConfig
    """

    def __init__(self, config: SSVEPConfig):
        self.cfg = config

    @classmethod
    def from_config(cls, path: str) -> "SSVEPPipeline":
        return cls(load_ssvep_config(path))

    @classmethod
    def from_defaults(
        self,
        bids_root: str = "./bids_output",
        stim_freqs: list[float] | None = None,
        task: str = "ssvep",
        methods: list[str] | None = None,
    ) -> "SSVEPPipeline":
        cfg = SSVEPConfig()
        cfg.bids_root = bids_root
        cfg.task = task
        if stim_freqs:
            cfg.stim_freqs = stim_freqs
        if methods:
            cfg.methods = methods
        return SSVEPPipeline(cfg)

    # ── Step 1 — Load & preprocess ─────────────────────────────────────────────
    def _load_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Load BIDS data and return preprocessed (X, y)."""
        log.info("Step 1 — Loading BIDS-EEG data from: %s", self.cfg.bids_root)
        prep = SSVEPPreprocessor(
            bids_root=self.cfg.bids_root,
            task=self.cfg.task,
            stim_freqs=self.cfg.stim_freqs,
            tmin=self.cfg.epochs.tmin,
            tmax=self.cfg.epochs.tmax,
            l_freq=self.cfg.filters.lowcut,
            h_freq=self.cfg.filters.highcut,
            notch_freq=self.cfg.filters.notch,
            baseline=self.cfg.epochs.baseline,
        )
        X, y = prep.load_and_preprocess()
        log.info(
            "Step 1 done — X=%s, y=%s, classes=%s",
            X.shape, y.shape, np.unique(y).tolist(),
        )
        return X, y

    # ── Step 2 — Build classifiers ─────────────────────────────────────────────
    def _build_classifiers(self, sfreq: float) -> dict[str, Any]:
        clfs: dict[str, Any] = {}
        cfg = self.cfg

        if "cca" in cfg.methods:
            clfs["CCA"] = SSVEP_CCA(
                stim_freqs=cfg.stim_freqs,
                sfreq=sfreq,
                n_harmonics=cfg.cca.n_harmonics,
                n_components=cfg.cca.n_components,
            )

        if "fbcca" in cfg.methods:
            subbands = [tuple(sb) for sb in cfg.fbcca.subbands]
            clfs["FBCCA"] = FBCCA(
                stim_freqs=cfg.stim_freqs,
                sfreq=sfreq,
                n_harmonics=cfg.fbcca.n_harmonics,
                n_components=cfg.fbcca.n_components,
                subbands=subbands,
                filter_order=cfg.fbcca.filter_order,
                a=cfg.fbcca.a,
                b=cfg.fbcca.b,
            )

        if "trca" in cfg.methods:
            clfs["TRCA"] = TRCA(
                stim_freqs=cfg.stim_freqs,
                sfreq=sfreq,
                n_components=cfg.trca.n_components,
                ensemble=cfg.trca.ensemble,
            )

        return clfs

    # ── Step 3 — Run evaluation ────────────────────────────────────────────────
    def _evaluate_all(
        self,
        clfs: dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, EvalResult]:
        ev = SSVEPEvaluator(
            n_classes=len(self.cfg.stim_freqs),
            epoch_duration=self.cfg.eval.epoch_duration,
            gap_duration=self.cfg.eval.gap_duration,
            n_splits=self.cfg.eval.n_splits,
        )
        results: dict[str, EvalResult] = {}
        for name, clf in clfs.items():
            log.info("Step 3 — Evaluating %s ...", name)
            try:
                result = ev.evaluate(clf, X, y, method_name=name, run_cv=True)
                results[name] = result
            except Exception as e:
                log.warning("  %s failed: %s", name, e)
        return results

    # ── Summary table ──────────────────────────────────────────────────────────
    @staticmethod
    def _print_summary(
        results: dict[str, EvalResult],
        stim_freqs: list[float],
        total_time: float,
    ) -> None:
        print("\n" + "═" * 72)
        print("  SSVEP PIPELINE — SUMMARY")
        print("═" * 72)
        print(
            f"  {'Method':<10}  {'Accuracy':>10}  {'ITR(bpt)':>10}  "
            f"{'ITR(bpm)':>10}  {'CV mean±std':>16}"
        )
        print("─" * 72)
        for name, r in results.items():
            cv_str = f"{r.cv_mean:.3f} ± {r.cv_std:.3f}" if r.cv_fold_accuracies else "—"
            above = "✓" if r.accuracy > 1.0 / r.n_classes else "✗"
            print(
                f"  {name:<10}  {r.accuracy:>9.4f}{above}  {r.itr_bpt:>10.4f}  "
                f"{r.itr_bpm:>10.2f}  {cv_str:>16}"
            )
        print("─" * 72)
        print(f"  Chance level  : {1.0/list(results.values())[0].n_classes:.4f}")
        print(f"  Total runtime : {total_time:.1f}s")
        print("═" * 72)

    # ── Main entry point ───────────────────────────────────────────────────────
    def run(self) -> dict[str, EvalResult]:
        """
        Run the full SSVEPFlow pipeline.

        Returns
        -------
        results : dict[str, EvalResult]
            One EvalResult per method (CCA, FBCCA, TRCA).
        """
        print(BANNER)
        t0 = time.time()

        # Load data
        X, y = self._load_data()
        if X is None or len(X) == 0:
            log.error("No data loaded — check BIDS root and task name.")
            return {}

        # Infer sfreq from config or fallback
        sfreq = self.cfg.sfreq or 256.0
        log.info("Step 2 — Building classifiers: %s", self.cfg.methods)
        clfs = self._build_classifiers(sfreq)

        # Evaluate
        results = self._evaluate_all(clfs, X, y)

        # Reports
        ev = SSVEPEvaluator(
            n_classes=len(self.cfg.stim_freqs),
            epoch_duration=self.cfg.eval.epoch_duration,
            gap_duration=self.cfg.eval.gap_duration,
        )
        for name, result in results.items():
            ev.print_report(result, stim_freqs=self.cfg.stim_freqs)

        self._print_summary(results, self.cfg.stim_freqs, time.time() - t0)
        return results


# ── CLI ────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SSVEPFlow — SSVEP BCI Pipeline")
    p.add_argument("--bids-root", default="./bids_output", help="BIDS root directory")
    p.add_argument("--config", default=None, help="Path to ssvep_config.yaml")
    p.add_argument("--task", default="ssvep", help="BIDS task name")
    p.add_argument(
        "--freqs", nargs="+", type=float, default=[6.0, 8.0, 10.0, 12.0],
        help="Stimulus frequencies in Hz",
    )
    p.add_argument(
        "--methods", nargs="+", default=["cca", "fbcca", "trca"],
        choices=["cca", "fbcca", "trca"],
        help="Methods to run",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.config:
        pipe = SSVEPPipeline.from_config(args.config)
    else:
        cfg = SSVEPConfig()
        cfg.bids_root = args.bids_root
        cfg.task = args.task
        cfg.stim_freqs = args.freqs
        cfg.methods = args.methods
        pipe = SSVEPPipeline(cfg)

    pipe.run()