"""
NeuroBIDS-Flow — SSVEP Benchmark
==================================
Cross-dataset / cross-device benchmarking for the SSVEP pipeline.
Runs CCA, FBCCA, TRCA per subject and produces a comparison table
identical to what goes in the paper results section.

Usage:
    python src/neurobids_flow/ssvep/benchmark.py \\
        --bids-root ./bids_output \\
        --freqs 6.0 8.0 10.0 12.0

    Or from Python:
        from neurobids_flow.ssvep.benchmark import SSVEPBenchmark
        bm = SSVEPBenchmark(bids_root="./bids_output", stim_freqs=[6.0, 8.0])
        bm.run()
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from .preprocessor import SSVEPPreprocessor
from .cca import CCA as SSVEP_CCA
from .fbcca import FBCCA
from .trca import TRCA
from .evaluator import SSVEPEvaluator, itr_bits_per_minute

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║       NeuroBIDS-Flow — SSVEP Cross-Device Benchmark             ║
║       CCA  ·  FBCCA  ·  TRCA  |  Per-Subject Results           ║
╚══════════════════════════════════════════════════════════════════╝"""


@dataclass
class SubjectBenchmarkResult:
    subject: str
    session: str
    device: str
    n_channels: int
    n_epochs: int
    cca_acc: float | None = None
    fbcca_acc: float | None = None
    trca_acc: float | None = None
    cca_itr: float | None = None
    fbcca_itr: float | None = None
    trca_itr: float | None = None
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class SSVEPBenchmark:
    """
    Cross-subject, cross-device SSVEP benchmark.

    Parameters
    ----------
    bids_root : str
        Path to BIDS output from Target 1 NeuroBIDS-Flow.
    stim_freqs : list[float]
        SSVEP stimulus frequencies in Hz.
    task : str
        BIDS task name.
    epoch_duration : float
        Epoch length for ITR calculation.
    gap_duration : float
        Inter-trial gap for ITR calculation.
    n_harmonics : int
        Harmonics for CCA / FBCCA.
    output_dir : str
        Where to save benchmark JSON results.
    """

    def __init__(
        self,
        bids_root: str = "./bids_output",
        stim_freqs: list[float] | None = None,
        task: str = "ssvep",
        epoch_duration: float = 2.0,
        gap_duration: float = 0.5,
        n_harmonics: int = 3,
        output_dir: str = "./results/ssvep_benchmark",
    ):
        self.bids_root = bids_root
        self.stim_freqs = stim_freqs or [6.0, 8.0, 10.0, 12.0]
        self.task = task
        self.epoch_duration = epoch_duration
        self.gap_duration = gap_duration
        self.n_harmonics = n_harmonics
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover subjects ──────────────────────────────────────────────────────
    def _discover_subjects(self) -> list[tuple[str, str]]:
        """Return list of (subject_id, session_id) found in BIDS root."""
        bids = Path(self.bids_root)
        pairs = []
        for sub_dir in sorted(bids.glob("sub-*")):
            if not sub_dir.is_dir():
                continue
            sub_id = sub_dir.name.replace("sub-", "")
            for ses_dir in sorted(sub_dir.glob("ses-*")):
                ses_id = ses_dir.name.replace("ses-", "")
                pairs.append((sub_id, ses_id))
        return pairs

    # ── Infer device from sidecar JSON ────────────────────────────────────────
    @staticmethod
    def _get_device(bids_root: str, sub: str, ses: str, task: str) -> str:
        pattern = (
            Path(bids_root)
            / f"sub-{sub}"
            / f"ses-{ses}"
            / "eeg"
            / f"sub-{sub}_ses-{ses}_task-{task}_eeg.json"
        )
        if pattern.exists():
            try:
                import json as _json
                with open(pattern) as f:
                    meta = _json.load(f)
                return meta.get("Manufacturer", "Unknown Device")
            except Exception:
                pass
        return "Unknown Device"

    # ── Per-subject evaluation ─────────────────────────────────────────────────
    def _eval_subject(
        self, sub: str, ses: str
    ) -> SubjectBenchmarkResult:
        device = self._get_device(self.bids_root, sub, ses, self.task)
        result = SubjectBenchmarkResult(
            subject=sub, session=ses, device=device,
            n_channels=0, n_epochs=0,
        )

        try:
            prep = SSVEPPreprocessor(
                bids_root=self.bids_root,
                task=self.task,
                stim_freqs=self.stim_freqs,
                subjects=[sub],
            )
            X, y = prep.load_and_preprocess()

            if X is None or len(X) == 0:
                result.note = "no epochs"
                return result

            result.n_channels = X.shape[1]
            result.n_epochs = len(y)
            n_classes = len(self.stim_freqs)
            sfreq = 256.0  # fallback; ideally read from sidecar

            ev = SSVEPEvaluator(
                n_classes=n_classes,
                epoch_duration=self.epoch_duration,
                gap_duration=self.gap_duration,
                n_splits=min(5, max(2, len(y) // n_classes)),
            )

            # CCA (training-free)
            cca = SSVEP_CCA(stim_freqs=self.stim_freqs, sfreq=sfreq, n_harmonics=self.n_harmonics)
            r_cca = ev.evaluate(cca, X, y, method_name="CCA", run_cv=False)
            result.cca_acc = round(r_cca.accuracy, 4)
            result.cca_itr = round(r_cca.itr_bpm, 2)

            # FBCCA (training-free)
            fbcca = FBCCA(stim_freqs=self.stim_freqs, sfreq=sfreq, n_harmonics=self.n_harmonics)
            r_fb = ev.evaluate(fbcca, X, y, method_name="FBCCA", run_cv=False)
            result.fbcca_acc = round(r_fb.accuracy, 4)
            result.fbcca_itr = round(r_fb.itr_bpm, 2)

            # TRCA (needs fit — at least 2 epochs per class)
            if len(y) >= n_classes * 2:
                trca = TRCA(stim_freqs=self.stim_freqs, sfreq=sfreq)
                r_trca = ev.evaluate(trca, X, y, method_name="TRCA", run_cv=False)
                result.trca_acc = round(r_trca.accuracy, 4)
                result.trca_itr = round(r_trca.itr_bpm, 2)
            else:
                result.note = "too few epochs for TRCA"

        except Exception as e:
            result.note = f"error: {e}"
            log.warning("  sub-%s | ses-%s | %s", sub, ses, e)

        return result

    # ── Print table ────────────────────────────────────────────────────────────
    @staticmethod
    def _print_table(rows: list[SubjectBenchmarkResult], total_time: float) -> None:
        def fmt(v: float | None) -> str:
            return f"{v:.4f}" if v is not None else "  N/A "

        def fmt_itr(v: float | None) -> str:
            return f"{v:6.2f}" if v is not None else "  N/A"

        w = 104
        print("\n" + "═" * w)
        print("  SSVEP CROSS-DEVICE BENCHMARK — SUMMARY")
        print("═" * w)
        header = (
            f"  {'Sub':>5}  {'Device':<22}  {'Ch':>4}  {'Ep':>4}  "
            f"{'CCA':>8}  {'FBCCA':>8}  {'TRCA':>8}  "
            f"{'ITR-CCA':>8}  {'ITR-FB':>8}  {'ITR-TRCA':>8}  Note"
        )
        print(header)
        print("─" * w)

        cca_accs, fb_accs, tr_accs = [], [], []
        cca_itrs, fb_itrs, tr_itrs = [], [], []

        for r in rows:
            line = (
                f"  {r.subject:>5}  {r.device:<22}  {r.n_channels:>4}  {r.n_epochs:>4}  "
                f"{fmt(r.cca_acc):>8}  {fmt(r.fbcca_acc):>8}  {fmt(r.trca_acc):>8}  "
                f"{fmt_itr(r.cca_itr):>8}  {fmt_itr(r.fbcca_itr):>8}  {fmt_itr(r.trca_itr):>8}  {r.note}"
            )
            print(line)
            if r.cca_acc is not None:
                cca_accs.append(r.cca_acc)
                cca_itrs.append(r.cca_itr or 0.0)
            if r.fbcca_acc is not None:
                fb_accs.append(r.fbcca_acc)
                fb_itrs.append(r.fbcca_itr or 0.0)
            if r.trca_acc is not None:
                tr_accs.append(r.trca_acc)
                tr_itrs.append(r.trca_itr or 0.0)

        print("─" * w)
        def mean_str(lst: list[float]) -> str:
            return f"{np.mean(lst):.4f}" if lst else "  N/A "

        print(
            f"  {'Mean':>5}  {'':22}  {'':4}  {'':4}  "
            f"{mean_str(cca_accs):>8}  {mean_str(fb_accs):>8}  {mean_str(tr_accs):>8}  "
            f"{mean_str(cca_itrs):>8}  {mean_str(fb_itrs):>8}  {mean_str(tr_itrs):>8}"
        )
        print("─" * w)
        print(f"  Chance level : {1.0/4:.4f}  |  Total runtime: {total_time:.1f}s")
        print(f"  Note: Results on synthetic data — expect near-chance. Real EEG will show above-chance performance.")
        print("═" * w)

    # ── Save results JSON ──────────────────────────────────────────────────────
    def _save_results(self, rows: list[SubjectBenchmarkResult]) -> Path:
        out = self.output_dir / "benchmark_results.json"
        payload = {
            "bids_root": self.bids_root,
            "stim_freqs": self.stim_freqs,
            "task": self.task,
            "subjects": [r.to_dict() for r in rows],
        }
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        log.info("Benchmark results saved: %s", out)
        return out

    # ── Main entry ─────────────────────────────────────────────────────────────
    def run(self) -> list[SubjectBenchmarkResult]:
        """Run full cross-device benchmark. Returns list of SubjectBenchmarkResult."""
        print(BANNER)
        t0 = time.time()

        subjects = self._discover_subjects()
        if not subjects:
            log.error("No subjects found in BIDS root: %s", self.bids_root)
            return []

        log.info("Found %d subject-session pairs", len(subjects))
        rows = []
        for sub, ses in subjects:
            log.info("Evaluating sub-%s ses-%s ...", sub, ses)
            r = self._eval_subject(sub, ses)
            rows.append(r)
            acc_str = f"CCA={r.cca_acc}  FBCCA={r.fbcca_acc}  TRCA={r.trca_acc}"
            log.info("  sub-%s | %s", sub, acc_str)

        self._print_table(rows, time.time() - t0)
        self._save_results(rows)
        return rows


# ── CLI ────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NeuroBIDS-Flow SSVEP Benchmark")
    p.add_argument("--bids-root", default="./bids_output")
    p.add_argument("--task", default="ssvep")
    p.add_argument("--freqs", nargs="+", type=float, default=[6.0, 8.0, 10.0, 12.0])
    p.add_argument("--epoch-duration", type=float, default=2.0)
    p.add_argument("--output-dir", default="./results/ssvep_benchmark")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    bm = SSVEPBenchmark(
        bids_root=args.bids_root,
        stim_freqs=args.freqs,
        task=args.task,
        epoch_duration=args.epoch_duration,
        output_dir=args.output_dir,
    )
    bm.run()