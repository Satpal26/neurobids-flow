"""
NeuroBIDS-Flow — Nakanishi 2018 SSVEP Benchmark
=================================================
Loads the 12JFPM SSVEP dataset (Nakanishi et al. 2018) and runs
all three SSVEPFlow classifiers: CCA, FBCCA, TRCA.

Dataset: 10 subjects, 12 frequencies, 8 electrodes, 256 Hz
Data format: .mat files, shape (n_ch, n_times, n_trials, n_freqs)

Usage:
    python src/neurobids_flow/evaluate_nakanishi.py
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import scipy.io as sio

from scipy.signal import butter, filtfilt

def bandpass(X: np.ndarray, low: float = 7.0, high: float = 90.0,
             fs: float = 256.0, order: int = 4) -> np.ndarray:
    """Bandpass filter epochs. X: (n_epochs, n_ch, n_times)"""
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype='band')
    return filtfilt(b, a, X, axis=-1)

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
log = logging.getLogger(__name__)

# ── Dataset constants ─────────────────────────────────────────────────────────
DATA_DIR     = Path("datasets/ssvep_nakanishi/data")
SFREQ        = 256.0          # Hz
N_SUBJECTS   = 10
N_FREQS      = 12             # stimulus frequencies
N_TRIALS     = 15             # trials per frequency
N_CH         = 8              # EEG channels
EPOCH_DUR    = 2.0            # seconds
GAP_DUR      = 0.5            # inter-trial gap (for ITR)
STIM_FREQS   = [9.25, 11.25, 13.25, 9.75, 11.75, 13.75,
                10.25, 12.25, 14.25, 10.75, 12.75, 14.75]

# Samples: skip first 0.5s (visual latency), take 2s epoch
SKIP_SAMPLES = int(0.5 * SFREQ)   # 128 samples
EPOCH_SAMPLES = int(EPOCH_DUR * SFREQ)  # 512 samples


# ── Data loader ───────────────────────────────────────────────────────────────

def load_subject(subject_id: int) -> tuple[np.ndarray, np.ndarray]:
    path = DATA_DIR / f"s{subject_id}.mat"
    mat = sio.loadmat(str(path))

    # shape: (n_freqs, n_channels, n_times, n_trials) = (12, 8, 1114, 15)
    data = mat["eeg"]
    n_freqs, n_ch, n_times, n_trials = data.shape

    X_list, y_list = [], []
    for freq_idx in range(n_freqs):
        for trial_idx in range(n_trials):
            epoch = data[freq_idx, :, SKIP_SAMPLES:SKIP_SAMPLES + EPOCH_SAMPLES, trial_idx]
            X_list.append(epoch)
            y_list.append(freq_idx)

    X = np.array(X_list, dtype=np.float64)  # (180, 8, 512)
    y = np.array(y_list, dtype=np.int64)    # (180,)
    return X, y

# ── Main evaluation ───────────────────────────────────────────────────────────

def run_benchmark():
    from neurobids_flow.ssvep.cca import CCA
    from neurobids_flow.ssvep.fbcca import FBCCA
    from neurobids_flow.ssvep.trca import TRCA
    from neurobids_flow.ssvep.evaluator import SSVEPEvaluator, itr_bits_per_minute

    print("\n" + "═" * 65)
    print("  NeuroBIDS-Flow — SSVEPFlow Benchmark")
    print("  Dataset: Nakanishi 2018 | 10 subjects | 12 freqs | 256 Hz")
    print("═" * 65)

    evaluator = SSVEPEvaluator(
        n_classes=N_FREQS,
        epoch_duration=EPOCH_DUR,
        gap_duration=GAP_DUR,
        n_splits=5,
    )

    methods = ["CCA", "FBCCA", "TRCA"]
    all_results = {m: [] for m in methods}

    for sub_id in range(1, N_SUBJECTS + 1):
        log.info(f"Subject {sub_id:02d}/{N_SUBJECTS}")
        X, y = load_subject(sub_id)
        X_filt = bandpass(X)

        # ── CCA ──────────────────────────────────────────────────────────────
        cca = CCA(
            stim_freqs=STIM_FREQS,
            sfreq=SFREQ,
            n_harmonics=3,
        )
        t0 = time.time()
        result_cca = evaluator.evaluate(cca, X, y, method_name="CCA", run_cv=False)
        t_cca = time.time() - t0
        all_results["CCA"].append(result_cca.accuracy)
        log.info(f"  CCA   acc={result_cca.accuracy:.3f} itr={result_cca.itr_bpm:.1f} bpm  ({t_cca:.1f}s)")

        # ── FBCCA ────────────────────────────────────────────────────────────
        fbcca = FBCCA(
            stim_freqs=STIM_FREQS,
            sfreq=SFREQ,
            n_harmonics=3,
        )
        t0 = time.time()
        result_fbcca = evaluator.evaluate(fbcca, X, y, method_name="FBCCA", run_cv=False)
        t_fbcca = time.time() - t0
        all_results["FBCCA"].append(result_fbcca.accuracy)
        log.info(f"  FBCCA acc={result_fbcca.accuracy:.3f} itr={result_fbcca.itr_bpm:.1f} bpm  ({t_fbcca:.1f}s)")

# TRCA with leave-one-trial-out CV on filtered data
        correct, total = 0, 0
        for test_trial in range(N_TRIALS):
            mask = np.ones(len(y), dtype=bool)
            for cls in range(N_FREQS):
                mask[np.where(y == cls)[0][test_trial]] = False
            trca = TRCA(stim_freqs=STIM_FREQS, sfreq=SFREQ, ensemble=True)
            trca.fit(X_filt[mask], y[mask])
            preds = trca.predict(X_filt[~mask])
            correct += np.sum(preds == y[~mask])
            total += len(y[~mask])
        acc_trca = correct / total
        itr_trca = itr_bits_per_minute(N_FREQS, acc_trca, EPOCH_DUR, GAP_DUR)
        t_trca = time.time() - t0
        all_results["TRCA"].append(acc_trca)
        log.info(f"  eTRCA acc={acc_trca:.3f} itr={itr_trca:.1f} bpm  ({t_trca:.1f}s)")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  BENCHMARK SUMMARY — Mean across 10 subjects")
    print("═" * 65)
    print(f"  {'Method':<10} {'Mean Acc':>10} {'Std':>8} {'ITR (bpm)':>12} {'Min':>8} {'Max':>8}")
    print("─" * 65)

    for method in methods:
        accs = np.array(all_results[method])
        mean_acc = np.mean(accs)
        std_acc  = np.std(accs)
        min_acc  = np.min(accs)
        max_acc  = np.max(accs)
        mean_itr = itr_bits_per_minute(N_FREQS, mean_acc, EPOCH_DUR, GAP_DUR)
        print(f"  {method:<10} {mean_acc:>10.3f} {std_acc:>8.3f} {mean_itr:>12.2f} {min_acc:>8.3f} {max_acc:>8.3f}")

    print("─" * 65)
    print(f"  Chance level: {1/N_FREQS:.3f} ({100/N_FREQS:.1f}%)")
    print(f"  Epoch: {EPOCH_DUR}s | Subjects: {N_SUBJECTS} | Freqs: {N_FREQS}")
    print("═" * 65)

    # Save results
    import json
    results_path = Path("results/ssvep_benchmark/nakanishi_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        method: {
            "mean_acc": float(np.mean(all_results[method])),
            "std_acc":  float(np.std(all_results[method])),
            "per_subject": [float(a) for a in all_results[method]],
            "mean_itr_bpm": float(itr_bits_per_minute(
                N_FREQS, np.mean(all_results[method]), EPOCH_DUR, GAP_DUR
            )),
        }
        for method in methods
    }
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    run_benchmark()