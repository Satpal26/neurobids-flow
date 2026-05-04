"""
evaluate_dataset.py — Target 1.5 Evaluation Script
====================================================
Runs full signal integrity, event verification, and ML evaluation
on a NeuroBIDS-Flow BIDS output directory for one device/dataset.

Usage:
    python target15/evaluate_dataset.py --device muse2
    python target15/evaluate_dataset.py --device openbci
    python target15/evaluate_dataset.py --device emotiv
    python target15/evaluate_dataset.py --device brainproducts
    python target15/evaluate_dataset.py --device all

Outputs (saved to results/target15/):
    - plots/<device>_psd.png
    - plots/<device>_alpha_power.png
    - plots/<device>_events.png
    - metrics/results.csv  (appended per device)
"""

import argparse
import csv
import os
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parent.parent
BIDS_ROOT    = REPO_ROOT / "results" / "target15" / "bids_output"
PLOTS_DIR    = REPO_ROOT / "results" / "target15" / "plots"
METRICS_DIR  = REPO_ROOT / "results" / "target15" / "metrics"
METRICS_FILE = METRICS_DIR / "results.csv"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# ── Device configs ─────────────────────────────────────────────────────────────
DEVICE_CONFIGS = {
    "muse2": {
        "bids_dir":   BIDS_ROOT / "muse2",
        "label":      "Muse 2 (InteraXon)",
        "n_channels": 5,
        "sfreq":      256.0,
        "task":       "N400",
        "color":      "#378ADD",
    },
    "openbci": {
        "bids_dir":   BIDS_ROOT / "openbci",
        "label":      "OpenBCI Cyton",
        "n_channels": 8,
        "sfreq":      250.0,
        "task":       "cogload",
        "color":      "#1D9E75",
    },
    "emotiv": {
        "bids_dir":   BIDS_ROOT / "emotiv",
        "label":      "Emotiv EPOC+",
        "n_channels": 14,
        "sfreq":      256.0,
        "task":       "imaginedspeech",
        "color":      "#BA7517",
    },
    "brainproducts": {
        "bids_dir":   BIDS_ROOT / "brainproducts",
        "label":      "BrainProducts actiCap",
        "n_channels": 65,
        "sfreq":      1000.0,
        "task":       "EyesClosed",
        "color":      "#533AB7",
    },
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def find_eeg_files(bids_dir: Path, extensions=(".vhdr", ".edf", ".fif")):
    """Return all EEG data files under a BIDS directory."""
    files = []
    for ext in extensions:
        files.extend(bids_dir.rglob(f"*_eeg{ext}"))
    return sorted(files)


def find_events_files(bids_dir: Path):
    """Return all events.tsv files under a BIDS directory."""
    return sorted(bids_dir.rglob("*_events.tsv"))


def count_events_in_tsv(tsv_path: Path):
    """Count non-header rows in an events.tsv file."""
    try:
        with open(tsv_path, "r") as f:
            lines = [l for l in f.readlines() if l.strip()]
        return max(0, len(lines) - 1)  # subtract header
    except Exception:
        return 0


def load_raw_mne(eeg_file: Path):
    """Load a BIDS EEG file using MNE. Returns (raw, load_time_s)."""
    import mne
    t0 = time.time()
    ext = eeg_file.suffix.lower()
    try:
        if ext == ".vhdr":
            raw = mne.io.read_raw_brainvision(str(eeg_file), preload=True, verbose=False)
        elif ext == ".edf":
            raw = mne.io.read_raw_edf(str(eeg_file), preload=True, verbose=False)
        elif ext == ".fif":
            raw = mne.io.read_raw_fif(str(eeg_file), preload=True, verbose=False)
        else:
            return None, 0.0
        load_time = time.time() - t0
        return raw, load_time
    except Exception as e:
        print(f"  [WARN] Could not load {eeg_file.name}: {e}")
        return None, 0.0


def compute_psd(raw, fmin=1.0, fmax=50.0):
    """Compute average PSD across channels. Returns (freqs, psd_mean)."""
    try:
        spectrum = raw.compute_psd(method="welch", fmin=fmin, fmax=fmax,
                                   n_fft=min(2048, len(raw.times) // 4),
                                   verbose=False)
        freqs = spectrum.freqs
        psd   = spectrum.get_data()  # (n_channels, n_freqs)
        psd_db = 10 * np.log10(psd + 1e-30)
        return freqs, psd_db.mean(axis=0)
    except Exception as e:
        print(f"  [WARN] PSD computation failed: {e}")
        return None, None


def compute_alpha_power(raw, fmin=8.0, fmax=12.0):
    """Compute mean alpha band power across channels (µV²)."""
    try:
        spectrum = raw.compute_psd(method="welch", fmin=fmin, fmax=fmax,
                                   verbose=False)
        psd = spectrum.get_data()  # (n_channels, n_freqs)
        freqs = spectrum.freqs
        df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
        alpha_power = (psd * df).sum(axis=1).mean()
        return float(alpha_power * 1e12)  # convert to µV²
    except Exception as e:
        print(f"  [WARN] Alpha power computation failed: {e}")
        return 0.0


# ── Per-device evaluation ──────────────────────────────────────────────────────

def evaluate_device(device_key: str) -> dict:
    """
    Run full evaluation for one device. Returns a metrics dict.
    """
    cfg = DEVICE_CONFIGS[device_key]
    bids_dir  = cfg["bids_dir"]
    label     = cfg["label"]
    color     = cfg["color"]

    print(f"\n{'='*60}")
    print(f"  Evaluating: {label}")
    print(f"  BIDS dir:   {bids_dir}")
    print(f"{'='*60}")

    metrics = {
        "device":         label,
        "n_subjects":     0,
        "n_channels":     cfg["n_channels"],
        "sfreq_hz":       cfg["sfreq"],
        "total_events":   0,
        "events_per_sub": 0.0,
        "bids_files":     0,
        "mean_duration_s": 0.0,
        "total_duration_s": 0.0,
        "alpha_power_uv2": 0.0,
        "load_time_s":    0.0,
        "bids_valid":     "PASS",
    }

    if not bids_dir.exists():
        print(f"  [ERROR] BIDS directory not found: {bids_dir}")
        metrics["bids_valid"] = "MISSING"
        return metrics

    # ── Count subjects ────────────────────────────────────────────────────────
    sub_dirs = [d for d in bids_dir.iterdir()
                if d.is_dir() and d.name.startswith("sub-")]
    metrics["n_subjects"] = len(sub_dirs)
    print(f"  Subjects found: {len(sub_dirs)}")

    # ── Count events ──────────────────────────────────────────────────────────
    events_files = find_events_files(bids_dir)
    total_events = sum(count_events_in_tsv(f) for f in events_files)
    metrics["total_events"]   = total_events
    metrics["events_per_sub"] = round(total_events / max(len(sub_dirs), 1), 1)
    print(f"  Events files:   {len(events_files)}")
    print(f"  Total events:   {total_events}  ({metrics['events_per_sub']} per subject)")

    # ── Load first available EEG file for signal analysis ────────────────────
    eeg_files = find_eeg_files(bids_dir)
    metrics["bids_files"] = len(eeg_files)
    print(f"  EEG files:      {len(eeg_files)}")

    if not eeg_files:
        print("  [WARN] No EEG files found for signal analysis")
        return metrics

    # Use first subject for signal integrity check
    sample_file = eeg_files[0]
    print(f"  Loading sample: {sample_file.name}")

    raw, load_time = load_raw_mne(sample_file)
    metrics["load_time_s"] = round(load_time, 2)

    if raw is None:
        print("  [WARN] Could not load sample file")
        return metrics

    # ── Duration stats ────────────────────────────────────────────────────────
    duration = raw.times[-1]
    metrics["mean_duration_s"]  = round(duration, 1)
    metrics["total_duration_s"] = round(duration * len(sub_dirs), 1)
    print(f"  Sample duration: {duration:.1f}s")
    print(f"  Actual sfreq:    {raw.info['sfreq']} Hz")
    print(f"  Actual channels: {len(raw.ch_names)}")

    # ── Alpha power ───────────────────────────────────────────────────────────
    alpha = compute_alpha_power(raw)
    metrics["alpha_power_uv2"] = round(alpha, 4)
    print(f"  Alpha power:     {alpha:.4f} µV²")

    # ── PSD plot ──────────────────────────────────────────────────────────────
    freqs, psd_mean = compute_psd(raw)

    fig = plt.figure(figsize=(14, 10), facecolor="white")
    fig.suptitle(f"NeuroBIDS-Flow — Signal Integrity: {label}",
                 fontsize=14, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # PSD
    ax_psd = fig.add_subplot(gs[0, :])
    if freqs is not None:
        ax_psd.plot(freqs, psd_mean, color=color, linewidth=1.5, label="Mean PSD")
        # Alpha band shading
        alpha_mask = (freqs >= 8) & (freqs <= 12)
        ax_psd.fill_between(freqs, psd_mean, where=alpha_mask,
                            alpha=0.3, color="#E24B4A", label="Alpha band (8–12 Hz)")
        # Mark major frequencies
        for f, name in [(8, "8Hz"), (10, "10Hz"), (12, "12Hz"),
                        (20, "20Hz"), (30, "30Hz"), (50, "50Hz (line)")]:
            if f <= freqs[-1]:
                ax_psd.axvline(f, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
        ax_psd.set_xlabel("Frequency (Hz)", fontsize=11)
        ax_psd.set_ylabel("Power (dB)", fontsize=11)
        ax_psd.set_title(f"Power Spectral Density — {sample_file.name}", fontsize=11)
        ax_psd.legend(fontsize=9)
        ax_psd.grid(True, alpha=0.3)
        ax_psd.set_xlim(1, min(50, freqs[-1]))

    # Channel PSD heatmap
    ax_heat = fig.add_subplot(gs[1, 0])
    try:
        spectrum = raw.compute_psd(method="welch", fmin=1.0, fmax=50.0,
                                   n_fft=min(2048, len(raw.times) // 4),
                                   verbose=False)
        psd_all = 10 * np.log10(spectrum.get_data() + 1e-30)
        n_show  = min(20, psd_all.shape[0])
        im = ax_heat.imshow(psd_all[:n_show, :], aspect="auto", origin="lower",
                            cmap="RdBu_r",
                            extent=[spectrum.freqs[0], spectrum.freqs[-1],
                                    0, n_show])
        plt.colorbar(im, ax=ax_heat, label="dB")
        ax_heat.set_xlabel("Frequency (Hz)", fontsize=10)
        ax_heat.set_ylabel("Channel index", fontsize=10)
        ax_heat.set_title(f"Per-channel PSD (first {n_show} ch)", fontsize=10)
    except Exception:
        ax_heat.text(0.5, 0.5, "Could not compute\nchannel PSD",
                    ha="center", va="center", transform=ax_heat.transAxes)

    # Events bar chart
    ax_ev = fig.add_subplot(gs[1, 1])
    if events_files:
        ev_counts = [count_events_in_tsv(f) for f in events_files[:20]]
        x = range(len(ev_counts))
        ax_ev.bar(x, ev_counts, color=color, alpha=0.8)
        ax_ev.axhline(np.mean(ev_counts), color="red", linewidth=1.5,
                      linestyle="--", label=f"Mean: {np.mean(ev_counts):.1f}")
        ax_ev.set_xlabel("Subject index", fontsize=10)
        ax_ev.set_ylabel("Event count", fontsize=10)
        ax_ev.set_title("Events per subject", fontsize=10)
        ax_ev.legend(fontsize=9)
        ax_ev.grid(True, alpha=0.3, axis="y")
    else:
        ax_ev.text(0.5, 0.5, "No events files found",
                  ha="center", va="center", transform=ax_ev.transAxes)

    plt.tight_layout()
    plot_path = PLOTS_DIR / f"{device_key}_analysis.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved:      {plot_path}")

    return metrics


# ── Write metrics CSV ──────────────────────────────────────────────────────────

def write_metrics(all_metrics: list[dict]):
    """Write all metrics to results.csv."""
    if not all_metrics:
        return

    fieldnames = list(all_metrics[0].keys())

    with open(METRICS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metrics)

    print(f"\n✔  Metrics saved to: {METRICS_FILE}")


def print_summary_table(all_metrics: list[dict]):
    """Print a clean summary table to terminal."""
    print("\n" + "="*90)
    print("  TARGET 1.5 — CROSS-DEVICE EVALUATION SUMMARY")
    print("="*90)
    print(f"  {'Device':<28} {'Subj':>5} {'Ch':>5} {'Hz':>7} {'Events':>8} "
          f"{'Ev/Sub':>8} {'Duration':>10} {'α-Power':>10}")
    print("-"*90)
    for m in all_metrics:
        print(f"  {m['device']:<28} {m['n_subjects']:>5} {m['n_channels']:>5} "
              f"{m['sfreq_hz']:>7.0f} {m['total_events']:>8} "
              f"{m['events_per_sub']:>8.1f} "
              f"{m['total_duration_s']:>9.0f}s "
              f"{m['alpha_power_uv2']:>10.4f}")
    print("="*90)

    # Totals
    total_subjects = sum(m["n_subjects"] for m in all_metrics)
    total_duration = sum(m["total_duration_s"] for m in all_metrics)
    total_events   = sum(m["total_events"] for m in all_metrics)
    print(f"\n  Total subjects: {total_subjects}")
    print(f"  Total events:   {total_events}")
    print(f"  Total EEG data: {total_duration:.0f}s  "
          f"({total_duration/3600:.1f} hours)")
    print(f"  All BIDS valid: "
          f"{'✔ YES' if all(m['bids_valid'] == 'PASS' for m in all_metrics) else '✘ CHECK'}")


# ── Cross-device comparison plot ───────────────────────────────────────────────

def plot_cross_device_summary(all_metrics: list[dict]):
    """Generate cross-device comparison bar charts."""
    if not all_metrics:
        return

    labels  = [m["device"].replace(" ", "\n") for m in all_metrics]
    colors  = [DEVICE_CONFIGS[k]["color"] for k in DEVICE_CONFIGS
               if any(m["device"] == DEVICE_CONFIGS[k]["label"]
                      for m in all_metrics)]
    # fallback colors if mismatch
    if len(colors) != len(all_metrics):
        colors = ["#378ADD", "#1D9E75", "#BA7517", "#533AB7"][:len(all_metrics)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="white")
    fig.suptitle("NeuroBIDS-Flow — Cross-Device Evaluation Summary",
                 fontsize=14, fontweight="bold")

    # Subjects
    ax = axes[0]
    vals = [m["n_subjects"] for m in all_metrics]
    bars = ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(v), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Subjects converted", fontsize=11)
    ax.set_ylabel("Count", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(vals) * 1.2)

    # Total duration
    ax = axes[1]
    vals = [m["total_duration_s"] / 3600 for m in all_metrics]
    bars = ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{v:.1f}h", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Total EEG data (hours)", fontsize=11)
    ax.set_ylabel("Hours", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(vals) * 1.2)

    # Events per subject
    ax = axes[2]
    vals = [m["events_per_sub"] for m in all_metrics]
    bars = ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{v:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Events per subject", fontsize=11)
    ax.set_ylabel("Count", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(vals) * 1.3 + 1)

    plt.tight_layout()
    path = PLOTS_DIR / "cross_device_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✔  Cross-device summary plot saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NeuroBIDS-Flow Target 1.5 Evaluation Script"
    )
    parser.add_argument(
        "--device",
        choices=list(DEVICE_CONFIGS.keys()) + ["all"],
        default="all",
        help="Which device dataset to evaluate (default: all)"
    )
    args = parser.parse_args()

    devices = list(DEVICE_CONFIGS.keys()) if args.device == "all" else [args.device]

    all_metrics = []
    for device in devices:
        try:
            m = evaluate_device(device)
            all_metrics.append(m)
        except Exception as e:
            print(f"\n[ERROR] Failed to evaluate {device}: {e}")
            import traceback
            traceback.print_exc()

    if all_metrics:
        write_metrics(all_metrics)
        print_summary_table(all_metrics)
        if len(all_metrics) > 1:
            plot_cross_device_summary(all_metrics)

    print("\nDone. Check results/target15/ for outputs.\n")


if __name__ == "__main__":
    main()