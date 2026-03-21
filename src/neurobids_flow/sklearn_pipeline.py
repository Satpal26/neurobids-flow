"""
NeuroBIDS-Flow — Scikit-learn Baseline Classifier
==================================================
CSP + LDA pipeline on BIDS-EEG data from multiple devices.

Usage:
    python src/neurobids_flow/sklearn_pipeline.py --bids-root ./bids_output
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from mne.decoding import CSP

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

BANNER = """
╔══════════════════════════════════════════════════════════╗
║   NeuroBIDS-Flow — CSP + LDA Baseline Classifier        ║
║   Passive BCI | Cross-Device Evaluation                  ║
╚══════════════════════════════════════════════════════════╝
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_description(desc: str) -> str:
    """
    Clean up annotation descriptions.
    Handles HED dicts stored as strings, unknown_ prefixes, etc.
    """
    desc = str(desc).strip()

    # Handle HED dict stored as string e.g. "{'trial_type': 'rest_open', 'hed': '...'}"
    if desc.startswith("{") and "trial_type" in desc:
        try:
            import ast
            d = ast.literal_eval(desc)
            return d.get("trial_type", desc)
        except Exception:
            pass

    # Strip unknown_ prefix added by EventHarmonizer
    if desc.startswith("unknown_"):
        desc = desc[len("unknown_"):]

    return desc


def load_epochs_from_bids(bids_root: str, task: str = "workload",
                           tmin: float = 0.0, tmax: float = 3.0):
    """
    Load all subjects from BIDS root, epoch by annotations,
    return X (n_epochs, n_channels, n_times) and y (n_epochs,).
    Handles cross-device channel count mismatch by trimming to minimum.
    """
    import mne
    from mne_bids import BIDSPath, read_raw_bids

    bids_root = Path(bids_root)
    all_X, all_y, all_subjects, all_n_channels = [], [], [], []

    subject_dirs = sorted([
        d for d in bids_root.iterdir()
        if d.is_dir() and d.name.startswith("sub-")
    ])

    if not subject_dirs:
        raise RuntimeError(f"No subjects found in {bids_root}")

    EEG_EXTENSIONS = {".vhdr", ".edf", ".bdf", ".set", ".fif"}

    for sub_dir in subject_dirs:
        subject = sub_dir.name.replace("sub-", "")
        session_dirs = sorted([
            s for s in sub_dir.iterdir()
            if s.is_dir() and s.name.startswith("ses-")
        ])
        if not session_dirs:
            session_dirs = [sub_dir]

        for ses_dir in session_dirs:
            session = ses_dir.name.replace("ses-", "") if ses_dir.name.startswith("ses-") else "01"
            eeg_dir = ses_dir / "eeg"
            if not eeg_dir.exists():
                continue

            eeg_files = [
                f for f in eeg_dir.iterdir()
                if f.suffix in EEG_EXTENSIONS and "task-" in f.name
            ]
            if not eeg_files:
                continue

            try:
                bids_path = BIDSPath(
                    subject=subject,
                    session=session,
                    task=task,
                    root=str(bids_root),
                    datatype="eeg",
                )
                raw = read_raw_bids(bids_path=bids_path, verbose=False)
                raw.load_data(verbose=False)

                annotations = raw.annotations
                if len(annotations) == 0:
                    log.warning(f"sub-{subject} | No annotations — skipping")
                    continue

                # Clean descriptions
                clean_descs = [_clean_description(a["description"])
                               for a in annotations]

                # Rename annotations with cleaned descriptions
                raw.annotations.description[:] = clean_descs

                unique_descs = list(set(clean_descs))
                event_id = {desc: i + 1 for i, desc in enumerate(unique_descs)}

                events, _ = mne.events_from_annotations(
                    raw, event_id=event_id, verbose=False
                )

                if len(events) == 0:
                    log.warning(f"sub-{subject} | No events extracted — skipping")
                    continue

                epochs = mne.Epochs(
                    raw, events, event_id=event_id,
                    tmin=tmin, tmax=tmax,
                    baseline=None, preload=True, verbose=False
                )

                if len(epochs) == 0:
                    log.warning(f"sub-{subject} | No epochs after rejection — skipping")
                    continue

                X = epochs.get_data()
                y = epochs.events[:, 2]

                all_X.append(X)
                all_y.append(y)
                all_n_channels.append(X.shape[1])
                all_subjects.extend([f"sub-{subject}"] * len(epochs))

                log.info(
                    f"sub-{subject} | ses-{session} | "
                    f"ch={X.shape[1]} | epochs={X.shape[0]} | "
                    f"classes={list(event_id.keys())}"
                )

            except Exception as e:
                log.warning(f"sub-{subject} | Failed: {e} — skipping")
                continue

    if not all_X:
        raise RuntimeError(
            "No epochs could be extracted from any subject. "
            "Check that your BIDS data has event annotations."
        )

    # Trim all to minimum common channel count (cross-device compatibility)
    min_ch = min(all_n_channels)
    if len(set(all_n_channels)) > 1:
        log.info(
            f"Channel counts across devices: {all_n_channels} — "
            f"trimming all to {min_ch} channels"
        )
        all_X = [x[:, :min_ch, :] for x in all_X]

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    return X, y, all_subjects


def run_csp_lda(X: np.ndarray, y: np.ndarray, n_components: int = 4,
                n_splits: int = 5) -> dict:
    """
    Run CSP + LDA pipeline with stratified k-fold cross-validation.
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    n_classes = len(le.classes_)
    log.info(f"Classes     : {list(le.classes_)} ({n_classes} total)")
    log.info(f"Total epochs: {len(y_enc)}")

    if n_classes < 2:
        raise RuntimeError(
            f"Need at least 2 classes, got {n_classes}."
        )

    n_components = min(n_components, X.shape[1] - 1, len(y_enc) // n_splits)
    n_components = max(n_components, 1)

    pipeline = Pipeline([
        ("csp", CSP(n_components=n_components, reg=None,
                    log=True, norm_trace=False)),
        ("lda", LinearDiscriminantAnalysis()),
    ])

    cv = StratifiedKFold(
        n_splits=min(n_splits, len(y_enc) // n_classes),
        shuffle=True, random_state=42
    )

    scores = cross_val_score(pipeline, X, y_enc, cv=cv, scoring="accuracy")

    return {
        "scores": scores,
        "mean_accuracy": float(np.mean(scores)),
        "std_accuracy": float(np.std(scores)),
        "n_epochs": len(y_enc),
        "n_classes": n_classes,
        "class_labels": list(le.classes_),
        "n_components": n_components,
        "n_splits": cv.n_splits,
    }


def print_results(results: dict):
    print("\n" + "─" * 60)
    print("  CSP + LDA Classification Results")
    print("─" * 60)
    print(f"  Classes         : {results['class_labels']}")
    print(f"  Total epochs    : {results['n_epochs']}")
    print(f"  CSP components  : {results['n_components']}")
    print(f"  CV folds        : {results['n_splits']}")
    print(f"  Fold accuracies : {[f'{s:.3f}' for s in results['scores']]}")
    print(f"  Mean accuracy   : {results['mean_accuracy']:.3f} "
          f"± {results['std_accuracy']:.3f}")
    print(f"  Chance level    : {1.0 / results['n_classes']:.3f}")
    print("─" * 60)
    if results['mean_accuracy'] > 1.0 / results['n_classes']:
        print("  ✓ Above chance level")
    else:
        print("  ✗ At or below chance (expected for synthetic data)")
    print()


def run_pipeline(bids_root: str, task: str = "workload",
                 tmin: float = 0.0, tmax: float = 3.0,
                 n_components: int = 4, n_splits: int = 5) -> dict:
    """Full pipeline: load BIDS → epoch → CSP + LDA → cross-validate."""
    print(BANNER)
    log.info(f"BIDS root  : {bids_root}")
    log.info(f"Task       : {task}")
    log.info(f"Epoch      : [{tmin}, {tmax}] s")
    print()

    log.info("Loading epochs from BIDS dataset...")
    X, y, subjects = load_epochs_from_bids(
        bids_root=bids_root, task=task, tmin=tmin, tmax=tmax
    )
    log.info(f"Loaded X={X.shape}, y={y.shape}")
    print()

    log.info("Running CSP + LDA cross-validation...")
    results = run_csp_lda(X, y, n_components=n_components, n_splits=n_splits)
    print_results(results)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NeuroBIDS-Flow CSP+LDA Baseline Classifier"
    )
    parser.add_argument("--bids-root", default="./bids_output")
    parser.add_argument("--task", default="workload")
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=3.0)
    parser.add_argument("--n-components", type=int, default=4)
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    run_pipeline(
        bids_root=args.bids_root,
        task=args.task,
        tmin=args.tmin,
        tmax=args.tmax,
        n_components=args.n_components,
        n_splits=args.n_splits,
    )