"""
NeuroBIDS-Flow — Cross-Device Evaluation Script
================================================
Runs CSP+LDA and EEGNet on each device separately,
then compares results in a summary table.

Usage:
    python src/neurobids_flow/cross_device_eval.py --bids-root ./bids_output
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

BANNER = """
╔══════════════════════════════════════════════════════════╗
║   NeuroBIDS-Flow — Cross-Device Evaluation               ║
║   CSP+LDA vs EEGNet | Per-Device + Combined              ║
╚══════════════════════════════════════════════════════════╝
"""

DEVICE_LABELS = {
    "Brain Products": "BrainProducts ActiChamp",
    "BrainProducts":  "BrainProducts ActiChamp",
    "OpenBCI":        "OpenBCI Cyton",
    "Muse":           "InteraXon Muse 2",
    "InteraXon":      "InteraXon Muse 2",
    "Emotiv":         "Emotiv EPOC+",
    "Neuroscan":      "Neuroscan NuAmps",
}


def load_single_subject(bids_root: str, subject: str, session: str,
                        task: str, tmin: float, tmax: float):
    import mne
    from mne_bids import BIDSPath, read_raw_bids
    from neurobids_flow.sklearn_pipeline import _clean_description

    try:
        bids_path = BIDSPath(
            subject=subject, session=session, task=task,
            root=str(bids_root), datatype="eeg",
        )
        raw = read_raw_bids(bids_path=bids_path, verbose=False)
        raw.load_data(verbose=False)

        if len(raw.annotations) == 0:
            return None, None, 0

        clean_descs = [_clean_description(a["description"]) for a in raw.annotations]
        raw.annotations.description[:] = clean_descs
        unique_descs = list(set(clean_descs))
        event_id = {desc: i + 1 for i, desc in enumerate(unique_descs)}

        events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)
        if len(events) == 0:
            return None, None, 0

        epochs = mne.Epochs(
            raw, events, event_id=event_id,
            tmin=tmin, tmax=tmax,
            baseline=None, preload=True, verbose=False
        )

        if len(epochs) == 0:
            return None, None, 0

        X = epochs.get_data()
        y = epochs.events[:, 2]
        return X, y, X.shape[1]

    except Exception as e:
        log.debug(f"sub-{subject}: {e}")
        return None, None, 0


def detect_device(subject: str, bids_root: str, session: str, task: str) -> str:
    import json
    bids_root = Path(bids_root)
    sidecar = (bids_root / f"sub-{subject}" / f"ses-{session}" / "eeg" /
               f"sub-{subject}_ses-{session}_task-{task}_eeg.json")
    if sidecar.exists():
        with open(sidecar) as f:
            meta = json.load(f)
        manufacturer = meta.get("Manufacturer", "")
        for key, label in DEVICE_LABELS.items():
            if key.lower() in manufacturer.lower():
                return label
    return "Unknown Device"


def eval_csp_lda(X: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import LabelEncoder
    from mne.decoding import CSP
    try:
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        n_classes = len(le.classes_)
        if n_classes < 2:
            return {"mean_acc": None, "std_acc": None,
                    "n_epochs": len(y), "status": "too few classes"}
        # Simple train/test split when too few samples for CV
        n_components = min(1, X.shape[1] - 1)
        csp = CSP(n_components=n_components, reg=None, log=True)
        lda = LinearDiscriminantAnalysis()
        mid = max(1, len(y_enc) // 2)
        X_train, X_test = X[:mid], X[mid:]
        y_train, y_test = y_enc[:mid], y_enc[mid:]
        X_train_csp = csp.fit_transform(X_train, y_train)
        lda.fit(X_train_csp, y_train)
        X_test_csp = csp.transform(X_test)
        acc = float((lda.predict(X_test_csp) == y_test).mean())
        return {"mean_acc": acc, "std_acc": 0.0,
                "n_epochs": len(y_enc), "status": "ok (holdout)"}
    except Exception as e:
        return {"mean_acc": None, "std_acc": None,
                "n_epochs": len(y), "status": f"failed: {e}"}


def eval_eegnet(X: np.ndarray, y: np.ndarray) -> dict:
    from neurobids_flow.braindecode_pipeline import (
        build_eegnet, EEGArrayDataset, train_epoch, eval_epoch
    )
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, random_split
    from sklearn.preprocessing import LabelEncoder

    try:
        le = LabelEncoder()
        y_enc = le.fit_transform(y).astype(np.int64)
        n_classes = len(le.classes_)
        n_channels, n_times = X.shape[1], X.shape[2]

        if n_classes < 2 or len(y_enc) < 2:
            return {"best_acc": None, "n_epochs": len(y_enc),
                    "status": "too few samples"}

        device = torch.device("cpu")
        dataset = EEGArrayDataset(X, y_enc)
        n_val = max(1, len(dataset) // 4)
        n_train = max(1, len(dataset) - n_val)
        train_ds, val_ds = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        train_loader = DataLoader(train_ds, batch_size=max(1, n_train), shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=max(1, n_val), shuffle=False)

        model = build_eegnet(n_channels, n_times, n_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        best_acc = 0.0
        for _ in range(15):
            train_epoch(model, train_loader, optimizer, criterion, device)
            _, val_acc = eval_epoch(model, val_loader, criterion, device)
            if val_acc > best_acc:
                best_acc = val_acc

        return {"best_acc": best_acc, "n_epochs": len(y_enc), "status": "ok"}

    except Exception as e:
        return {"best_acc": None, "n_epochs": len(y),
                "status": f"failed: {e}"}


def run_evaluation(bids_root: str, task: str = "workload",
                   tmin: float = 0.0, tmax: float = 3.0) -> list:

    print(BANNER)
    log.info(f"BIDS root : {bids_root}")
    log.info(f"Task      : {task}")
    print()

    bids_root_path = Path(bids_root)
    subject_dirs = sorted([
        d for d in bids_root_path.iterdir()
        if d.is_dir() and d.name.startswith("sub-")
    ])

    if not subject_dirs:
        raise RuntimeError(f"No subjects found in {bids_root}")

    rows = []

    for sub_dir in subject_dirs:
        subject = sub_dir.name.replace("sub-", "")
        session_dirs = sorted([
            s for s in sub_dir.iterdir()
            if s.is_dir() and s.name.startswith("ses-")
        ])
        session = session_dirs[0].name.replace("ses-", "") if session_dirs else "01"

        log.info(f"Evaluating sub-{subject} ses-{session}...")

        X, y, n_ch = load_single_subject(
            bids_root, subject, session, task, tmin, tmax
        )
        device_label = detect_device(subject, bids_root, session, task)

        if X is None:
            rows.append({
                "subject": f"sub-{subject}",
                "device": device_label,
                "n_channels": n_ch,
                "n_epochs": 0,
                "csp_lda_acc": None,
                "eegnet_acc": None,
                "status": "no epochs",
            })
            log.warning(f"  sub-{subject} | No epochs — skipping")
            continue

        log.info(f"  sub-{subject} | device={device_label} | ch={n_ch} | epochs={len(y)}")

        csp_result = eval_csp_lda(X, y)
        eegnet_result = eval_eegnet(X, y)

        csp_acc = csp_result["mean_acc"]
        eegnet_acc = eegnet_result["best_acc"]

        rows.append({
            "subject": f"sub-{subject}",
            "device": device_label,
            "n_channels": n_ch,
            "n_epochs": len(y),
            "csp_lda_acc": csp_acc,
            "eegnet_acc": eegnet_acc,
            "status": "ok",
        })

        csp_str = f"{csp_acc:.3f}" if csp_acc is not None else "N/A"
        eeg_str = f"{eegnet_acc:.3f}" if eegnet_acc is not None else "N/A"
        log.info(f"  CSP+LDA={csp_str}  EEGNet={eeg_str}")

    return rows


def print_summary_table(rows: list):
    print("\n" + "═" * 80)
    print("  CROSS-DEVICE EVALUATION SUMMARY")
    print("═" * 80)
    print(f"  {'Subject':<10} {'Device':<26} {'Ch':>3} {'Epochs':>6} "
          f"{'CSP+LDA':>8} {'EEGNet':>8}")
    print("─" * 80)

    valid_csp, valid_eeg = [], []

    for row in rows:
        csp_str = f"{row['csp_lda_acc']:.3f}" if row['csp_lda_acc'] is not None else "  N/A "
        eeg_str = f"{row['eegnet_acc']:.3f}" if row['eegnet_acc'] is not None else "  N/A "
        status = "" if row["status"] == "ok" else f"  [{row['status']}]"

        print(f"  {row['subject']:<10} {row['device']:<26} "
              f"{row['n_channels']:>3} {row['n_epochs']:>6} "
              f"{csp_str:>8} {eeg_str:>8}{status}")

        if row["csp_lda_acc"] is not None:
            valid_csp.append(row["csp_lda_acc"])
        if row["eegnet_acc"] is not None:
            valid_eeg.append(row["eegnet_acc"])

    print("─" * 80)
    if valid_csp:
        print(f"  {'Mean (CSP+LDA)':<38} {np.mean(valid_csp):>8.3f}")
    if valid_eeg:
        print(f"  {'Mean (EEGNet)':<38} {np.mean(valid_eeg):>8.3f}")

    print("═" * 80)
    print(f"\n  Chance level (2-class): 0.500")
    print(f"  Note: Results on synthetic data — expect near-chance accuracy.")
    print(f"  With real passive BCI recordings, both models perform above chance.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NeuroBIDS-Flow Cross-Device Evaluation"
    )
    parser.add_argument("--bids-root", default="./bids_output")
    parser.add_argument("--task", default="workload")
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=3.0)
    args = parser.parse_args()

    rows = run_evaluation(
        bids_root=args.bids_root,
        task=args.task,
        tmin=args.tmin,
        tmax=args.tmax,
    )
    print_summary_table(rows)