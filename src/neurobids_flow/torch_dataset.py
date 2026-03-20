# torch_dataset.py
# NeuroBIDS-Flow — PyTorch Dataset Wrapper
# ─────────────────────────────────────────────────────────────────────────────
# Wraps NeuroBIDS-Flow BIDS-EEG output into a torch.utils.data.Dataset
# so it can be used directly with PyTorch DataLoader for deep learning.
#
# Place this file at:
#   src/neurobids_flow/torch_dataset.py
#
# Quick usage:
#   from neurobids_flow.torch_dataset import NeuroBIDSFlowTorchDataset
#   dataset = NeuroBIDSFlowTorchDataset(bids_root="./bids_output")
#   loader  = DataLoader(dataset, batch_size=32, shuffle=True)
#   for X, y in loader:
#       print(X.shape, y.shape)  # (32, n_channels, n_times), (32,)
#
# NeuroBIDS-Flow | NTU Singapore BCI Lab
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from neurobids_flow.moabb_wrapper import NBIDSFDataset, NEUROBIDS_EVENTS

logger = logging.getLogger(__name__)


class NeuroBIDSFlowTorchDataset(Dataset):
    """
    PyTorch Dataset wrapping NeuroBIDS-Flow BIDS-EEG output.

    Loads EEG data from a BIDS root directory produced by NeuroBIDS-Flow,
    slices it into fixed-size epochs around event onsets, and returns
    (X, y) pairs as PyTorch tensors ready for deep learning.

    Internally uses NBIDSFDataset (MOABB wrapper) to load MNE Raw objects,
    then epochs them using MNE and converts to tensors.

    Parameters
    ----------
    bids_root : str
        Path to BIDS root directory produced by NeuroBIDS-Flow.
    task : str
        BIDS task label. Must match --task used during conversion.
        Default: "workload"
    subjects : list[int], optional
        Subject IDs to include. Auto-detected if None.
    events : dict[str, int], optional
        Event label → integer class mapping.
        Default: NEUROBIDS_EVENTS from moabb_wrapper.
    tmin : float
        Epoch start in seconds relative to event onset. Default: -0.5
    tmax : float
        Epoch end in seconds relative to event onset. Default: 3.0
    baseline : tuple, optional
        Baseline correction window. Default: None (no baseline)
    dtype : torch.dtype
        Tensor dtype. Default: torch.float32

    Examples
    --------
    Basic usage:

        from neurobids_flow.torch_dataset import NeuroBIDSFlowTorchDataset
        from torch.utils.data import DataLoader

        dataset = NeuroBIDSFlowTorchDataset(bids_root="./bids_output")
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        for X, y in loader:
            print(X.shape)  # (32, n_channels, n_times)
            print(y.shape)  # (32,)

    Binary cognitive workload classification:

        dataset = NeuroBIDSFlowTorchDataset(
            bids_root="./bids_output",
            events={"cognitive_low": 0, "cognitive_high": 1},
            tmin=0.0,
            tmax=4.0,
        )

    Train/val split:

        from torch.utils.data import random_split
        train, val = random_split(dataset, [0.8, 0.2])
        train_loader = DataLoader(train, batch_size=32, shuffle=True)
        val_loader   = DataLoader(val,   batch_size=32, shuffle=False)
    """

    def __init__(
        self,
        bids_root: str = "./bids_output",
        task: str = "workload",
        subjects: Optional[list[int]] = None,
        events: Optional[dict[str, int]] = None,
        tmin: float = -0.5,
        tmax: float = 3.0,
        baseline: Optional[tuple] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.bids_root = Path(bids_root).resolve()
        self.task = task
        self.tmin = tmin
        self.tmax = tmax
        self.baseline = baseline
        self.dtype = dtype
        self._events = events or NEUROBIDS_EVENTS

        # ── Load and epoch all data ───────────────────────────────────
        self.X, self.y = self._load_all(subjects)

        logger.info(
            f"NeuroBIDSFlowTorchDataset ready | "
            f"n_epochs={len(self.y)} | "
            f"shape={self.X.shape} | "
            f"classes={torch.unique(self.y).tolist()}"
        )

    # ─────────────────────────────────────────────────────────────────
    # PyTorch Dataset interface
    # ─────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    # ─────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────

    @property
    def n_channels(self) -> int:
        """Number of EEG channels."""
        return self.X.shape[1]

    @property
    def n_times(self) -> int:
        """Number of time samples per epoch."""
        return self.X.shape[2]

    @property
    def n_classes(self) -> int:
        """Number of unique class labels."""
        return len(torch.unique(self.y))

    @property
    def class_labels(self) -> list[str]:
        """List of event label strings."""
        return list(self._events.keys())

    # ─────────────────────────────────────────────────────────────────
    # Data loading
    # ─────────────────────────────────────────────────────────────────

    def _load_all(
        self,
        subjects: Optional[list[int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load all subjects, epoch the data, return (X, y) tensors.

        Flow:
          NBIDSFDataset._get_single_subject_data()
            → MNE Raw objects with events.tsv annotations
            → mne.Epochs sliced around event onsets
            → NumPy arrays
            → PyTorch tensors
        """
        import mne

        moabb_dataset = NBIDSFDataset(
            bids_root=str(self.bids_root),
            task=self.task,
            subjects=subjects,
            events=self._events,
            interval=[self.tmin, self.tmax],
        )

        all_X: list[np.ndarray] = []
        all_y: list[np.ndarray] = []

        # Build reverse map: integer → label for MNE event_id
        event_id = {k: v for k, v in self._events.items()}

        for subject in moabb_dataset.subject_list:
            subject_data = moabb_dataset._get_single_subject_data(subject)

            for session_key, runs in subject_data.items():
                for run_key, raw in runs.items():
                    try:
                        # Match expected event labels against actual annotations
                        ann_descriptions = set(raw.annotations.description)
                        matched = {k: v for k, v in event_id.items()
                                   if k in ann_descriptions}

                        # Fallback — auto-map whatever annotations exist to integers
                        if not matched:
                            matched = {desc: i + 1 for i, desc in
                                       enumerate(sorted(ann_descriptions))}
                            logger.warning(
                                f"No expected event labels found in annotations "
                                f"for sub-{subject:02d} {session_key}/{run_key}. "
                                f"Auto-mapping annotations: {matched}"
                            )

                        # Extract events from annotations (written by NeuroBIDS-Flow)
                        events_array, _ = mne.events_from_annotations(
                            raw,
                            event_id=matched,
                            verbose=False,
                        )

                        if len(events_array) == 0:
                            logger.warning(
                                f"No matching events for sub-{subject:02d} "
                                f"{session_key}/{run_key} — skipping."
                            )
                            continue

                        # Epoch the raw data
                        epochs = mne.Epochs(
                            raw,
                            events=events_array,
                            event_id=matched,
                            tmin=self.tmin,
                            tmax=self.tmax,
                            baseline=self.baseline,
                            preload=True,
                            verbose=False,
                        )

                        # (n_epochs, n_channels, n_times)
                        X_sub = epochs.get_data()
                        y_sub = epochs.events[:, 2]  # integer labels

                        all_X.append(X_sub)
                        all_y.append(y_sub)

                        logger.info(
                            f"Epoched | sub-{subject:02d} | {session_key}/{run_key} | "
                            f"epochs={len(y_sub)} | shape={X_sub.shape}"
                        )

                    except Exception as exc:
                        logger.warning(
                            f"Epoching failed for sub-{subject:02d} "
                            f"{session_key}/{run_key}: {exc} — skipping."
                        )
                        continue

        if not all_X:
            raise RuntimeError(
                "No epochs could be extracted from the BIDS dataset.\n"
                "Check that event trial_type labels in events.tsv match "
                "the keys in your events dictionary."
            )

        # Stack all epochs across subjects/sessions/runs
        X_np = np.concatenate(all_X, axis=0)   # (total_epochs, n_ch, n_times)
        y_np = np.concatenate(all_y, axis=0)   # (total_epochs,)

        X_tensor = torch.tensor(X_np, dtype=self.dtype)
        y_tensor = torch.tensor(y_np, dtype=torch.long)

        return X_tensor, y_tensor


# ─────────────────────────────────────────────────────────────────────────────
# Quick-start demo  (python torch_dataset.py --bids-root ./bids_output)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader, random_split

    parser = argparse.ArgumentParser(
        description="NeuroBIDSFlow PyTorch Dataset — quick-start demo"
    )
    parser.add_argument("--bids-root", default="./bids_output")
    parser.add_argument("--task", default="workload")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("\n" + "─" * 60)
    print("  NeuroBIDSFlow PyTorch Dataset — Quick-start Demo")
    print("─" * 60)

    dataset = NeuroBIDSFlowTorchDataset(
        bids_root=args.bids_root,
        task=args.task,
    )

    print(f"\n✓ Dataset loaded")
    print(f"  Total epochs  : {len(dataset)}")
    print(f"  X shape       : {dataset.X.shape}  (epochs, channels, times)")
    print(f"  y shape       : {dataset.y.shape}")
    print(f"  n_channels    : {dataset.n_channels}")
    print(f"  n_times       : {dataset.n_times}")
    print(f"  n_classes     : {dataset.n_classes}")
    print(f"  Classes       : {dataset.class_labels}")

    # Single item
    X_item, y_item = dataset[0]
    print(f"\n  Single epoch  : X={X_item.shape}  y={y_item.item()}")

    # DataLoader
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    X_batch, y_batch = next(iter(loader))
    print(f"  Batch shape   : X={X_batch.shape}  y={y_batch.shape}")

    # Train/val split
    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = n - n_train
    if n_train > 0 and n_val > 0:
        train_set, val_set = random_split(dataset, [n_train, n_val])
        print(f"\n  Train/val split: {n_train} / {n_val}")

    print("""
─── Next steps ──────────────────────────────────────────────────────

  # Use with any PyTorch model
  from torch.utils.data import DataLoader
  loader = DataLoader(dataset, batch_size=32, shuffle=True)

  for X_batch, y_batch in loader:
      out = model(X_batch)   # X_batch: (32, n_channels, n_times)
      loss = criterion(out, y_batch)

  # EEGNet example (via Braindecode)
  from braindecode.models import EEGNetv4
  model = EEGNetv4(
      n_chans=dataset.n_channels,
      n_classes=dataset.n_classes,
      input_window_samples=dataset.n_times,
  )

──────────────────────────────────────────────────────────────────────
""")