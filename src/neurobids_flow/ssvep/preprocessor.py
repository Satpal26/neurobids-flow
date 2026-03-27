"""
NeuroBIDS-Flow — SSVEP Preprocessor
=====================================
Loads BIDS-EEG data and applies standard SSVEP preprocessing:
  1. Load raw EEG from BIDS output (Target 1)
  2. Bandpass filter (1-40 Hz)
  3. Notch filter (power line)
  4. Re-reference to common average
  5. Epoch around stimulus events
  6. Baseline correction
  7. Return clean epochs ready for SSVEP classification

Usage:
    from neurobids_flow.ssvep.preprocessor import SSVEPPreprocessor
    preprocessor = SSVEPPreprocessor(bids_root="./bids_output")
    epochs, labels = preprocessor.get_epochs(subject="01", session="01")
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import mne
import numpy as np

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


# ── Default SSVEP preprocessing parameters ───────────────────────────────────

DEFAULTS = {
    "l_freq":          1.0,    # Hz — highpass
    "h_freq":          40.0,   # Hz — lowpass
    "notch_freq":      50.0,   # Hz — power line (50 for Asia/Europe, 60 for USA)
    "tmin":           -0.5,    # s  — epoch start relative to stimulus onset
    "tmax":            4.0,    # s  — epoch end relative to stimulus onset
    "baseline":       (-0.5, 0.0),  # baseline correction window
    "reject_peak_uv":  150.0,  # μV — peak-to-peak rejection threshold
    "reference":       "average",   # re-reference scheme
}

# Standard SSVEP stimulus frequencies (Hz)
SSVEP_FREQS = [6, 7, 8, 10, 12, 15, 20]


class SSVEPPreprocessor:
    """
    Loads BIDS-EEG output from NeuroBIDS-Flow (Target 1) and
    applies standard SSVEP preprocessing to produce clean epochs.
    """

    def __init__(
        self,
        bids_root: str,
        task: str = "ssvep",
        l_freq: float = DEFAULTS["l_freq"],
        h_freq: float = DEFAULTS["h_freq"],
        notch_freq: float = DEFAULTS["notch_freq"],
        tmin: float = DEFAULTS["tmin"],
        tmax: float = DEFAULTS["tmax"],
        baseline: tuple = DEFAULTS["baseline"],
        reject_peak_uv: float = DEFAULTS["reject_peak_uv"],
        reference: str = DEFAULTS["reference"],
    ):
        self.bids_root      = Path(bids_root)
        self.task           = task
        self.l_freq         = l_freq
        self.h_freq         = h_freq
        self.notch_freq     = notch_freq
        self.tmin           = tmin
        self.tmax           = tmax
        self.baseline       = baseline
        self.reject_peak_uv = reject_peak_uv
        self.reference      = reference

        log.info(f"SSVEPPreprocessor | bids_root={bids_root} | task={task}")
        log.info(f"  Filter: [{l_freq}, {h_freq}] Hz | Notch: {notch_freq} Hz")
        log.info(f"  Epoch: [{tmin}, {tmax}] s | Baseline: {baseline}")

    # ── Public API ────────────────────────────────────────────────────────────

    def get_epochs(
        self,
        subject: str,
        session: str = "01",
        verbose: bool = False,
    ) -> tuple[mne.Epochs, np.ndarray]:
        """
        Full preprocessing pipeline for one subject.

        Returns
        -------
        epochs : mne.Epochs
            Clean, preprocessed SSVEP epochs.
        labels : np.ndarray
            Integer label per epoch (stimulus frequency index).
        """
        raw = self._load_raw(subject, session, verbose)
        raw = self._preprocess_raw(raw)
        epochs, labels = self._epoch(raw, subject)
        return epochs, labels

    def get_epochs_data(
        self,
        subject: str,
        session: str = "01",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convenience wrapper — returns numpy arrays directly.

        Returns
        -------
        X : np.ndarray, shape (n_epochs, n_channels, n_times)
        y : np.ndarray, shape (n_epochs,)
        """
        epochs, labels = self.get_epochs(subject, session)
        X = epochs.get_data()
        return X, labels

    def get_all_subjects(
        self,
        session: str = "01",
    ) -> tuple[np.ndarray, np.ndarray, list]:
        """
        Load and preprocess all subjects in the BIDS root.

        Returns
        -------
        X : np.ndarray, shape (total_epochs, n_channels, n_times)
        y : np.ndarray, shape (total_epochs,)
        subjects : list of subject IDs per epoch
        """
        subject_dirs = sorted([
            d.name.replace("sub-", "")
            for d in self.bids_root.iterdir()
            if d.is_dir() and d.name.startswith("sub-")
        ])

        all_X, all_y, all_subs = [], [], []

        for subject in subject_dirs:
            try:
                X, y = self.get_epochs_data(subject, session)
                all_X.append(X)
                all_y.append(y)
                all_subs.extend([subject] * len(y))
                log.info(f"  sub-{subject} | epochs={len(y)} | shape={X.shape}")
            except Exception as e:
                log.warning(f"  sub-{subject} | Failed: {e} — skipping")

        if not all_X:
            raise RuntimeError("No epochs loaded from any subject.")

        # Trim to minimum channel count (cross-device compatibility)
        min_ch = min(x.shape[1] for x in all_X)
        if len(set(x.shape[1] for x in all_X)) > 1:
            log.info(f"  Trimming to {min_ch} channels (cross-device)")
            all_X = [x[:, :min_ch, :] for x in all_X]

        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)
        return X, y, all_subs

    # ── Internal steps ────────────────────────────────────────────────────────

    def _load_raw(
        self,
        subject: str,
        session: str,
        verbose: bool = False,
    ) -> mne.io.BaseRaw:
        """Load raw EEG from BIDS root using MNE-BIDS."""
        from mne_bids import BIDSPath, read_raw_bids

        bids_path = BIDSPath(
            subject=subject,
            session=session,
            task=self.task,
            root=str(self.bids_root),
            datatype="eeg",
        )
        raw = read_raw_bids(bids_path=bids_path, verbose=verbose)
        raw.load_data(verbose=verbose)
        log.info(
            f"  Loaded sub-{subject} | "
            f"ch={len(raw.ch_names)} | "
            f"sfreq={raw.info['sfreq']} Hz | "
            f"dur={raw.times[-1]:.1f}s"
        )
        return raw

    def _preprocess_raw(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Apply bandpass, notch, and re-reference."""

        # 1. Bandpass filter
        raw.filter(
            l_freq=self.l_freq,
            h_freq=self.h_freq,
            method="fir",
            fir_design="firwin",
            verbose=False,
        )
        log.info(f"  Filtered: [{self.l_freq}, {self.h_freq}] Hz")

        # 2. Notch filter
        raw.notch_filter(
            freqs=self.notch_freq,
            verbose=False,
        )
        log.info(f"  Notch: {self.notch_freq} Hz")

        # 3. Re-reference
        if self.reference == "average":
            raw.set_eeg_reference("average", projection=False, verbose=False)
            log.info("  Re-referenced: average")
        elif self.reference != "none":
            raw.set_eeg_reference([self.reference], verbose=False)
            log.info(f"  Re-referenced: {self.reference}")

        return raw

    def _epoch(
        self,
        raw: mne.io.BaseRaw,
        subject: str,
    ) -> tuple[mne.Epochs, np.ndarray]:
        """Extract SSVEP epochs from stimulus annotations."""

        if len(raw.annotations) == 0:
            raise ValueError(f"sub-{subject}: No annotations found in raw data.")

        # Build event_id from annotations
        unique_descs = list(set(raw.annotations.description))
        event_id = {desc: i + 1 for i, desc in enumerate(unique_descs)}

        events, _ = mne.events_from_annotations(
            raw, event_id=event_id, verbose=False
        )

        if len(events) == 0:
            raise ValueError(f"sub-{subject}: No events extracted.")

        # Rejection threshold
        reject = {"eeg": self.reject_peak_uv * 1e-6}

        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=self.tmin,
            tmax=self.tmax,
            baseline=self.baseline,
            reject=reject,
            preload=True,
            verbose=False,
        )

        n_dropped = len(events) - len(epochs)
        log.info(
            f"  Epochs: {len(epochs)} kept, {n_dropped} dropped | "
            f"shape={epochs.get_data().shape}"
        )

        labels = epochs.events[:, 2]
        return epochs, labels


# ── Utility functions ─────────────────────────────────────────────────────────

def load_ssvep_data(
    bids_root: str,
    subject: str,
    session: str = "01",
    task: str = "ssvep",
    tmin: float = 0.0,
    tmax: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Quick one-line loader for SSVEP epochs.

    Returns X (n_epochs, n_channels, n_times) and y (n_epochs,).
    """
    preprocessor = SSVEPPreprocessor(
        bids_root=bids_root,
        task=task,
        tmin=tmin,
        tmax=tmax,
    )
    return preprocessor.get_epochs_data(subject=subject, session=session)