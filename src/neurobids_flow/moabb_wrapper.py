# moabb_wrapper.py
# NeuroBIDS-Flow — MOABB Dataset Wrapper
# ─────────────────────────────────────────────────────────────────────────────
# Bridges BIDS-EEG output from NeuroBIDS-Flow into the MOABB benchmarking
# ecosystem. Compatible with PyTorch (Braindecode / TorchEEG), TensorFlow
# (Keras), and Scikit-learn — without duplicating or reformatting the data.
#
# Place this file at:
#   src/neurobids_flow/moabb_wrapper.py
#
# Quick usage:
#   from neurobids_flow.moabb_wrapper import NeuroBIDSFlowDataset
#   dataset = NeuroBIDSFlowDataset(bids_root="./bids_output")
#   X, y, meta = paradigm.get_data(dataset=dataset, subjects=[1, 2])
#   print(X.shape)   # (n_trials, n_channels, n_times)
#
# NeuroBIDS-Flow | NTU Singapore BCI Lab
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from mne_bids import BIDSPath, read_raw_bids
from moabb.datasets.base import BaseDataset

logger = logging.getLogger(__name__)

# Valid EEG file extensions accepted by mne_bids.read_raw_bids
EEG_EXTENSIONS = {".vhdr", ".edf", ".bdf", ".set", ".fif"}

# ─────────────────────────────────────────────────────────────────────────────
# Default event → integer mapping
# These trial_type values are written by NeuroBIDS-Flow's EventHarmonizer
# into events.tsv during BIDS conversion. Keys must match exactly.
# ─────────────────────────────────────────────────────────────────────────────

NEUROBIDS_EVENTS: dict[str, int] = {
    # ── Resting state ──────────────────────────────────────────────
    "rest_open":        1,
    "rest_closed":      2,
    "rest":             3,

    # ── Cognitive workload (core passive BCI target) ───────────────
    "cognitive_low":    4,
    "cognitive_high":   5,
    "fatigue":          6,
    "alert":            7,

    # ── Emotion ────────────────────────────────────────────────────
    "emotion_positive": 8,
    "emotion_negative": 9,
    "arousal_high":     10,
    "arousal_low":      11,
}


# ─────────────────────────────────────────────────────────────────────────────
# NeuroBIDSFlowDataset
# ─────────────────────────────────────────────────────────────────────────────

class NBIDSFDataset(BaseDataset):
    """
    MOABB-compatible wrapper for BIDS-EEG datasets produced by NeuroBIDS-Flow.

    NeuroBIDS-Flow converts raw EEG from consumer/research devices
    (BrainProducts, Emotiv EPOC+, Muse 2, OpenBCI Cyton) into BIDS-EEG + HED.
    This class wraps that output so it can be immediately used with:

      - MOABB paradigms  → benchmarking against EEGNet, ShallowFBCSP, etc.
      - Braindecode      → deep learning with PyTorch backend
      - TorchEEG         → PyTorch Dataset / DataLoader
      - TensorFlow/Keras → tf.data.Dataset
      - Scikit-learn     → SVM, LDA, Riemannian geometry classifiers

    The data stays in BIDS format on disk. This wrapper reads and structures
    it on-the-fly. No data duplication or reformatting needed.

    Parameters
    ----------
    bids_root : str
        Path to the BIDS root directory written by NeuroBIDS-Flow.
        Expected structure:
            bids_root/
              sub-01/ses-01/eeg/sub-01_ses-01_task-workload_eeg.vhdr
              sub-01/ses-01/eeg/sub-01_ses-01_task-workload_events.tsv
              sub-01/ses-01/eeg/sub-01_ses-01_task-workload_events.json
              dataset_description.json
    task : str
        BIDS task label used during NeuroBIDS-Flow conversion.
        Must exactly match the --task argument passed to the CLI.
        Default: "workload"
    subjects : list[int], optional
        Subject IDs as integers (e.g. [1, 2, 3] maps to sub-01, sub-02, sub-03).
        If None, auto-detected from sub-XX folders in bids_root.
    sessions : list[str], optional
        Session labels (e.g. ["01", "02"]).
        If None, auto-detected from ses-XX folders inside sub-01.
    events : dict[str, int], optional
        Custom event label → integer mapping.
        Keys must match trial_type values in events.tsv exactly.
        Defaults to NEUROBIDS_EVENTS (see top of file).
    interval : list[float]
        Epoch window [tmin, tmax] in seconds relative to event onset.
        Default: [-0.5, 3.0] — suitable for passive BCI / cognitive workload.
    doi : str
        DOI for the published dataset (e.g. Zenodo). Fill in when published.

    Examples
    --------
    Basic MOABB benchmarking:

        from neurobids_flow.moabb_wrapper import NBIDSFDataset
        dataset = NBIDSFDataset(bids_root="./bids_output")
        X, y, meta = paradigm.get_data(dataset=dataset, subjects=[1, 2])
        print(X.shape)   # (n_trials, n_channels, n_times)

    Cognitive workload only (binary classification):

        dataset = NBIDSFDataset(
            bids_root="./bids_output",
            events={"cognitive_low": 0, "cognitive_high": 1},
            interval=[0.0, 4.0],
        )

    With Braindecode (PyTorch):

        from braindecode.datasets.moabb import MOABBDataset
        bd = MOABBDataset(dataset_name="NBIDSF", subject_ids=[1, 2])
    """

    def __init__(
        self,
        bids_root: str = "./bids_output",
        task: str = "workload",
        subjects: Optional[list[int]] = None,
        sessions: Optional[list[str]] = None,
        events: Optional[dict[str, int]] = None,
        interval: Optional[list[float]] = None,
        doi: str = "",
    ):
        self.bids_root = Path(bids_root).resolve()
        self.task = task

        # ── Auto-detect subjects from BIDS root ───────────────────────
        if subjects is None:
            subjects = self._detect_subjects()
        if not subjects:
            raise ValueError(
                f"No subjects found under: {self.bids_root}\n"
                f"Expected folders named sub-01, sub-02, ...\n"
                f"Run NeuroBIDS-Flow conversion first:\n"
                f"  neurobids-flow convert --file <file> "
                f"--bids-root {bids_root} --subject 01 --session 01 "
                f"--task {task}"
            )

        # ── Auto-detect sessions ──────────────────────────────────────
        self._sessions: list[str] = sessions or self._detect_sessions()
        n_sessions = max(len(self._sessions), 1)

        # ── Event map and epoch window ────────────────────────────────
        _events   = events   or NEUROBIDS_EVENTS
        _interval = interval or [-0.5, 3.0]

        super().__init__(
            subjects=subjects,
            sessions_per_subject=n_sessions,
            events=_events,
            code="NBIDSF",          # ← MOABB requires class name to abbreviate code
            interval=_interval,
            paradigm="resting",     # passive BCI — no active motor / SSVEP task
            doi=doi,
        )

        logger.info(
            "NBIDSFDataset ready | "
            f"bids_root={self.bids_root} | task={self.task} | "
            f"n_subjects={len(subjects)} | sessions={self._sessions} | "
            f"n_events={len(_events)} | interval={_interval}"
        )

    # ─────────────────────────────────────────────────────────────────
    # MOABB required method 1 — data_path
    # ─────────────────────────────────────────────────────────────────

    def data_path(
        self,
        subject: int,
        path=None,
        force_update: bool = False,
        update_path=None,
        verbose=None,
    ) -> list[str]:
        """
        Return the BIDS subject directory path for a given subject ID.

        MOABB calls this internally to locate files. Since NeuroBIDS-Flow
        writes data locally (no remote download), we simply resolve
        the sub-XX directory and verify it exists.

        Parameters
        ----------
        subject : int
            Integer subject ID. 1 → sub-01, 12 → sub-12.

        Returns
        -------
        list[str]
            Single-element list with the resolved subject folder path.
        """
        subject_id = f"{subject:02d}"
        subj_dir = self.bids_root / f"sub-{subject_id}"

        if not subj_dir.exists():
            raise FileNotFoundError(
                f"BIDS subject directory not found: {subj_dir}\n"
                f"Make sure NeuroBIDS-Flow converted data for sub-{subject_id}.\n"
                f"Command: neurobids-flow convert --file <file> "
                f"--bids-root {self.bids_root} "
                f"--subject {subject_id} --task {self.task}"
            )

        logger.debug(f"data_path(sub-{subject_id}) → {subj_dir}")
        return [str(subj_dir)]

    # ─────────────────────────────────────────────────────────────────
    # MOABB required method 2 — _get_single_subject_data
    # ─────────────────────────────────────────────────────────────────

    def _get_single_subject_data(self, subject: int) -> dict:
        """
        Load all BIDS EEG files for one subject into MNE Raw objects.

        MOABB calls this automatically when paradigm.get_data() is invoked.
        Returns a nested dict:
            { session_name: { run_name: mne.io.BaseRaw } }

        How it works with NeuroBIDS-Flow output:
          - read_raw_bids reads the EEG file (.vhdr / .edf / .fif)
          - Events from events.tsv (onset, trial_type) are automatically
            attached to raw.annotations by read_raw_bids
          - HED strings from events.json are also accessible via annotations
          - MOABB's paradigm slices the continuous Raw into epochs using
            the interval set in __init__ — no manual epoching needed

        Parameters
        ----------
        subject : int
            Integer subject ID.

        Returns
        -------
        dict
            { "session_01": { "run_0": mne.io.BaseRaw, ... }, ... }
        """
        subject_id = f"{subject:02d}"
        result: dict[str, dict] = {}

        sessions = self._sessions if self._sessions else ["01"]

        for session in sessions:
            session_key = f"session_{session}"
            result[session_key] = {}

            # Build BIDS path template for this subject + session + task
            bids_path = BIDSPath(
                subject=subject_id,
                session=session,
                task=self.task,
                datatype="eeg",
                root=str(self.bids_root),
            )

            # Find all matching files — handles multiple runs automatically
            try:
                matched_paths = bids_path.match()
            except Exception as exc:
                logger.warning(
                    f"BIDS match failed for sub-{subject_id} "
                    f"ses-{session}: {exc} — skipping."
                )
                continue

            if not matched_paths:
                logger.warning(
                    f"No BIDS files: sub-{subject_id} | "
                    f"ses-{session} | task-{self.task}. Skipping."
                )
                continue

            # ── Filter to EEG files only (.vhdr, .edf, .bdf, .set, .fif)
            # bids_path.match() also returns .tsv/.json sidecars — skip those
            matched_paths = [
                p for p in matched_paths
                if p.fpath.suffix in EEG_EXTENSIONS
            ]

            if not matched_paths:
                logger.warning(
                    f"No EEG files (after filtering) for sub-{subject_id} "
                    f"ses-{session}. Skipping."
                )
                continue

            for run_idx, matched_path in enumerate(matched_paths):
                # read_raw_bids:
                #   reads EEG + attaches events.tsv annotations automatically
                #   HED strings from events.json accessible via raw.annotations
                raw = read_raw_bids(bids_path=matched_path, verbose=False)
                raw.load_data()

                run_key = f"run_{run_idx}"
                result[session_key][run_key] = raw

                logger.info(
                    f"Loaded | sub-{subject_id} | ses-{session} | {run_key} | "
                    f"ch={len(raw.ch_names)} | "
                    f"dur={raw.times[-1]:.1f}s | "
                    f"sfreq={raw.info['sfreq']:.0f}Hz | "
                    f"annotations={len(raw.annotations)}"
                )

        # Guard — raise clearly if nothing loaded
        loaded_runs = sum(len(v) for v in result.values())
        if loaded_runs == 0:
            raise RuntimeError(
                f"No EEG runs loaded for sub-{subject_id}.\n"
                f"Check NeuroBIDS-Flow conversion completed for this subject."
            )

        return result

    # ─────────────────────────────────────────────────────────────────
    # Private helpers — BIDS structure auto-detection
    # ─────────────────────────────────────────────────────────────────

    def _detect_subjects(self) -> list[int]:
        """
        Scan bids_root for sub-XX folders and return IDs as integers.
        sub-01, sub-02 → [1, 2]
        """
        if not self.bids_root.exists():
            return []

        subjects = []
        for folder in sorted(self.bids_root.iterdir()):
            if folder.is_dir():
                match = re.fullmatch(r"sub-(\d+)", folder.name)
                if match:
                    subjects.append(int(match.group(1)))

        if subjects:
            logger.info(f"Auto-detected subjects: {subjects}")
        else:
            logger.warning(
                f"No sub-XX folders in {self.bids_root}. "
                f"Pass subjects= explicitly."
            )
        return subjects

    def _detect_sessions(self) -> list[str]:
        """
        Scan the first subject folder for ses-XX directories.
        Returns zero-padded session labels e.g. ["01", "02"].
        Falls back to ["01"] if no session folders found.
        """
        if not self.bids_root.exists():
            return ["01"]

        for folder in sorted(self.bids_root.iterdir()):
            if folder.is_dir() and re.fullmatch(r"sub-\d+", folder.name):
                sessions = []
                for sf in sorted(folder.iterdir()):
                    if sf.is_dir():
                        match = re.fullmatch(r"ses-(\d+)", sf.name)
                        if match:
                            sessions.append(match.group(1))
                if sessions:
                    logger.info(f"Auto-detected sessions: {sessions}")
                    return sessions
                break

        logger.info("No ses-XX folders found — defaulting to ['01'].")
        return ["01"]


# ─────────────────────────────────────────────────────────────────────────────
# Quick-start demo  (python moabb_wrapper.py --bids-root ./bids_output)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="NeuroBIDSFlow MOABB wrapper — quick-start demo"
    )
    parser.add_argument(
        "--bids-root", default="./bids_output",
        help="Path to BIDS root produced by NeuroBIDS-Flow"
    )
    parser.add_argument(
        "--task", default="workload",
        help="BIDS task label (must match --task used during conversion)"
    )
    parser.add_argument(
        "--subjects", nargs="+", type=int, default=None,
        help="Subject IDs to load e.g. --subjects 1 2 3"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("\n" + "─" * 60)
    print("  NeuroBIDSFlow MOABB Wrapper — Quick-start Demo")
    print("─" * 60)

    dataset = NBIDSFDataset(
        bids_root=args.bids_root,
        task=args.task,
        subjects=args.subjects,
    )

    print(f"\n✓ Dataset initialised")
    print(f"  Subjects  : {dataset.subject_list}")
    print(f"  Sessions  : {dataset._sessions}")
    print(f"  Events    : {list(dataset.event_id.keys())}")
    print(f"  Interval  : {dataset.interval} s")

    print(f"\nLoading raw data for subject {dataset.subject_list[0]} ...")
    subject_data = dataset._get_single_subject_data(dataset.subject_list[0])
    for session_key, runs in subject_data.items():
        for run_key, raw in runs.items():
            print(
                f"  {session_key}/{run_key} | "
                f"ch={len(raw.ch_names)} | "
                f"dur={raw.times[-1]:.1f}s | "
                f"sfreq={raw.info['sfreq']:.0f}Hz | "
                f"annotations={len(raw.annotations)}"
            )

    print("""
─── Next steps ──────────────────────────────────────────────────────

  # MOABB benchmarking
  from moabb.paradigms import <YourParadigm>
  paradigm = <YourParadigm>(events=["cognitive_low", "cognitive_high"])
  X, y, meta = paradigm.get_data(dataset=dataset, subjects=[1, 2])
  print(X.shape)   # (n_trials, n_channels, n_times)

  # Braindecode (PyTorch backend)
  from braindecode.datasets.moabb import MOABBDataset
  bd = MOABBDataset(dataset_name="NBIDSF", subject_ids=[1])

  # Scikit-learn pipeline
  from moabb.evaluations import CrossSubjectEvaluation
  from sklearn.pipeline import make_pipeline
  from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
  pipe = make_pipeline(LinearDiscriminantAnalysis())
  evaluation = CrossSubjectEvaluation(paradigm=paradigm, datasets=[dataset])
  results = evaluation.process({"LDA": pipe})

──────────────────────────────────────────────────────────────────────
""")