# converter.py
# EEGConverter — main pipeline orchestrator
# Detects hardware plugin → reads raw → harmonizes events (+ HED) → writes BIDS
# NeuroBIDS-Flow | NTU Singapore BCI Lab

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional

import mne
import mne_bids

from neurobids_flow.core.harmonizer import EventHarmonizer
from neurobids_flow.core.config import load_config
from neurobids_flow.core.dataset_description import (
    DatasetDescription, write_dataset_description
)
from neurobids_flow.plugins.brainproducts import BrainProductsPlugin
from neurobids_flow.plugins.neuroscan import NeuroscanPlugin
from neurobids_flow.plugins.openbci import OpenBCIPlugin
from neurobids_flow.plugins.muse import MusePlugin
from neurobids_flow.plugins.emotiv import EmotivPlugin

logger = logging.getLogger(__name__)

# All registered plugins — order matters for detection priority
PLUGINS = [
    BrainProductsPlugin,
    NeuroscanPlugin,
    EmotivPlugin,   # before Muse — both can be .edf
    MusePlugin,
    OpenBCIPlugin,
]

# Format map for mne_bids.write_raw_bids
FORMAT_MAP = {
    "BrainProductsPlugin": "BrainVision",
    "NeuroscanPlugin":     "BrainVision",
    "EmotivPlugin":        "EDF",
    "MusePlugin":          "BrainVision",
    "OpenBCIPlugin":       "BrainVision",
}


class ConversionResult:
    """Holds the result of a single EEG → BIDS conversion."""

    def __init__(
        self,
        bids_path: mne_bids.BIDSPath,
        plugin_name: str,
        n_events: int,
        hed_injected: bool,
        validation_passed: Optional[bool],
    ):
        self.bids_path = bids_path
        self.plugin_name = plugin_name
        self.n_events = n_events
        self.hed_injected = hed_injected
        self.validation_passed = validation_passed

    def __repr__(self):
        hed_str = "HED injected" if self.hed_injected else "no HED"
        val_str = (
            "validation passed" if self.validation_passed
            else "validation failed" if self.validation_passed is False
            else "validation skipped"
        )
        return (
            f"ConversionResult({self.plugin_name}, "
            f"{self.n_events} events, {hed_str}, {val_str})"
        )


class EEGConverter:
    """
    Orchestrates the full NeuroBIDS-Flow pipeline:

    1.  Load YAML configuration
    2.  Auto-detect hardware plugin
    3.  Read raw EEG data → mne.io.BaseRaw
    4.  Extract raw events → List[EventInfo]
    5.  Harmonize events → List[HarmonizedEvent]  (trial_type + HED)
    6.  Generate dataset_description.json          (HEDVersion if applicable)
    7.  Write BIDS output via mne_bids
    8.  Write events.tsv                           (onset | duration | trial_type | ...)
    9.  Write events.json sidecar                  (HED dictionary — if HED configured)
    10. Validate BIDS output                       (mne_bids.make_report)
    """

    def __init__(self, config_path: str = "configs/default_config.yaml"):
        self.config = load_config(config_path)
        self.harmonizer = EventHarmonizer(
            self.config.event_mapping          # ← FIXED: was self.config.get("event_mapping", {})
        )
        self._plugins = [P() for P in PLUGINS]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert(
        self,
        filepath: str,
        bids_root: str,
        subject: str = "01",
        session: str = "01",
        task: Optional[str] = None,
    ) -> ConversionResult:
        """
        Convert a single EEG file to BIDS-EEG format.

        Parameters
        ----------
        filepath : str
            Path to the source EEG file.
        bids_root : str
            Root directory for BIDS output.
        subject : str
            Subject label (e.g. "01").
        session : str
            Session label (e.g. "01").
        task : str, optional
            Task label. Falls back to config value if not provided.

        Returns
        -------
        ConversionResult
        """
        task = task or self.config.recording.task          # ← FIXED
        logger.info(f"Converting: {filepath} → subject={subject} session={session} task={task}")

        # ── Step 2: Detect plugin ─────────────────────────────────────
        plugin = self._detect_plugin(filepath)
        plugin_name = type(plugin).__name__
        logger.info(f"Plugin detected: {plugin_name}")

        # ── Step 3: Read raw ──────────────────────────────────────────
        raw = plugin.read_raw(filepath)

        # Set power line frequency from config
        power_line_freq = self.config.recording.power_line_freq   # ← FIXED
        raw.info["line_freq"] = power_line_freq

        # ── Step 4: Extract events ────────────────────────────────────
        raw_events = plugin.extract_events(filepath, raw)
        logger.info(f"Extracted {len(raw_events)} raw events")

        # ── Step 5: Harmonize events ──────────────────────────────────
        harmonized = self.harmonizer.harmonize(raw_events)
        logger.info(f"Harmonized {len(harmonized)} events")
        if self.harmonizer.has_hed():
            logger.info("HED strings configured — events.json sidecar will be written")

        # ── Step 6: Dataset description ───────────────────────────────
        description = DatasetDescription(
            name=self.config.dataset.name,                    # ← FIXED
            authors=self.config.dataset.authors,              # ← FIXED
            institution=self.config.dataset.institution,      # ← FIXED
            ethics_approval=self.config.dataset.ethics_approval,  # ← FIXED
        )
        write_dataset_description(
            bids_root=bids_root,
            description=description,
            inject_hed=self.harmonizer.has_hed(),
        )

        # ── Step 7: Write BIDS output via mne_bids ────────────────────
        bids_path = mne_bids.BIDSPath(
            subject=subject,
            session=session,
            task=task,
            datatype="eeg",
            root=bids_root,
        )

        metadata = plugin.get_metadata(filepath)
        bids_format = FORMAT_MAP.get(plugin_name, "BrainVision")

        mne_bids.write_raw_bids(
            raw,
            bids_path=bids_path,
            allow_preload=True,
            format=bids_format,
            overwrite=self.config.output.overwrite,           # ← FIXED
        )
        logger.info(f"BIDS output written: {bids_path.fpath}")

        # ── Step 8: Write events.tsv ──────────────────────────────────
        eeg_dir = str(bids_path.fpath.parent)
        basename = bids_path.basename
        events_tsv_path = os.path.join(eeg_dir, f"{basename}_events.tsv")
        self.harmonizer.write_events_tsv(harmonized, events_tsv_path)
        logger.info(f"events.tsv written: {events_tsv_path}")

        # ── Step 9: Write events.json (HED sidecar) ───────────────────
        events_json_path = os.path.join(eeg_dir, f"{basename}_events.json")
        hed_injected = self.harmonizer.write_events_json(events_json_path)
        if hed_injected:
            logger.info(f"events.json (HED sidecar) written: {events_json_path}")
        else:
            logger.info("No HED strings in config — events.json sidecar skipped")

        # ── Step 10: Validate ─────────────────────────────────────────
        validation_passed = None
        if self.config.output.validate_bids:                  # ← FIXED
            validation_passed = self._validate(bids_root)

        return ConversionResult(
            bids_path=bids_path,
            plugin_name=plugin_name,
            n_events=len(harmonized),
            hed_injected=hed_injected,
            validation_passed=validation_passed,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_plugin(self, filepath: str):
        for plugin in self._plugins:
            if plugin.detect(filepath):
                return plugin
        raise ValueError(
            f"No plugin found for file: {filepath}\n"
            f"Supported formats: .vhdr, .cnt, .edf, .txt, .csv, .xdf"
        )

    def _validate(self, bids_root: str) -> bool:
        try:
            report = mne_bids.make_report(bids_root)
            logger.info(f"BIDS validation passed:\n{report}")
            return True
        except Exception as e:
            logger.warning(f"BIDS validation issue: {e}")
            return False