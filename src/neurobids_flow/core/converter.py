# converter.py
# This is the brain of the tool.
# It takes any EEG file → finds the right plugin → converts to BIDS

import mne
import mne_bids
import pandas as pd
from pathlib import Path
from ..plugins.base import EventInfo
from ..plugins.brainproducts import BrainProductsPlugin
from ..plugins.neuroscan import NeuroscanPlugin
from ..plugins.openbci import OpenBCIPlugin
from ..plugins.muse import MusePlugin
from ..plugins.emotiv import EmotivPlugin
from .harmonizer import EventHarmonizer
from .config import load_config, AppConfig
from .validator import validate_bids
from .dataset_description import generate_dataset_description


class EEGConverter:
    """
    Main converter class.
    1. Loads config (YAML or defaults)
    2. Auto-detects hardware from file
    3. Loads correct plugin
    4. Reads raw EEG
    5. Extracts events
    6. Harmonizes events into unified format
    7. Generates dataset_description.json
    8. Writes BIDS output
    9. Validates BIDS output
    """

    def __init__(self, config_path: str = None):
        # Load config
        self.config = load_config(config_path)

        # All 5 hardware plugins registered
        self.plugins = [
            BrainProductsPlugin(),
            NeuroscanPlugin(),
            OpenBCIPlugin(),
            MusePlugin(),
            EmotivPlugin(),
        ]

        # EventHarmonizer uses event mapping from config
        self.harmonizer = EventHarmonizer(
            custom_mapping=self.config.event_mapping
        )

    def detect_plugin(self, filepath: str):
        """Find which plugin can handle this file."""
        for plugin in self.plugins:
            if plugin.detect(filepath):
                return plugin
        raise ValueError(f"No plugin found for file: {filepath}")

    def convert(
        self,
        filepath: str,
        bids_root: str,
        subject: str,
        session: str,
        task: str = None,
    ):
        """
        Full conversion pipeline.
        filepath  — path to raw EEG file
        bids_root — where to write the BIDS dataset
        subject   — e.g. "01"
        session   — e.g. "01"
        task      — e.g. "ssvep" (falls back to config if not given)
        """

        # Use task from config if not explicitly passed
        if task is None:
            task = self.config.recording.task

        print(f"[eeg2bids] Processing: {filepath}")
        print(f"[eeg2bids] Subject: {subject} | Session: {session} | Task: {task}")

        # Step 1 — find the right plugin
        plugin = self.detect_plugin(filepath)
        print(f"[eeg2bids] Detected hardware: {plugin.__class__.__name__}")

        # Step 2 — read raw EEG
        raw = plugin.read_raw(filepath)
        print(f"[eeg2bids] Loaded {len(raw.ch_names)} channels @ {raw.info['sfreq']} Hz")

        # Step 3 — extract raw events
        raw_events = plugin.extract_events(filepath, raw)
        print(f"[eeg2bids] Found {len(raw_events)} raw events")

        # Step 4 — harmonize events into unified format
        harmonized = self.harmonizer.harmonize(raw_events)
        print(f"[eeg2bids] Harmonized {len(harmonized)} events")

        # Step 5 — get hardware metadata
        metadata = plugin.get_metadata(filepath)

        # Step 6 — generate dataset_description.json
        generate_dataset_description(bids_root, self.config)

        # Step 7 — build BIDS path
        bids_path = mne_bids.BIDSPath(
            subject=subject,
            session=session,
            task=task,
            root=bids_root,
            datatype="eeg",
        )

# Step 8 — write BIDS
        fmt_map = {
            'BrainProductsPlugin': 'BrainVision',
            'EmotivPlugin':        'EDF',
            'NeuroscanPlugin':     'BrainVision',
            'OpenBCIPlugin':       'BrainVision',
            'MusePlugin':          'BrainVision',
        }
        bids_format = fmt_map.get(type(plugin).__name__, 'BrainVision')

        mne_bids.write_raw_bids(
            raw,
            bids_path=bids_path,
            allow_preload=True,
            format=bids_format,
            overwrite=True,
        )

        # Step 9 — write harmonized events.tsv
        events_path = (
            Path(bids_root)
            / f"sub-{subject}"
            / f"ses-{session}"
            / "eeg"
            / f"sub-{subject}_ses-{session}_task-{task}_events.tsv"
        )
        events_path.parent.mkdir(parents=True, exist_ok=True)
        self.harmonizer.to_bids_tsv(harmonized, str(events_path))

        # Step 10 — validate BIDS output if enabled in config
        if self.config.output.validate_bids:
            validate_bids(bids_root)

        print(f"[eeg2bids] Done! BIDS output written to: {bids_root}")
        return bids_path
