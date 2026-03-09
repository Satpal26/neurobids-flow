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


class EEGConverter:
    """
    Main converter class.
    1. Auto-detects hardware from file
    2. Loads correct plugin
    3. Reads raw EEG
    4. Extracts events
    5. Writes BIDS output
    """

    def __init__(self):
        # All 5 hardware plugins registered
        self.plugins = [
            BrainProductsPlugin(),
            NeuroscanPlugin(),
            OpenBCIPlugin(),
            MusePlugin(),
            EmotivPlugin(),
        ]

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
        task: str,
    ):
        """
        Full conversion pipeline.
        filepath  — path to raw EEG file
        bids_root — where to write the BIDS dataset
        subject   — e.g. "01"
        session   — e.g. "01"
        task      — e.g. "ssvep"
        """

        print(f"[eeg2bids] Processing: {filepath}")

        # Step 1 — find the right plugin
        plugin = self.detect_plugin(filepath)
        print(f"[eeg2bids] Detected hardware: {plugin.__class__.__name__}")

        # Step 2 — read raw EEG
        raw = plugin.read_raw(filepath)
        print(f"[eeg2bids] Loaded {len(raw.ch_names)} channels @ {raw.info['sfreq']} Hz")

        # Step 3 — extract events
        events = plugin.extract_events(filepath, raw)
        print(f"[eeg2bids] Found {len(events)} events")

        # Step 4 — get hardware metadata
        metadata = plugin.get_metadata(filepath)

        # Step 5 — build BIDS path
        bids_path = mne_bids.BIDSPath(
            subject=subject,
            session=session,
            task=task,
            root=bids_root,
            datatype="eeg",
        )

        # Step 6 — write BIDS
        mne_bids.write_raw_bids(
            raw=raw,
            bids_path=bids_path,
            overwrite=True,
            verbose=False,
        )

        print(f"[eeg2bids] BIDS output written to: {bids_root}")
        return bids_path