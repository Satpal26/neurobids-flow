# openbci.py
# Plugin for OpenBCI Cyton board
# OpenBCI saves data as .txt (OpenBCI GUI format) or .bdf (BrainFlow format)
# MNE has no native OpenBCI reader — we parse it manually

import mne
import numpy as np
import pandas as pd
from pathlib import Path
from .base import BaseHardwarePlugin, EventInfo, HardwareMetadata


class OpenBCIPlugin(BaseHardwarePlugin):

    def detect(self, filepath: str) -> bool:
        """Detect OpenBCI .txt file by extension and header content."""
        if not filepath.lower().endswith(".txt"):
            return False
        # OpenBCI txt files always start with "%OpenBCI" in first line
        try:
            with open(filepath, "r") as f:
                first_line = f.readline()
            return "%OpenBCI" in first_line
        except Exception:
            return False

    def read_raw(self, filepath: str, **kwargs) -> mne.io.BaseRaw:
        """
        Parse OpenBCI .txt format manually and return MNE RawArray.
        OpenBCI .txt has comment lines starting with % followed by CSV data.
        """
        sfreq = 250.0  # OpenBCI Cyton default sampling rate

        # Read all non-comment lines
        rows = []
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("%") or line == "":
                    continue
                rows.append(line.split(","))

        df = pd.DataFrame(rows)
        df = df.apply(pd.to_numeric, errors="coerce").dropna()

        # OpenBCI Cyton: columns 1-8 are EEG channels (0-indexed)
        eeg_data = df.iloc[:, 1:9].values.T  # shape: (8, n_samples)

        # Convert from raw counts to microvolts
        # OpenBCI scale factor for Cyton = 0.022351744455307625 uV/count
        scale_factor = 0.022351744455307625
        eeg_data = eeg_data * scale_factor * 1e-6  # convert to Volts for MNE

        # Build MNE info
        ch_names = [f"EEG{i+1}" for i in range(8)]
        ch_types = ["eeg"] * 8
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        info.set_montage("standard_1020", on_missing="ignore")

        raw = mne.io.RawArray(eeg_data, info, verbose=False)
        return raw

    def extract_events(self, filepath: str, raw: mne.io.BaseRaw) -> list[EventInfo]:
        """
        Extract events from OpenBCI .txt marker column (column index 12).
        OpenBCI GUI writes event markers in the last column.
        """
        event_list = []
        sfreq = raw.info["sfreq"]
        sample_idx = 0

        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("%") or line == "":
                    continue
                parts = line.split(",")
                if len(parts) > 12:
                    marker = parts[12].strip()
                    if marker and marker != "0":
                        event_list.append(EventInfo(
                            onset=sample_idx / sfreq,
                            duration=0.0,
                            description=marker,
                            trigger_source="software"
                        ))
                sample_idx += 1

        return event_list

    def get_metadata(self, filepath: str) -> HardwareMetadata:
        """Return OpenBCI Cyton metadata for BIDS sidecar."""
        return HardwareMetadata(
            manufacturer="OpenBCI",
            model="Cyton 8-channel",
            sampling_rate=250.0,
            channel_count=8,
            reference_scheme="SRB2",
            power_line_freq=50.0,
            eeg_ground="AGND"
        )
