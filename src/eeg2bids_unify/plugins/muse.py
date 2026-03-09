# muse.py
# Plugin for InteraXon Muse 2 headband
# Muse records via Mind Monitor app → saves as .csv
# or via MuseLSL → saves as .xdf (Lab Streaming Layer format)
# We support both formats

import mne
import numpy as np
import pandas as pd
from .base import BaseHardwarePlugin, EventInfo, HardwareMetadata


class MusePlugin(BaseHardwarePlugin):

    def detect(self, filepath: str) -> bool:
        """
        Detect Muse files.
        Muse .csv files from Mind Monitor always have 'TimeStamp' and 'RAW_TP9' columns.
        Muse .xdf files are detected by extension.
        """
        if filepath.lower().endswith(".xdf"):
            return True
        if filepath.lower().endswith(".csv"):
            try:
                df = pd.read_csv(filepath, nrows=1)
                return "RAW_TP9" in df.columns or "RAW_AF7" in df.columns
            except Exception:
                return False
        return False

    def read_raw(self, filepath: str, **kwargs) -> mne.io.BaseRaw:
        """
        Read Muse file and return MNE RawArray.
        Muse 2 has 4 EEG channels: TP9, AF7, AF8, TP10
        Sampling rate: 256 Hz
        """
        if filepath.lower().endswith(".xdf"):
            return self._read_xdf(filepath)
        else:
            return self._read_csv(filepath)

    def _read_csv(self, filepath: str) -> mne.io.BaseRaw:
        """Parse Mind Monitor CSV format."""
        sfreq = 256.0
        df = pd.read_csv(filepath)

        # Muse 2 EEG channel columns in Mind Monitor CSV
        eeg_cols = ["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]
        available = [c for c in eeg_cols if c in df.columns]
        eeg_data = df[available].dropna().values.T  # shape: (4, n_samples)

        # Convert from microvolts to Volts for MNE
        eeg_data = eeg_data * 1e-6

        ch_names = ["TP9", "AF7", "AF8", "TP10"][:len(available)]
        ch_types = ["eeg"] * len(available)
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

        raw = mne.io.RawArray(eeg_data, info, verbose=False)
        return raw

    def _read_xdf(self, filepath: str) -> mne.io.BaseRaw:
        """Parse XDF format from MuseLSL using pyxdf."""
        import pyxdf
        streams, _ = pyxdf.load_xdf(filepath)

        # Find EEG stream
        eeg_stream = None
        for stream in streams:
            if stream["info"]["type"][0].lower() == "eeg":
                eeg_stream = stream
                break

        if eeg_stream is None:
            raise ValueError("No EEG stream found in XDF file")

        eeg_data = eeg_stream["time_series"].T  # shape: (n_channels, n_samples)
        sfreq = float(eeg_stream["info"]["nominal_srate"][0])

        n_channels = eeg_data.shape[0]
        ch_names = [f"EEG{i+1}" for i in range(n_channels)]
        ch_types = ["eeg"] * n_channels
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

        # Convert to Volts
        eeg_data = eeg_data * 1e-6

        raw = mne.io.RawArray(eeg_data, info, verbose=False)
        return raw

    def extract_events(self, filepath: str, raw: mne.io.BaseRaw) -> list[EventInfo]:
        """
        Extract events from Muse CSV marker column if present.
        Mind Monitor writes markers in 'Marker' column.
        """
        event_list = []

        if filepath.lower().endswith(".csv"):
            try:
                df = pd.read_csv(filepath)
                sfreq = raw.info["sfreq"]
                if "Marker" in df.columns:
                    markers = df["Marker"].dropna()
                    for idx, marker in markers.items():
                        if str(marker).strip() not in ["", "0"]:
                            event_list.append(EventInfo(
                                onset=idx / sfreq,
                                duration=0.0,
                                description=str(marker),
                                trigger_source="software"
                            ))
            except Exception:
                pass

        return event_list

    def get_metadata(self, filepath: str) -> HardwareMetadata:
        """Return Muse 2 metadata for BIDS sidecar."""
        return HardwareMetadata(
            manufacturer="InteraXon",
            model="Muse 2",
            sampling_rate=256.0,
            channel_count=4,
            reference_scheme="unknown",
            power_line_freq=50.0,
            eeg_ground="unknown"
        )