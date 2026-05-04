# emotiv.py
# Plugin for Emotiv EPOC / EPOC+ headset
# Supports .edf (EmotivPRO) and .csv (FEIS dataset / Emotiv Xavier format)
import mne
import numpy as np
import pandas as pd
from pathlib import Path
from .base import BaseHardwarePlugin, EventInfo, HardwareMetadata

EMOTIV_CHANNELS = ["F3", "FC5", "AF3", "F7", "T7", "P7",
                   "O1", "O2", "P8", "T8", "F8", "AF4", "FC6", "F4"]

EMOTIV_CHANNEL_SET = set(EMOTIV_CHANNELS)


def _is_emotiv_csv(filepath: str) -> bool:
    """
    Detect FEIS-style Emotiv CSV.
    Header must contain at least 6 Emotiv channel names and a Label column.
    """
    try:
        with open(filepath, "r") as f:
            header = f.readline().strip()
        cols = [c.strip() for c in header.split(",")]
        col_set = set(cols)
        emotiv_matches = len(EMOTIV_CHANNEL_SET & col_set)
        return emotiv_matches >= 6 and "Label" in col_set
    except Exception:
        return False


class EmotivPlugin(BaseHardwarePlugin):

    def detect(self, filepath: str) -> bool:
        """
        Detect Emotiv files.
        Accepts:
        - .edf files with Emotiv channel fingerprint
        - .csv files with Emotiv channel names + Label column (FEIS dataset)
        """
        ext = filepath.lower()
        if ext.endswith(".edf"):
            try:
                raw = mne.io.read_raw_edf(filepath, preload=False, verbose=False)
                file_channels = set(raw.ch_names)
                return len(EMOTIV_CHANNEL_SET & file_channels) >= 6
            except Exception:
                return False
        if ext.endswith(".csv"):
            return _is_emotiv_csv(filepath)
        return False

    def read_raw(self, filepath: str, **kwargs) -> mne.io.BaseRaw:
        """
        Read Emotiv data.
        EDF: use MNE native reader.
        CSV: parse manually, extract 14 EEG channels, return RawArray at 256 Hz.
        """
        if filepath.lower().endswith(".edf"):
            return mne.io.read_raw_edf(filepath, preload=True, verbose=False)

        # CSV path (FEIS dataset)
        df = pd.read_csv(filepath)

        # Find which Emotiv channels are present in this file
        present_channels = [c for c in EMOTIV_CHANNELS if c in df.columns]

        # Extract EEG data — values are already in µV
        eeg_data = df[present_channels].values.T.astype(np.float64)  # (n_ch, n_samples)

        # Convert µV to Volts for MNE
        eeg_data = eeg_data * 1e-6

        sfreq = 256.0  # FEIS dataset is 256 Hz
        info = mne.create_info(
            ch_names=present_channels,
            sfreq=sfreq,
            ch_types=["eeg"] * len(present_channels)
        )
        info.set_montage("standard_1020", on_missing="ignore")
        raw = mne.io.RawArray(eeg_data, info, verbose=False)
        return raw

    def extract_events(self, filepath: str, raw: mne.io.BaseRaw) -> list[EventInfo]:
        """
        Extract events.
        EDF: use MNE annotations.
        CSV: extract from Label column — each transition to a new label = one event.
        """
        if filepath.lower().endswith(".edf"):
            event_list = []
            try:
                events, _ = mne.events_from_annotations(raw, verbose=False)
                for event in events:
                    onset_sec = event[0] / raw.info["sfreq"]
                    event_list.append(EventInfo(
                        onset=onset_sec,
                        duration=0.0,
                        description=str(event[2]),
                        trigger_source="software"
                    ))
            except Exception:
                pass
            return event_list

        # CSV path — extract label transitions
        df = pd.read_csv(filepath)
        event_list = []
        sfreq = raw.info["sfreq"]

        if "Label" not in df.columns:
            return event_list

        labels = df["Label"].fillna("").astype(str).values
        prev_label = ""

        for i, label in enumerate(labels):
            label = label.strip()
            if label and label != prev_label and label != "Label":
                # Compute duration: how many samples until label changes
                j = i + 1
                while j < len(labels) and labels[j].strip() == label:
                    j += 1
                duration = (j - i) / sfreq
                event_list.append(EventInfo(
                    onset=i / sfreq,
                    duration=duration,
                    description=label,
                    trigger_source="software"
                ))
                prev_label = label

        return event_list

    def get_metadata(self, filepath: str) -> HardwareMetadata:
        """Return Emotiv EPOC+ metadata for BIDS sidecar."""
        srate = 256.0 if filepath.lower().endswith(".csv") else 128.0
        return HardwareMetadata(
            manufacturer="Emotiv",
            model="EPOC+",
            sampling_rate=srate,
            channel_count=14,
            reference_scheme="CMS/DRL",
            power_line_freq=50.0,
            eeg_ground="CMS/DRL"
        )