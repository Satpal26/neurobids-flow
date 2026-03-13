# emotiv.py
# Plugin for Emotiv EPOC / EPOC+ headset
# Emotiv exports as .edf (via EmotivPRO) or .csv (via Emotiv Xavier)
# We support .edf format as it's most common for research use

import mne
import numpy as np
import pandas as pd
from .base import BaseHardwarePlugin, EventInfo, HardwareMetadata


class EmotivPlugin(BaseHardwarePlugin):

    def detect(self, filepath: str) -> bool:
        """
        Detect Emotiv .edf files.
        Emotiv EDF files contain channel names like AF3, F7, F3 etc.
        We check extension + channel names to distinguish from other EDF devices.
        """
        if not filepath.lower().endswith(".edf"):
            return False
        try:
            raw = mne.io.read_raw_edf(filepath, preload=False, verbose=False)
            emotiv_channels = {"AF3", "F7", "F3", "FC5", "T7", "P7",
                               "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"}
            file_channels = set(raw.ch_names)
            # If at least 6 Emotiv channels are present — it's an Emotiv file
            return len(emotiv_channels & file_channels) >= 6
        except Exception:
            return False

    def read_raw(self, filepath: str, **kwargs) -> mne.io.BaseRaw:
        """Read Emotiv EDF file using MNE's native EDF reader."""
        raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
        return raw

    def extract_events(self, filepath: str, raw: mne.io.BaseRaw) -> list[EventInfo]:
        """
        Extract events from Emotiv EDF annotations.
        EmotivPRO writes stimulus markers as EDF annotations.
        """
        event_list = []
        try:
            events, event_id = mne.events_from_annotations(raw, verbose=False)
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

    def get_metadata(self, filepath: str) -> HardwareMetadata:
        """Return Emotiv EPOC metadata for BIDS sidecar."""
        return HardwareMetadata(
            manufacturer="Emotiv",
            model="EPOC+",
            sampling_rate=128.0,
            channel_count=14,
            reference_scheme="CMS/DRL",
            power_line_freq=50.0,
            eeg_ground="CMS/DRL"
        )
