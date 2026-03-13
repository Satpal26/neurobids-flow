# neuroscan.py
# Plugin for Neuroscan NuAmps (.cnt format)
# MNE has a native reader for .cnt so this is straightforward

import mne
from .base import BaseHardwarePlugin, EventInfo, HardwareMetadata


class NeuroscanPlugin(BaseHardwarePlugin):

    def detect(self, filepath: str) -> bool:
        """Detect if file is a Neuroscan .cnt file."""
        return filepath.lower().endswith(".cnt")

    def read_raw(self, filepath: str, **kwargs) -> mne.io.BaseRaw:
        """Read Neuroscan .cnt file using MNE's native reader."""
        raw = mne.io.read_raw_cnt(filepath, preload=True, verbose=False)
        return raw

    def extract_events(self, filepath: str, raw: mne.io.BaseRaw) -> list[EventInfo]:
        """Extract events embedded in Neuroscan .cnt event channel."""
        try:
            events = mne.find_events(raw, verbose=False)
            event_list = []
            for event in events:
                onset_sec = event[0] / raw.info["sfreq"]
                event_list.append(EventInfo(
                    onset=onset_sec,
                    duration=0.0,
                    description=str(event[2]),
                    trigger_source="software"
                ))
            return event_list
        except Exception:
            # Some .cnt files have no events — return empty list
            return []

    def get_metadata(self, filepath: str) -> HardwareMetadata:
        """Return Neuroscan NuAmps metadata for BIDS sidecar."""
        return HardwareMetadata(
            manufacturer="Neuroscan",
            model="NuAmps 40ch",
            sampling_rate=1000.0,
            channel_count=40,
            reference_scheme="linked mastoids",
            power_line_freq=50.0,
            eeg_ground="AFz"
        )
