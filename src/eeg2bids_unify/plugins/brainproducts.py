# brainproducts.py
# Plugin for BrainProduct ActiChamp Plus (.vhdr / .vmrk / .eeg)
# Easiest plugin — already BIDS-recommended format

import mne
from .base import BaseHardwarePlugin, EventInfo, HardwareMetadata


class BrainProductsPlugin(BaseHardwarePlugin):

    def detect(self, filepath: str) -> bool:
        """Detect if file is a BrainVision file by checking extension."""
        return filepath.lower().endswith(".vhdr")

    def read_raw(self, filepath: str, **kwargs) -> mne.io.BaseRaw:
        """Read BrainVision file using MNE's native reader."""
        raw = mne.io.read_raw_brainvision(filepath, preload=True, verbose=False)
        return raw

    def extract_events(self, filepath: str, raw: mne.io.BaseRaw) -> list[EventInfo]:
        """Extract events from .vmrk marker file via MNE."""
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        event_list = []
        for event in events:
            onset_sec = event[0] / raw.info["sfreq"]
            description = str(event[2])
            event_list.append(EventInfo(
                onset=onset_sec,
                duration=0.0,
                description=description,
                trigger_source="hardware_ttl"
            ))
        return event_list

    def get_metadata(self, filepath: str) -> HardwareMetadata:
        """Return BrainProducts ActiChamp metadata for BIDS sidecar."""
        return HardwareMetadata(
            manufacturer="BrainProducts",
            model="ActiChamp Plus",
            sampling_rate=1000.0,  # default — overridden from actual file
            channel_count=32,      # default — overridden from actual file
            reference_scheme="FCz",
            power_line_freq=50.0,
            eeg_ground="AFz"
        )