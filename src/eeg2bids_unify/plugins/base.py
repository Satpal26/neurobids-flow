# base.py
# This is the BaseHardwarePlugin — every hardware plugin MUST implement these 4 methods.
# Think of this as a contract. Muse plugin, OpenBCI plugin, BrainProducts plugin — all follow this.

from abc import ABC, abstractmethod
from dataclasses import dataclass
import mne


@dataclass
class EventInfo:
    """Represents a single event/stimulus marker."""
    onset: float          # seconds from recording start
    duration: float       # duration of event (0.0 if instantaneous)
    description: str      # what happened — e.g. "stimulus_6hz"
    trigger_source: str   # where did this come from: 'hardware_ttl', 'lsl', 'software', 'keystroke'


@dataclass
class HardwareMetadata:
    """Hardware-specific info needed for BIDS sidecar JSON."""
    manufacturer: str         # e.g. "BrainProducts", "OpenBCI", "InteraXon"
    model: str                # e.g. "ActiChamp Plus", "Cyton", "Muse 2"
    sampling_rate: float      # Hz — e.g. 256.0
    channel_count: int        # number of EEG channels
    reference_scheme: str     # e.g. "FCz", "linked mastoids", "CMS/DRL"
    power_line_freq: float    # 50 or 60 Hz depending on country
    eeg_ground: str           # e.g. "AFz"


class BaseHardwarePlugin(ABC):
    """
    Every hardware plugin inherits from this class.
    If a plugin doesn't implement all 4 methods, Python will throw an error — by design.
    """

    @abstractmethod
    def detect(self, filepath: str) -> bool:
        """
        Look at the file and decide: can THIS plugin handle it?
        Returns True if yes, False if no.
        Example: BrainProductsPlugin checks if file ends with .vhdr
        """
        pass

    @abstractmethod
    def read_raw(self, filepath: str, **kwargs) -> mne.io.BaseRaw:
        """
        Read the raw EEG file and return an MNE Raw object.
        This is where the actual file parsing happens.
        Example: for .vhdr files, call mne.io.read_raw_brainvision()
        """
        pass

    @abstractmethod
    def extract_events(self, filepath: str, raw: mne.io.BaseRaw) -> list[EventInfo]:
        """
        Extract all events/markers from the recording.
        Returns a list of EventInfo objects — one per stimulus/response.
        Example: for BrainProducts, parse the .vmrk file
        """
        pass

    @abstractmethod
    def get_metadata(self, filepath: str) -> HardwareMetadata:
        """
        Return hardware info needed for BIDS sidecar JSON.
        Example: BrainProducts returns manufacturer="BrainProducts", model="ActiChamp Plus"
        """
        pass