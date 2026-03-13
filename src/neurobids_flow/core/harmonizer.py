# harmonizer.py
# EventHarmonizer — the core innovation of this tool.
# Every hardware device stores events differently:
#   BrainProducts → TTL triggers in .vmrk file
#   Neuroscan     → event channel in .cnt file
#   OpenBCI       → marker column in .txt file
#   Muse          → Marker column in .csv file
#   Emotiv        → EDF annotations
# This class normalizes ALL of them into one unified BIDS events.tsv format

import pandas as pd
from dataclasses import dataclass
from typing import Optional
from ..plugins.base import EventInfo


# Standard SSVEP stimulus frequencies used in our lab
SSVEP_FREQ_MAP = {
    # By frequency value
    "6":  "stimulus_6hz",
    "7":  "stimulus_7hz",
    "8":  "stimulus_8hz",
    "10": "stimulus_10hz",
    "12": "stimulus_12hz",
    "15": "stimulus_15hz",
    "20": "stimulus_20hz",
    # By trigger number (BrainProducts S1, S2 format)
    "1": "stimulus_6hz",
    "2": "stimulus_8hz",
    "3": "stimulus_10hz",
    "4": "stimulus_12hz",
    "5": "stimulus_15hz",
}


@dataclass
class HarmonizedEvent:
    """A single unified event ready for BIDS events.tsv"""
    onset: float          # seconds from recording start
    duration: float       # seconds (0.0 if instantaneous)
    trial_type: str       # standardized event name e.g. "stimulus_12hz"
    value: str            # original raw value from hardware
    trigger_source: str   # where it came from


class EventHarmonizer:
    """
    Takes raw EventInfo list from any hardware plugin
    and normalizes it into unified HarmonizedEvent list
    ready for BIDS events.tsv
    """

    def __init__(self, custom_mapping: Optional[dict] = None):
        """
        custom_mapping — optional dict to override default frequency mapping
        e.g. {"S  1": "stimulus_6hz", "S  2": "stimulus_12hz"}
        This is loaded from YAML config
        """
        self.mapping = {**SSVEP_FREQ_MAP}
        if custom_mapping:
            self.mapping.update(custom_mapping)

    def harmonize(self, events: list[EventInfo]) -> list[HarmonizedEvent]:
        """
        Convert raw EventInfo list → HarmonizedEvent list.
        Tries to map description to a standard trial_type.
        If no mapping found — keeps original description as trial_type.
        """
        harmonized = []
        for event in events:
            trial_type = self._map_event(event.description)
            harmonized.append(HarmonizedEvent(
                onset=event.onset,
                duration=event.duration,
                trial_type=trial_type,
                value=event.description,
                trigger_source=event.trigger_source
            ))
        return harmonized

    def _map_event(self, description: str) -> str:
        """
        Try to find a standard name for this event description.
        Handles common hardware-specific formats automatically.
        """
        desc = description.strip()

        # Direct match
        if desc in self.mapping:
            return self.mapping[desc]

        # BrainProducts format — "S  1", "S  2", "S 12" etc
        if desc.startswith("S"):
            number = desc.replace("S", "").strip()
            if number in self.mapping:
                return self.mapping[number]

        # Neuroscan format — numeric string "1", "2" etc
        if desc.isdigit():
            if desc in self.mapping:
                return self.mapping[desc]

        # Already looks like a standard name
        if desc.startswith("stimulus_"):
            return desc

        # Unknown — keep original
        return f"unknown_{desc}"

    def to_dataframe(self, harmonized: list[HarmonizedEvent]) -> pd.DataFrame:
        """
        Convert harmonized events to pandas DataFrame
        ready to be written as BIDS events.tsv
        """
        if not harmonized:
            return pd.DataFrame(columns=["onset", "duration", "trial_type", "value", "trigger_source"])

        return pd.DataFrame([{
            "onset": e.onset,
            "duration": e.duration,
            "trial_type": e.trial_type,
            "value": e.value,
            "trigger_source": e.trigger_source
        } for e in harmonized])

    def to_bids_tsv(self, harmonized: list[HarmonizedEvent], output_path: str):
        """Write harmonized events directly to BIDS events.tsv file."""
        df = self.to_dataframe(harmonized)
        df.to_csv(output_path, sep="\t", index=False)
        print(f"[harmonizer] Written {len(df)} events to {output_path}")
