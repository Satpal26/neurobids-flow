# harmonizer.py
# EventHarmonizer — normalizes raw EEG event markers into BIDS-compliant
# events.tsv AND injects HED strings into events.json sidecar
# NeuroBIDS-Flow | NTU Singapore BCI Lab

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EventInfo:
    """Single EEG event extracted by a hardware plugin."""
    onset: float                          # seconds from recording start
    duration: float                       # seconds (0.0 if instantaneous)
    description: str                      # raw marker string from hardware
    trigger_source: str = "hardware_ttl"  # hardware_ttl | lsl | software | keystroke


@dataclass
class HarmonizedEvent:
    """Single event after harmonization — ready for BIDS output."""
    onset: float
    duration: float
    trial_type: str        # standardized name (e.g. "rest_open")
    original_value: str    # raw marker preserved for traceability
    trigger_source: str
    hed: Optional[str] = None  # HED string (e.g. "Sensory-event, (Eyes, Open)")


class EventHarmonizer:
    """
    Normalizes heterogeneous consumer EEG event markers into:
      1. BIDS-compliant events.tsv  (onset | duration | trial_type | value | trigger_source)
      2. BIDS-compliant events.json (HED string dictionary sidecar)

    Supports 5 raw marker formats:
      - TTL triggers       (BrainProducts: "S  1", "S  2")
      - Numerical IDs      (OpenBCI: "1", "2", "99")
      - LSL markers        (Muse XDF: Lab Streaming Layer annotations)
      - Software strings   (Muse CSV, Emotiv: "eyes_open", "workload_high")
      - EDF annotations    (Emotiv EDF: annotation-based labels)

    Unknown markers are preserved with "unknown_" prefix for manual review.
    """

    def __init__(self, event_mapping: dict):
        """
        Parameters
        ----------
        event_mapping : dict
            From YAML config. Two supported formats:

            Simple (no HED):
                "1": "rest_open"
                "2": "cognitive_high"

            Extended (with HED):
                "1":
                    trial_type: "rest_open"
                    hed: "Sensory-event, (Eyes, Open)"
                "2":
                    trial_type: "cognitive_high"
                    hed: "Cognitive-effort, Task-difficulty/High"
        """
        self.trial_type_map: dict[str, str] = {}
        self.hed_map: dict[str, str] = {}

        for raw_code, mapping in event_mapping.items():
            key = str(raw_code)
            if isinstance(mapping, dict):
                # Extended format with HED
                self.trial_type_map[key] = mapping.get("trial_type", f"unknown_{key}")
                if "hed" in mapping:
                    self.hed_map[key] = mapping["hed"]
            else:
                # Simple format — just trial_type string
                self.trial_type_map[key] = str(mapping)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def harmonize(self, events: list[EventInfo]) -> list[HarmonizedEvent]:
        """
        Normalize a list of raw EventInfo objects into HarmonizedEvent objects.

        Parameters
        ----------
        events : list[EventInfo]
            Raw events from hardware plugin.

        Returns
        -------
        list[HarmonizedEvent]
            Harmonized events with trial_type and optional HED strings.
        """
        harmonized = []
        for ev in events:
            raw = str(ev.description).strip()
            trial_type = self.trial_type_map.get(raw, f"unknown_{raw}")
            hed = self.hed_map.get(raw, None)
            harmonized.append(HarmonizedEvent(
                onset=round(ev.onset, 6),
                duration=round(ev.duration, 6),
                trial_type=trial_type,
                original_value=raw,
                trigger_source=ev.trigger_source,
                hed=hed,
            ))
        return harmonized

    def write_events_tsv(self, events: list[HarmonizedEvent], filepath: str) -> None:
        """
        Write BIDS-compliant events.tsv file.

        Columns: onset | duration | trial_type | value | trigger_source
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write("onset\tduration\ttrial_type\tvalue\ttrigger_source\n")
            for ev in events:
                f.write(
                    f"{ev.onset}\t{ev.duration}\t{ev.trial_type}\t"
                    f"{ev.original_value}\t{ev.trigger_source}\n"
                )

    def write_events_json(self, filepath: str) -> bool:
        """
        Write BIDS-compliant events.json sidecar with HED string dictionary.

        Only written if at least one HED string is defined in the config.
        Returns True if file was written, False if no HED strings available.

        Output format:
        {
            "onset":          {"Description": "..."},
            "duration":       {"Description": "..."},
            "trial_type":     {"Description": "...", "HED": { "rest_open": "...", ... }},
            "value":          {"Description": "..."},
            "trigger_source": {"Description": "..."}
        }
        """
        if not self.hed_map:
            return False

        # Build HED dictionary keyed by trial_type (not raw code)
        hed_by_trial_type: dict[str, str] = {}
        for raw_code, hed_string in self.hed_map.items():
            trial_type = self.trial_type_map.get(raw_code, f"unknown_{raw_code}")
            hed_by_trial_type[trial_type] = hed_string

        sidecar = {
            "onset": {
                "Description": "Onset of the event in seconds from the start of the recording."
            },
            "duration": {
                "Description": "Duration of the event in seconds. 0 indicates an instantaneous event."
            },
            "trial_type": {
                "Description": (
                    "Standardized event label produced by NeuroBIDS-Flow EventHarmonizer. "
                    "HED strings provide semantic annotation compliant with HED schema 8.2.0."
                ),
                "HED": hed_by_trial_type
            },
            "value": {
                "Description": "Original raw marker code as recorded by the hardware device."
            },
            "trigger_source": {
                "Description": (
                    "Source type of the event marker. One of: hardware_ttl, lsl, software, keystroke."
                )
            }
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(sidecar, f, indent=4)

        return True

    def get_unique_trial_types(self, events: list[HarmonizedEvent]) -> list[str]:
        """Return sorted list of unique trial_type values in this recording."""
        return sorted(set(ev.trial_type for ev in events))

    def has_hed(self) -> bool:
        """Return True if at least one HED string is configured."""
        return len(self.hed_map) > 0