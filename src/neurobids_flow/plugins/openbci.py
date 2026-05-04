# openbci.py
# Plugin for OpenBCI Cyton board
# Supports both standard format (with %OpenBCI header) and
# headerless format (raw CSV with 25 columns, no % comments)
import mne
import numpy as np
import pandas as pd
from pathlib import Path
from .base import BaseHardwarePlugin, EventInfo, HardwareMetadata


def _is_headerless_openbci(filepath: str) -> bool:
    """
    Detect headerless OpenBCI Cyton format.
    Characteristics: no % comments, first line starts with '0.0,'
    and has ~25 comma-separated numeric columns.
    """
    try:
        with open(filepath, "r") as f:
            first_line = f.readline().strip()
        if first_line.startswith("%"):
            return False
        parts = first_line.split(",")
        if len(parts) < 20:
            return False
        # Try parsing first few values as floats
        float(parts[0])
        float(parts[1])
        float(parts[2])
        return True
    except Exception:
        return False


class OpenBCIPlugin(BaseHardwarePlugin):

    def detect(self, filepath: str) -> bool:
        """
        Detect OpenBCI .txt file.
        Accepts both:
        - Standard format: first line contains '%OpenBCI'
        - Headerless format: raw 25-column numeric CSV (Mendeley dataset style)
        """
        if not filepath.lower().endswith(".txt"):
            return False
        try:
            with open(filepath, "r") as f:
                first_line = f.readline()
            if "%OpenBCI" in first_line:
                return True
            return _is_headerless_openbci(filepath)
        except Exception:
            return False

    def read_raw(self, filepath: str, **kwargs) -> mne.io.BaseRaw:
        """
        Parse OpenBCI .txt format and return MNE RawArray.
        Handles both standard (% comments) and headerless formats.
        Columns 1-8 are EEG channels in both formats.
        """
        sfreq = 250.0  # OpenBCI Cyton default sampling rate
        rows = []
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("%") or line == "":
                    continue
                rows.append(line.split(","))

        df = pd.DataFrame(rows)
        # Only convert EEG columns (1-8) — other columns may have non-numeric
        # values (timestamps, status bytes) that cause dropna to remove all rows
        eeg_cols = df.iloc[:, 1:9].apply(pd.to_numeric, errors="coerce")
        eeg_cols = eeg_cols.dropna()

        # EEG data shape: (8, n_samples)
        eeg_data = eeg_cols.values.T

        # OpenBCI Cyton scale factor: 0.022351744455307625 uV/count
        # Headerless datasets already store values in uV (large numbers ~35000)
        # detect by magnitude: if mean abs > 100, already in uV
        mean_abs = np.mean(np.abs(eeg_data))
        if mean_abs > 100:
            # Already in uV — just convert to Volts for MNE
            eeg_data = eeg_data * 1e-6
        else:
            # Raw counts — apply scale factor then convert to Volts
            scale_factor = 0.022351744455307625
            eeg_data = eeg_data * scale_factor * 1e-6

        ch_names = [f"EEG{i+1}" for i in range(8)]
        ch_types = ["eeg"] * 8
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        info.set_montage("standard_1020", on_missing="ignore")
        raw = mne.io.RawArray(eeg_data, info, verbose=False)
        return raw

    def extract_events(self, filepath: str, raw: mne.io.BaseRaw) -> list[EventInfo]:
        """
        Extract events from OpenBCI .txt.
        Standard format: marker in last column (string).
        Headerless format: task label encoded in filename (natural/low/mid/high).
        Falls back to filename-based label if no inline markers found.
        """
        event_list = []
        sfreq = raw.info["sfreq"]
        sample_idx = 0
        found_inline = False

        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("%") or line == "":
                    continue
                parts = line.split(",")
                if len(parts) > 12:
                    marker = parts[12].strip()
                    if marker and marker not in ("0", "0.0", "192.0"):
                        event_list.append(EventInfo(
                            onset=sample_idx / sfreq,
                            duration=0.0,
                            description=marker,
                            trigger_source="software"
                        ))
                        found_inline = True
                sample_idx += 1

        # Fallback: derive label from filename for headerless datasets
        # e.g. natural-1.txt -> trial_type=natural, highlevel-3.txt -> highlevel
        if not found_inline:
            stem = Path(filepath).stem  # e.g. "natural-1"
            label = stem.split("-")[0]  # e.g. "natural"
            n_samples = sample_idx
            duration = n_samples / sfreq
            event_list.append(EventInfo(
                onset=0.0,
                duration=duration,
                description=label,
                trigger_source="filename"
            ))

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