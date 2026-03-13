# config.py
# Loads and validates YAML config file
# Provides clean config object to the rest of the tool

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DatasetConfig:
    name: str = "EEG Dataset"
    authors: list = field(default_factory=list)
    institution: str = ""
    ethics_approval: str = ""


@dataclass
class RecordingConfig:
    task: str = "ssvep"
    power_line_freq: float = 50.0
    subject_prefix: str = "sub"
    session_prefix: str = "ses"


@dataclass
class OutputConfig:
    overwrite: bool = True
    validate_bids: bool = True
    verbose: bool = False


@dataclass
class AppConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    event_mapping: dict = field(default_factory=dict)
    hardware: dict = field(default_factory=dict)


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load config from YAML file.
    If no path given — loads default_config.yaml from configs folder.
    If file not found — returns default config silently.
    """

    if config_path is None:
        # Look for default config relative to project root
        default_path = Path(__file__).parent.parent.parent.parent / "configs" / "default_config.yaml"
        config_path = str(default_path)

    config_path = Path(config_path)

    if not config_path.exists():
        print(f"[config] No config file found at {config_path} — using defaults")
        return AppConfig()

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        return AppConfig()

    # Parse dataset section
    dataset_raw = raw.get("dataset", {})
    dataset = DatasetConfig(
        name=dataset_raw.get("name", "EEG Dataset"),
        authors=dataset_raw.get("authors", []),
        institution=dataset_raw.get("institution", ""),
        ethics_approval=dataset_raw.get("ethics_approval", ""),
    )

    # Parse recording section
    recording_raw = raw.get("recording", {})
    recording = RecordingConfig(
        task=recording_raw.get("task", "ssvep"),
        power_line_freq=recording_raw.get("power_line_freq", 50.0),
        subject_prefix=recording_raw.get("subject_prefix", "sub"),
        session_prefix=recording_raw.get("session_prefix", "ses"),
    )

    # Parse output section
    output_raw = raw.get("output", {})
    output = OutputConfig(
        overwrite=output_raw.get("overwrite", True),
        validate_bids=output_raw.get("validate_bids", True),
        verbose=output_raw.get("verbose", False),
    )

    # Parse event mapping
    event_mapping = raw.get("event_mapping", {})
    # Convert all keys to strings
    event_mapping = {str(k): str(v) for k, v in event_mapping.items()}

    # Parse hardware overrides
    hardware = raw.get("hardware", {})

    config = AppConfig(
        dataset=dataset,
        recording=recording,
        output=output,
        event_mapping=event_mapping,
        hardware=hardware,
    )

    print(f"[config] Loaded config from {config_path}")
    return config
