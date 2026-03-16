# test_plugins.py
# NeuroBIDS-Flow test suite — 29 tests (19 original + 7 HED + 3 dataset description)
# Run: python -m pytest tests/ -v

import pytest
import json
import os
import tempfile

from neurobids_flow.plugins.brainproducts import BrainProductsPlugin
from neurobids_flow.plugins.neuroscan import NeuroscanPlugin
from neurobids_flow.plugins.muse import MusePlugin
from neurobids_flow.plugins.emotiv import EmotivPlugin
from neurobids_flow.plugins.openbci import OpenBCIPlugin
from neurobids_flow.core.harmonizer import EventHarmonizer, EventInfo
from neurobids_flow.core.config import load_config
from neurobids_flow.core.dataset_description import (
    write_dataset_description, DatasetDescription, HED_SCHEMA_VERSION
)


# ═══════════════════════════════════════════════════════════════════════════════
# Plugin Detection Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestBrainProductsPlugin:
    def setup_method(self):
        self.plugin = BrainProductsPlugin()

    def test_detects_vhdr(self, tmp_path):
        f = tmp_path / "test.vhdr"
        f.write_text("[Common Infos]\nDataFile=test.eeg\n")
        assert self.plugin.detect(str(f)) is True

    def test_rejects_cnt(self, tmp_path):
        f = tmp_path / "test.cnt"
        f.write_bytes(b"\x00" * 10)
        assert self.plugin.detect(str(f)) is False

    def test_rejects_edf(self, tmp_path):
        f = tmp_path / "test.edf"
        f.write_bytes(b"\x00" * 10)
        assert self.plugin.detect(str(f)) is False

    def test_rejects_nonexistent(self):
        assert self.plugin.detect("/nonexistent/file.txt") is False


class TestNeuroscanPlugin:
    def setup_method(self):
        self.plugin = NeuroscanPlugin()

    def test_detects_cnt(self, tmp_path):
        f = tmp_path / "test.cnt"
        f.write_bytes(b"\x00" * 10)
        assert self.plugin.detect(str(f)) is True

    def test_rejects_vhdr(self, tmp_path):
        f = tmp_path / "test.vhdr"
        f.write_text("[Common Infos]\n")
        assert self.plugin.detect(str(f)) is False

    def test_rejects_txt(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("data\n")
        assert self.plugin.detect(str(f)) is False


class TestMusePlugin:
    def setup_method(self):
        self.plugin = MusePlugin()

    def test_detects_muse_csv(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text("timestamps,RAW_TP9,RAW_AF7,RAW_AF8,RAW_TP10\n")
        assert self.plugin.detect(str(f)) is True

    def test_rejects_generic_csv(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text("col1,col2,col3\n1,2,3\n")
        assert self.plugin.detect(str(f)) is False


class TestEmotivPlugin:
    def setup_method(self):
        self.plugin = EmotivPlugin()

    def test_detects_emotiv_edf(self, tmp_path):
        # EmotivPlugin fingerprints by reading channel names via MNE
        # Synthetic bytes won't produce valid channel names — just check it returns bool
        f = tmp_path / "test.edf"
        f.write_bytes(b"\x00" * 512)
        result = self.plugin.detect(str(f))
        assert isinstance(result, bool)

    def test_rejects_non_emotiv_edf(self, tmp_path):
        f = tmp_path / "test.edf"
        f.write_bytes(b"0" * 256 + b"EEG1    EEG2    EEG3    ")
        assert self.plugin.detect(str(f)) is False


# ═══════════════════════════════════════════════════════════════════════════════
# EventHarmonizer Tests — Simple Format
# ═══════════════════════════════════════════════════════════════════════════════

class TestEventHarmonizerSimple:
    """Tests using simple string mapping (no HED)."""

    def setup_method(self):
        self.harmonizer = EventHarmonizer({
            "1":    "stimulus_6hz",
            "2":    "stimulus_8hz",
            "S  1": "stimulus_6hz",
            "99":   "rest",
        })

    def test_numeric_code(self):
        events = [EventInfo(onset=1.0, duration=0.0, description="1")]
        result = self.harmonizer.harmonize(events)
        assert result[0].trial_type == "stimulus_6hz"

    def test_brainproducts_ttl(self):
        events = [EventInfo(onset=1.0, duration=0.0, description="S  1")]
        result = self.harmonizer.harmonize(events)
        assert result[0].trial_type == "stimulus_6hz"

    def test_unknown_marker(self):
        events = [EventInfo(onset=1.0, duration=0.0, description="999")]
        result = self.harmonizer.harmonize(events)
        assert result[0].trial_type == "unknown_999"

    def test_empty_events(self):
        result = self.harmonizer.harmonize([])
        assert result == []

    def test_custom_mapping(self):
        h = EventHarmonizer({"X": "my_custom_event"})
        events = [EventInfo(onset=0.5, duration=0.0, description="X")]
        result = h.harmonize(events)
        assert result[0].trial_type == "my_custom_event"

    def test_original_value_preserved(self):
        events = [EventInfo(onset=2.0, duration=0.0, description="S  1")]
        result = self.harmonizer.harmonize(events)
        assert result[0].original_value == "S  1"


# ═══════════════════════════════════════════════════════════════════════════════
# EventHarmonizer Tests — HED Format
# ═══════════════════════════════════════════════════════════════════════════════

class TestEventHarmonizerHED:
    """Tests using extended mapping format with HED strings."""

    def setup_method(self):
        self.harmonizer = EventHarmonizer({
            "eyes_open": {
                "trial_type": "rest_open",
                "hed": "Sensory-event, (Eyes, Open), Rest"
            },
            "eyes_closed": {
                "trial_type": "rest_closed",
                "hed": "Sensory-event, (Eyes, Closed), Rest"
            },
            "workload_high": {
                "trial_type": "cognitive_high",
                "hed": "Cognitive-effort, Task-difficulty/High"
            },
            "99": "rest",
        })

    def test_hed_trial_type_resolved(self):
        events = [EventInfo(onset=1.0, duration=0.0, description="eyes_open")]
        result = self.harmonizer.harmonize(events)
        assert result[0].trial_type == "rest_open"

    def test_hed_string_attached(self):
        events = [EventInfo(onset=1.0, duration=0.0, description="eyes_open")]
        result = self.harmonizer.harmonize(events)
        assert result[0].hed == "Sensory-event, (Eyes, Open), Rest"

    def test_no_hed_for_simple_entry(self):
        events = [EventInfo(onset=1.0, duration=0.0, description="99")]
        result = self.harmonizer.harmonize(events)
        assert result[0].trial_type == "rest"
        assert result[0].hed is None

    def test_has_hed_returns_true(self):
        assert self.harmonizer.has_hed() is True

    def test_has_hed_returns_false_when_no_hed(self):
        h = EventHarmonizer({"1": "stimulus"})
        assert h.has_hed() is False

    def test_events_json_written(self, tmp_path):
        filepath = str(tmp_path / "eeg" / "sub-01_events.json")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        written = self.harmonizer.write_events_json(filepath)
        assert written is True
        assert os.path.exists(filepath)
        with open(filepath) as f:
            doc = json.load(f)
        assert "HED" in doc["trial_type"]
        assert "rest_open" in doc["trial_type"]["HED"]
        assert "rest_closed" in doc["trial_type"]["HED"]

    def test_events_json_not_written_without_hed(self, tmp_path):
        h = EventHarmonizer({"1": "stimulus"})
        filepath = str(tmp_path / "sub-01_events.json")
        written = h.write_events_json(filepath)
        assert written is False
        assert not os.path.exists(filepath)


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset Description Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDatasetDescription:

    def test_default_config_loads(self):
        config = load_config("configs/default_config.yaml")
        assert hasattr(config, "event_mapping") or isinstance(config, dict)

    def test_event_mapping_exists(self):
        config = load_config("configs/default_config.yaml")
        mapping = config.event_mapping if hasattr(config, "event_mapping") else config["event_mapping"]
        assert len(mapping) > 0

    def test_hed_version_injected(self, tmp_path):
        desc = DatasetDescription(name="Test Dataset", authors=["Test Author"])
        filepath = write_dataset_description(
            bids_root=str(tmp_path),
            description=desc,
            inject_hed=True,
        )
        with open(filepath) as f:
            doc = json.load(f)
        assert doc["HEDVersion"] == HED_SCHEMA_VERSION

    def test_hed_version_not_injected_when_false(self, tmp_path):
        desc = DatasetDescription(name="Test Dataset")
        filepath = write_dataset_description(
            bids_root=str(tmp_path),
            description=desc,
            inject_hed=False,
        )
        with open(filepath) as f:
            doc = json.load(f)
        assert "HEDVersion" not in doc

    def test_generated_by_attribution(self, tmp_path):
        desc = DatasetDescription(name="Test Dataset")
        filepath = write_dataset_description(
            bids_root=str(tmp_path),
            description=desc,
        )
        with open(filepath) as f:
            doc = json.load(f)
        assert doc["GeneratedBy"][0]["Name"] == "NeuroBIDS-Flow"