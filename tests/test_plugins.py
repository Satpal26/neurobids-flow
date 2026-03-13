# test_plugins.py
# Basic tests for all 5 hardware plugins
# Tests plugin detection logic — no real EEG files needed

import pytest
import numpy as np
import mne
from src.neurobids_flow.plugins.brainproducts import BrainProductsPlugin
from src.neurobids_flow.plugins.neuroscan import NeuroscanPlugin
from src.neurobids_flow.plugins.openbci import OpenBCIPlugin
from src.neurobids_flow.plugins.muse import MusePlugin
from src.neurobids_flow.plugins.emotiv import EmotivPlugin
from src.neurobids_flow.plugins.base import EventInfo
from src.neurobids_flow.core.harmonizer import EventHarmonizer
from src.neurobids_flow.core.config import load_config


# ── Plugin Detection Tests ─────────────────────────────────

class TestBrainProductsPlugin:
    def setup_method(self):
        self.plugin = BrainProductsPlugin()

    def test_detects_vhdr(self):
        assert self.plugin.detect("recording.vhdr") == True

    def test_rejects_cnt(self):
        assert self.plugin.detect("recording.cnt") == False

    def test_rejects_txt(self):
        assert self.plugin.detect("recording.txt") == False

    def test_case_insensitive(self):
        assert self.plugin.detect("recording.VHDR") == True


class TestNeuroscanPlugin:
    def setup_method(self):
        self.plugin = NeuroscanPlugin()

    def test_detects_cnt(self):
        assert self.plugin.detect("recording.cnt") == True

    def test_rejects_vhdr(self):
        assert self.plugin.detect("recording.vhdr") == False

    def test_case_insensitive(self):
        assert self.plugin.detect("recording.CNT") == True


class TestMusePlugin:
    def setup_method(self):
        self.plugin = MusePlugin()

    def test_detects_xdf(self):
        assert self.plugin.detect("recording.xdf") == True

    def test_rejects_vhdr(self):
        assert self.plugin.detect("recording.vhdr") == False


class TestEmotivPlugin:
    def setup_method(self):
        self.plugin = EmotivPlugin()

    def test_rejects_non_edf(self):
        assert self.plugin.detect("recording.vhdr") == False

    def test_rejects_cnt(self):
        assert self.plugin.detect("recording.cnt") == False


# ── EventHarmonizer Tests ──────────────────────────────────

class TestEventHarmonizer:
    def setup_method(self):
        self.harmonizer = EventHarmonizer()

    def test_maps_numeric_trigger(self):
        events = [EventInfo(onset=1.0, duration=0.0,
                           description="6", trigger_source="software")]
        result = self.harmonizer.harmonize(events)
        assert result[0].trial_type == "stimulus_6hz"

    def test_maps_brainproducts_trigger(self):
        events = [EventInfo(onset=1.0, duration=0.0,
                           description="S  1", trigger_source="hardware_ttl")]
        result = self.harmonizer.harmonize(events)
        assert result[0].trial_type == "stimulus_6hz"

    def test_unknown_trigger_prefixed(self):
        events = [EventInfo(onset=1.0, duration=0.0,
                           description="999", trigger_source="software")]
        result = self.harmonizer.harmonize(events)
        assert result[0].trial_type == "unknown_999"

    def test_empty_events(self):
        result = self.harmonizer.harmonize([])
        assert result == []

    def test_custom_mapping(self):
        harmonizer = EventHarmonizer(custom_mapping={"X1": "stimulus_6hz"})
        events = [EventInfo(onset=1.0, duration=0.0,
                           description="X1", trigger_source="software")]
        result = harmonizer.harmonize(events)
        assert result[0].trial_type == "stimulus_6hz"

    def test_to_dataframe(self):
        events = [EventInfo(onset=1.0, duration=0.0,
                           description="6", trigger_source="software")]
        harmonized = self.harmonizer.harmonize(events)
        df = self.harmonizer.to_dataframe(harmonized)
        assert len(df) == 1
        assert "onset" in df.columns
        assert "trial_type" in df.columns


# ── Config Tests ───────────────────────────────────────────

class TestConfig:
    def test_default_config_loads(self):
        config = load_config()
        assert config is not None
        assert config.recording.task == "ssvep"
        assert config.recording.power_line_freq == 50.0

    def test_default_event_mapping_exists(self):
        config = load_config()
        # Default config should have event mappings
        assert isinstance(config.event_mapping, dict)
