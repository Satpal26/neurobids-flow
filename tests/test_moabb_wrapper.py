# test_moabb_wrapper.py
# NeuroBIDS-Flow — MOABB wrapper test suite
# Tests the NBIDSFDataset class that bridges BIDS-EEG output to MOABB
# Run: python -m pytest tests/test_moabb_wrapper.py -v
#
# NeuroBIDS-Flow | NTU Singapore BCI Lab

import os
import re
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from neurobids_flow.moabb_wrapper import NBIDSFDataset, NEUROBIDS_EVENTS


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_bids_root(tmp_path):
    """
    Create a minimal fake BIDS directory structure:
        tmp/
          sub-01/
            ses-01/
              eeg/
                sub-01_ses-01_task-workload_eeg.vhdr  (empty placeholder)
    """
    eeg_dir = tmp_path / "sub-01" / "ses-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    # Placeholder EEG file — just needs to exist for path detection tests
    (eeg_dir / "sub-01_ses-01_task-workload_eeg.vhdr").write_text("")
    return tmp_path


@pytest.fixture
def mock_raw():
    """Minimal MNE-like Raw mock object."""
    raw = MagicMock()
    raw.ch_names = ["Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2"]
    raw.times = np.linspace(0, 10.0, 2560)
    raw.info = {"sfreq": 256.0}
    raw.annotations = MagicMock()
    raw.annotations.__len__ = MagicMock(return_value=3)
    return raw


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Initialisation Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestNBIDSFDatasetInit:

    def test_init_with_valid_bids_root(self, mock_bids_root):
        dataset = NBIDSFDataset(bids_root=str(mock_bids_root), task="workload")
        assert dataset is not None

    def test_subjects_auto_detected(self, mock_bids_root):
        dataset = NBIDSFDataset(bids_root=str(mock_bids_root))
        assert dataset.subject_list == [1]

    def test_sessions_auto_detected(self, mock_bids_root):
        dataset = NBIDSFDataset(bids_root=str(mock_bids_root))
        assert dataset._sessions == ["01"]

    def test_default_events_loaded(self, mock_bids_root):
        dataset = NBIDSFDataset(bids_root=str(mock_bids_root))
        assert dataset.event_id == NEUROBIDS_EVENTS

    def test_custom_events_override(self, mock_bids_root):
        custom = {"cognitive_low": 0, "cognitive_high": 1}
        dataset = NBIDSFDataset(bids_root=str(mock_bids_root), events=custom)
        assert dataset.event_id == custom

    def test_default_interval(self, mock_bids_root):
        dataset = NBIDSFDataset(bids_root=str(mock_bids_root))
        assert dataset.interval == [-0.5, 3.0]

    def test_custom_interval(self, mock_bids_root):
        dataset = NBIDSFDataset(bids_root=str(mock_bids_root), interval=[0.0, 4.0])
        assert dataset.interval == [0.0, 4.0]

    def test_raises_when_no_subjects_found(self, tmp_path):
        # Empty directory — no sub-XX folders
        with pytest.raises(ValueError, match="No subjects found"):
            NBIDSFDataset(bids_root=str(tmp_path))

    def test_explicit_subjects_override_autodetect(self, mock_bids_root):
        dataset = NBIDSFDataset(
            bids_root=str(mock_bids_root),
            subjects=[1, 2, 3]
        )
        assert dataset.subject_list == [1, 2, 3]

    def test_moabb_code_is_set(self, mock_bids_root):
        dataset = NBIDSFDataset(bids_root=str(mock_bids_root))
        assert dataset.code == "NBIDSF"

    def test_paradigm_is_resting(self, mock_bids_root):
        dataset = NBIDSFDataset(bids_root=str(mock_bids_root))
        assert dataset.paradigm == "resting"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. data_path Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataPath:

    def test_returns_correct_subject_path(self, mock_bids_root):
        dataset = NBIDSFDataset(bids_root=str(mock_bids_root))
        paths = dataset.data_path(subject=1)
        assert len(paths) == 1
        assert "sub-01" in paths[0]

    def test_returns_list(self, mock_bids_root):
        dataset = NBIDSFDataset(bids_root=str(mock_bids_root))
        result = dataset.data_path(subject=1)
        assert isinstance(result, list)

    def test_raises_for_missing_subject(self, mock_bids_root):
        dataset = NBIDSFDataset(bids_root=str(mock_bids_root))
        with pytest.raises(FileNotFoundError):
            dataset.data_path(subject=99)

    def test_subject_id_zero_padded(self, mock_bids_root):
        dataset = NBIDSFDataset(bids_root=str(mock_bids_root))
        paths = dataset.data_path(subject=1)
        assert "sub-01" in paths[0]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Auto-detection Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestAutoDetection:

    def test_detects_multiple_subjects(self, tmp_path):
        # Create sub-01, sub-02, sub-03
        for i in [1, 2, 3]:
            (tmp_path / f"sub-{i:02d}" / "ses-01" / "eeg").mkdir(parents=True)
        dataset = NBIDSFDataset(bids_root=str(tmp_path), subjects=[1, 2, 3])
        assert dataset.subject_list == [1, 2, 3]

    def test_detects_multiple_sessions(self, tmp_path):
        # Create sub-01 with ses-01 and ses-02
        for ses in ["01", "02"]:
            (tmp_path / "sub-01" / f"ses-{ses}" / "eeg").mkdir(parents=True)
        dataset = NBIDSFDataset(bids_root=str(tmp_path))
        assert "01" in dataset._sessions
        assert "02" in dataset._sessions

    def test_ignores_non_subject_folders(self, tmp_path):
        # sub-01 valid, "docs" and "configs" should be ignored
        (tmp_path / "sub-01" / "ses-01" / "eeg").mkdir(parents=True)
        (tmp_path / "docs").mkdir()
        (tmp_path / "configs").mkdir()
        dataset = NBIDSFDataset(bids_root=str(tmp_path))
        assert dataset.subject_list == [1]

    def test_falls_back_to_session_01_when_no_sessions(self, tmp_path):
        # sub-01 with no ses-XX subfolder
        (tmp_path / "sub-01").mkdir()
        dataset = NBIDSFDataset(bids_root=str(tmp_path), subjects=[1])
        assert dataset._sessions == ["01"]

    def test_empty_bids_root_returns_empty_subjects(self, tmp_path):
        # Directly instantiate with bids_root set — no sub-XX folders exist
        instance = object.__new__(NBIDSFDataset)
        instance.bids_root = tmp_path
        assert instance._detect_subjects() == []


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Event Map Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestEventMap:

    def test_all_resting_events_present(self, mock_bids_root):
        dataset = NBIDSFDataset(bids_root=str(mock_bids_root))
        assert "rest_open"   in dataset.event_id
        assert "rest_closed" in dataset.event_id
        assert "rest"        in dataset.event_id

    def test_all_workload_events_present(self, mock_bids_root):
        dataset = NBIDSFDataset(bids_root=str(mock_bids_root))
        assert "cognitive_low"  in dataset.event_id
        assert "cognitive_high" in dataset.event_id
        assert "fatigue"        in dataset.event_id
        assert "alert"          in dataset.event_id

    def test_all_emotion_events_present(self, mock_bids_root):
        dataset = NBIDSFDataset(bids_root=str(mock_bids_root))
        assert "emotion_positive" in dataset.event_id
        assert "emotion_negative" in dataset.event_id
        assert "arousal_high"     in dataset.event_id
        assert "arousal_low"      in dataset.event_id

    def test_event_values_are_integers(self, mock_bids_root):
        dataset = NBIDSFDataset(bids_root=str(mock_bids_root))
        for val in dataset.event_id.values():
            assert isinstance(val, int)

    def test_event_values_are_unique(self, mock_bids_root):
        dataset = NBIDSFDataset(bids_root=str(mock_bids_root))
        values = list(dataset.event_id.values())
        assert len(values) == len(set(values))


# ═══════════════════════════════════════════════════════════════════════════════
# 5. _get_single_subject_data Tests (with mocked read_raw_bids)
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetSingleSubjectData:

    def test_returns_nested_dict_structure(self, mock_bids_root, mock_raw):
        dataset = NBIDSFDataset(bids_root=str(mock_bids_root))
        mock_path = MagicMock()
        mock_path.fpath.suffix = ".vhdr"

        with patch("neurobids_flow.moabb_wrapper.BIDSPath") as mock_bids_path, \
             patch("neurobids_flow.moabb_wrapper.read_raw_bids", return_value=mock_raw):
            mock_bids_path.return_value.match.return_value = [mock_path]
            result = dataset._get_single_subject_data(1)

        assert isinstance(result, dict)
        assert "session_01" in result
        assert "run_0" in result["session_01"]

    def test_raw_object_is_returned(self, mock_bids_root, mock_raw):
        dataset = NBIDSFDataset(bids_root=str(mock_bids_root))
        mock_path = MagicMock()
        mock_path.fpath.suffix = ".vhdr"

        with patch("neurobids_flow.moabb_wrapper.BIDSPath") as mock_bids_path, \
             patch("neurobids_flow.moabb_wrapper.read_raw_bids", return_value=mock_raw):
            mock_bids_path.return_value.match.return_value = [mock_path]
            result = dataset._get_single_subject_data(1)

        raw = result["session_01"]["run_0"]
        assert raw.ch_names == mock_raw.ch_names

    def test_tsv_files_are_filtered_out(self, mock_bids_root, mock_raw):
        dataset = NBIDSFDataset(bids_root=str(mock_bids_root))

        tsv_path  = MagicMock(); tsv_path.fpath.suffix  = ".tsv"
        json_path = MagicMock(); json_path.fpath.suffix = ".json"
        vhdr_path = MagicMock(); vhdr_path.fpath.suffix = ".vhdr"

        with patch("neurobids_flow.moabb_wrapper.BIDSPath") as mock_bids_path, \
             patch("neurobids_flow.moabb_wrapper.read_raw_bids", return_value=mock_raw):
            mock_bids_path.return_value.match.return_value = [
                tsv_path, json_path, vhdr_path
            ]
            result = dataset._get_single_subject_data(1)

        # Only vhdr should have been loaded — exactly 1 run
        assert len(result["session_01"]) == 1

    def test_raises_when_no_runs_loaded(self, mock_bids_root):
        dataset = NBIDSFDataset(bids_root=str(mock_bids_root))

        with patch("neurobids_flow.moabb_wrapper.BIDSPath") as mock_bids_path:
            mock_bids_path.return_value.match.return_value = []
            with pytest.raises(RuntimeError, match="No EEG runs loaded"):
                dataset._get_single_subject_data(1)

    def test_multiple_runs_indexed_correctly(self, mock_bids_root, mock_raw):
        dataset = NBIDSFDataset(bids_root=str(mock_bids_root))

        path1 = MagicMock(); path1.fpath.suffix = ".vhdr"
        path2 = MagicMock(); path2.fpath.suffix = ".vhdr"

        with patch("neurobids_flow.moabb_wrapper.BIDSPath") as mock_bids_path, \
             patch("neurobids_flow.moabb_wrapper.read_raw_bids", return_value=mock_raw):
            mock_bids_path.return_value.match.return_value = [path1, path2]
            result = dataset._get_single_subject_data(1)

        assert "run_0" in result["session_01"]
        assert "run_1" in result["session_01"]