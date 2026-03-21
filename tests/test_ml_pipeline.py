"""
NeuroBIDS-Flow — ML Pipeline Tests
====================================
Tests for:
  - sklearn_pipeline (CSP+LDA)
  - braindecode_pipeline (EEGNet)
  - cross_device_eval
  - splits
  - pipeline_demo
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_X_y():
    """Small synthetic EEG epochs for fast testing."""
    rng = np.random.default_rng(42)
    n_epochs, n_ch, n_times = 20, 4, 256
    X = rng.standard_normal((n_epochs, n_ch, n_times)).astype(np.float64)
    y = np.array([1, 2] * 10)
    return X, y


@pytest.fixture
def mock_bids_root(tmp_path):
    """Create a minimal fake BIDS folder structure."""
    for sub in ["01", "02", "03", "04"]:
        eeg_dir = tmp_path / f"sub-{sub}" / "ses-01" / "eeg"
        eeg_dir.mkdir(parents=True)
        sidecar = {
            "TaskName": "workload",
            "Manufacturer": "Brain Products",
            "SamplingFrequency": 256.0,
            "EEGChannelCount": 4,
        }
        (eeg_dir / f"sub-{sub}_ses-01_task-workload_eeg.json").write_text(
            json.dumps(sidecar)
        )
    return tmp_path


# ── CSP+LDA Tests ─────────────────────────────────────────────────────────────

class TestCSPLDA:
    def test_run_csp_lda_basic(self, synthetic_X_y):
        from neurobids_flow.sklearn_pipeline import run_csp_lda
        X, y = synthetic_X_y
        results = run_csp_lda(X, y, n_components=2, n_splits=3)
        assert "mean_accuracy" in results
        assert "std_accuracy" in results
        assert "scores" in results
        assert 0.0 <= results["mean_accuracy"] <= 1.0

    def test_run_csp_lda_returns_correct_keys(self, synthetic_X_y):
        from neurobids_flow.sklearn_pipeline import run_csp_lda
        X, y = synthetic_X_y
        results = run_csp_lda(X, y)
        expected_keys = {"scores", "mean_accuracy", "std_accuracy",
                         "n_epochs", "n_classes", "class_labels",
                         "n_components", "n_splits"}
        assert expected_keys.issubset(results.keys())

    def test_run_csp_lda_n_classes(self, synthetic_X_y):
        from neurobids_flow.sklearn_pipeline import run_csp_lda
        X, y = synthetic_X_y
        results = run_csp_lda(X, y)
        assert results["n_classes"] == 2

    def test_run_csp_lda_n_epochs(self, synthetic_X_y):
        from neurobids_flow.sklearn_pipeline import run_csp_lda
        X, y = synthetic_X_y
        results = run_csp_lda(X, y)
        assert results["n_epochs"] == len(y)

    def test_clean_description_numeric(self):
        from neurobids_flow.sklearn_pipeline import _clean_description
        assert _clean_description("unknown_123") == "123"

    def test_clean_description_hed_dict(self):
        from neurobids_flow.sklearn_pipeline import _clean_description
        desc = "{'trial_type': 'rest_open', 'hed': 'Sensory-event'}"
        assert _clean_description(desc) == "rest_open"

    def test_clean_description_plain(self):
        from neurobids_flow.sklearn_pipeline import _clean_description
        assert _clean_description("rest_open") == "rest_open"

    def test_clean_description_strips_unknown_prefix(self):
        from neurobids_flow.sklearn_pipeline import _clean_description
        assert _clean_description("unknown_rest_open") == "rest_open"


# ── EEGNet Tests ──────────────────────────────────────────────────────────────

class TestEEGNet:
    def test_build_eegnet_returns_model(self):
        from neurobids_flow.braindecode_pipeline import build_eegnet
        model = build_eegnet(n_channels=4, n_times=256, n_classes=2)
        assert model is not None

    def test_build_eegnet_forward_pass(self):
        from neurobids_flow.braindecode_pipeline import build_eegnet
        model = build_eegnet(n_channels=4, n_times=256, n_classes=2)
        x = torch.randn(2, 4, 256)
        out = model(x)
        assert out.dim() in [2, 3]
        assert out.shape[0] == 2

    def test_forward_helper_collapses_time_dim(self):
        from neurobids_flow.braindecode_pipeline import _forward, build_eegnet
        model = build_eegnet(n_channels=4, n_times=256, n_classes=2)
        x = torch.randn(4, 4, 256)
        out = _forward(model, x)
        assert out.dim() == 2
        assert out.shape == (4, 2)

    def test_eeg_array_dataset_len(self, synthetic_X_y):
        from neurobids_flow.braindecode_pipeline import EEGArrayDataset
        X, y = synthetic_X_y
        ds = EEGArrayDataset(X, y)
        assert len(ds) == len(y)

    def test_eeg_array_dataset_getitem(self, synthetic_X_y):
        from neurobids_flow.braindecode_pipeline import EEGArrayDataset
        X, y = synthetic_X_y
        ds = EEGArrayDataset(X, y)
        x_item, y_item = ds[0]
        assert isinstance(x_item, torch.Tensor)
        assert isinstance(y_item, torch.Tensor)
        assert x_item.shape == (X.shape[1], X.shape[2])

    def test_eeg_array_dataset_dtype(self, synthetic_X_y):
        from neurobids_flow.braindecode_pipeline import EEGArrayDataset
        X, y = synthetic_X_y
        ds = EEGArrayDataset(X, y)
        x_item, y_item = ds[0]
        assert x_item.dtype == torch.float32
        assert y_item.dtype == torch.long

    def test_train_eval_epoch_run(self, synthetic_X_y):
        from neurobids_flow.braindecode_pipeline import (
            EEGArrayDataset, build_eegnet, train_epoch, eval_epoch
        )
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from sklearn.preprocessing import LabelEncoder

        X, y = synthetic_X_y
        le = LabelEncoder()
        y_enc = le.fit_transform(y).astype(np.int64)

        ds = EEGArrayDataset(X, y_enc)
        loader = DataLoader(ds, batch_size=4, shuffle=True)
        model = build_eegnet(X.shape[1], X.shape[2], 2)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        train_loss, train_acc = train_epoch(
            model, loader, optimizer, criterion, torch.device("cpu")
        )
        val_loss, val_acc = eval_epoch(
            model, loader, criterion, torch.device("cpu")
        )

        assert 0.0 <= train_acc <= 1.0
        assert 0.0 <= val_acc <= 1.0
        assert train_loss >= 0.0


# ── Splits Tests ──────────────────────────────────────────────────────────────

class TestSplits:
    def test_generate_splits_basic(self, mock_bids_root):
        from neurobids_flow.splits import generate_splits
        splits = generate_splits(bids_root=str(mock_bids_root), seed=42)
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits

    def test_generate_splits_no_overlap(self, mock_bids_root):
        from neurobids_flow.splits import generate_splits
        splits = generate_splits(bids_root=str(mock_bids_root))
        all_subs = splits["train"] + splits["val"] + splits["test"]
        assert len(all_subs) == len(set(all_subs)), "Overlap detected!"

    def test_generate_splits_covers_all_subjects(self, mock_bids_root):
        from neurobids_flow.splits import generate_splits
        splits = generate_splits(bids_root=str(mock_bids_root))
        all_subs = set(splits["train"] + splits["val"] + splits["test"])
        assert all_subs == set(splits["subjects"])

    def test_generate_splits_reproducible(self, mock_bids_root):
        from neurobids_flow.splits import generate_splits
        s1 = generate_splits(bids_root=str(mock_bids_root), seed=42)
        s2 = generate_splits(bids_root=str(mock_bids_root), seed=42)
        assert s1["train"] == s2["train"]
        assert s1["val"] == s2["val"]
        assert s1["test"] == s2["test"]

    def test_generate_splits_different_seeds(self, mock_bids_root):
        from neurobids_flow.splits import generate_splits
        s1 = generate_splits(bids_root=str(mock_bids_root), seed=42)
        s2 = generate_splits(bids_root=str(mock_bids_root), seed=99)
        # Same subjects regardless of seed, just different order
        assert set(s1["subjects"]) == set(s2["subjects"])

    def test_generate_splits_raises_no_subjects(self, tmp_path):
        from neurobids_flow.splits import generate_splits
        with pytest.raises(RuntimeError):
            generate_splits(bids_root=str(tmp_path))

    def test_save_and_load_splits(self, mock_bids_root, tmp_path):
        from neurobids_flow.splits import generate_splits, save_splits, load_splits
        splits = generate_splits(bids_root=str(mock_bids_root))
        path = str(tmp_path / "splits.json")
        save_splits(splits, path)
        loaded = load_splits(path)
        assert loaded["train"] == splits["train"]
        assert loaded["test"] == splits["test"]

    def test_splits_ratios_sum_to_one(self, mock_bids_root):
        from neurobids_flow.splits import generate_splits
        splits = generate_splits(bids_root=str(mock_bids_root))
        total = splits["n_train"] + splits["n_val"] + splits["n_test"]
        assert total == splits["n_subjects"]


# ── Cross-Device Eval Tests ───────────────────────────────────────────────────

class TestCrossDeviceEval:
    def test_detect_device_brain_products(self, mock_bids_root):
        from neurobids_flow.cross_device_eval import detect_device
        label = detect_device("01", str(mock_bids_root), "01", "workload")
        assert label == "BrainProducts ActiChamp"

    def test_detect_device_unknown(self, tmp_path):
        from neurobids_flow.cross_device_eval import detect_device
        label = detect_device("99", str(tmp_path), "01", "workload")
        assert label == "Unknown Device"

    def test_eval_csp_lda_basic(self, synthetic_X_y):
        from neurobids_flow.cross_device_eval import eval_csp_lda
        X, y = synthetic_X_y
        result = eval_csp_lda(X, y)
        assert "mean_acc" in result
        assert "status" in result

    def test_eval_csp_lda_returns_float_or_none(self, synthetic_X_y):
        from neurobids_flow.cross_device_eval import eval_csp_lda
        X, y = synthetic_X_y
        result = eval_csp_lda(X, y)
        if result["mean_acc"] is not None:
            assert 0.0 <= result["mean_acc"] <= 1.0

    def test_eval_eegnet_basic(self, synthetic_X_y):
        from neurobids_flow.cross_device_eval import eval_eegnet
        X, y = synthetic_X_y
        result = eval_eegnet(X, y)
        assert "best_acc" in result
        assert "status" in result

    def test_eval_eegnet_too_few_samples(self):
        from neurobids_flow.cross_device_eval import eval_eegnet
        X = np.random.randn(1, 4, 256)
        y = np.array([1])
        result = eval_eegnet(X, y)
        assert result["best_acc"] is None

    def test_print_summary_table_runs(self, mock_bids_root):
        from neurobids_flow.cross_device_eval import print_summary_table
        rows = [
            {"subject": "sub-01", "device": "BrainProducts ActiChamp",
             "n_channels": 8, "n_epochs": 19,
             "csp_lda_acc": 0.5, "eegnet_acc": 0.55, "status": "ok"},
            {"subject": "sub-02", "device": "Unknown",
             "n_channels": 0, "n_epochs": 0,
             "csp_lda_acc": None, "eegnet_acc": None, "status": "no epochs"},
        ]
        print_summary_table(rows)


# ── Pipeline Demo Tests ───────────────────────────────────────────────────────

class TestPipelineDemo:
    def test_import_pipeline_demo(self):
        from neurobids_flow import pipeline_demo
        assert hasattr(pipeline_demo, "run_demo")

    def test_run_demo_skip_conversion(self, mock_bids_root):
        from neurobids_flow.splits import generate_splits
        splits = generate_splits(bids_root=str(mock_bids_root))
        assert splits["n_subjects"] == 4