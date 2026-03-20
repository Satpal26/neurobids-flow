# test_torch_dataset.py
# NeuroBIDS-Flow — PyTorch Dataset wrapper test suite
# Tests the NeuroBIDSFlowTorchDataset class
# Run: uv run python -m pytest tests/test_torch_dataset.py -v
#
# NeuroBIDS-Flow | NTU Singapore BCI Lab

import pytest
import numpy as np
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch

from neurobids_flow.torch_dataset import NeuroBIDSFlowTorchDataset
from neurobids_flow.moabb_wrapper import NEUROBIDS_EVENTS


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

N_EPOCHS    = 6
N_CHANNELS  = 8
N_TIMES     = 897   # matches 3.5s window at 256 Hz


@pytest.fixture
def mock_bids_root(tmp_path):
    """Minimal BIDS folder structure for path resolution."""
    eeg_dir = tmp_path / "sub-01" / "ses-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    (eeg_dir / "sub-01_ses-01_task-workload_eeg.vhdr").write_text("")
    return tmp_path


@pytest.fixture
def fake_epochs_data():
    """Returns (X_np, y_np) matching realistic EEG epoch shapes."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1e-6, (N_EPOCHS, N_CHANNELS, N_TIMES))
    y = np.array([1, 2, 1, 2, 1, 2])
    return X, y


@pytest.fixture
def mock_dataset(mock_bids_root, fake_epochs_data):
    """
    A fully initialised NeuroBIDSFlowTorchDataset with mocked internals.
    Patches _load_all to return fake tensors — no real BIDS/MNE needed.
    """
    X_np, y_np = fake_epochs_data
    X_t = torch.tensor(X_np, dtype=torch.float32)
    y_t = torch.tensor(y_np, dtype=torch.long)

    with patch.object(NeuroBIDSFlowTorchDataset, "_load_all", return_value=(X_t, y_t)):
        ds = NeuroBIDSFlowTorchDataset(
            bids_root=str(mock_bids_root),
            task="workload",
        )
    return ds


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Initialisation Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestTorchDatasetInit:

    def test_init_succeeds(self, mock_dataset):
        assert mock_dataset is not None

    def test_default_task(self, mock_bids_root, fake_epochs_data):
        X_t = torch.tensor(fake_epochs_data[0], dtype=torch.float32)
        y_t = torch.tensor(fake_epochs_data[1], dtype=torch.long)
        with patch.object(NeuroBIDSFlowTorchDataset, "_load_all", return_value=(X_t, y_t)):
            ds = NeuroBIDSFlowTorchDataset(bids_root=str(mock_bids_root))
        assert ds.task == "workload"

    def test_custom_task(self, mock_bids_root, fake_epochs_data):
        X_t = torch.tensor(fake_epochs_data[0], dtype=torch.float32)
        y_t = torch.tensor(fake_epochs_data[1], dtype=torch.long)
        with patch.object(NeuroBIDSFlowTorchDataset, "_load_all", return_value=(X_t, y_t)):
            ds = NeuroBIDSFlowTorchDataset(bids_root=str(mock_bids_root), task="ssvep")
        assert ds.task == "ssvep"

    def test_default_interval(self, mock_dataset):
        assert mock_dataset.tmin == -0.5
        assert mock_dataset.tmax == 3.0

    def test_custom_interval(self, mock_bids_root, fake_epochs_data):
        X_t = torch.tensor(fake_epochs_data[0], dtype=torch.float32)
        y_t = torch.tensor(fake_epochs_data[1], dtype=torch.long)
        with patch.object(NeuroBIDSFlowTorchDataset, "_load_all", return_value=(X_t, y_t)):
            ds = NeuroBIDSFlowTorchDataset(
                bids_root=str(mock_bids_root), tmin=0.0, tmax=4.0
            )
        assert ds.tmin == 0.0
        assert ds.tmax == 4.0

    def test_default_dtype_is_float32(self, mock_dataset):
        assert mock_dataset.dtype == torch.float32

    def test_default_events_loaded(self, mock_dataset):
        assert mock_dataset._events == NEUROBIDS_EVENTS

    def test_custom_events(self, mock_bids_root, fake_epochs_data):
        X_t = torch.tensor(fake_epochs_data[0], dtype=torch.float32)
        y_t = torch.tensor(fake_epochs_data[1], dtype=torch.long)
        custom = {"cognitive_low": 0, "cognitive_high": 1}
        with patch.object(NeuroBIDSFlowTorchDataset, "_load_all", return_value=(X_t, y_t)):
            ds = NeuroBIDSFlowTorchDataset(
                bids_root=str(mock_bids_root), events=custom
            )
        assert ds._events == custom


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PyTorch Dataset Interface Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPyTorchDatasetInterface:

    def test_len_returns_correct_count(self, mock_dataset):
        assert len(mock_dataset) == N_EPOCHS

    def test_getitem_returns_tuple(self, mock_dataset):
        item = mock_dataset[0]
        assert isinstance(item, tuple)
        assert len(item) == 2

    def test_getitem_X_is_tensor(self, mock_dataset):
        X, y = mock_dataset[0]
        assert isinstance(X, torch.Tensor)

    def test_getitem_y_is_tensor(self, mock_dataset):
        X, y = mock_dataset[0]
        assert isinstance(y, torch.Tensor)

    def test_getitem_X_shape(self, mock_dataset):
        X, y = mock_dataset[0]
        assert X.shape == (N_CHANNELS, N_TIMES)

    def test_getitem_y_is_scalar(self, mock_dataset):
        X, y = mock_dataset[0]
        assert y.ndim == 0

    def test_all_indices_accessible(self, mock_dataset):
        for i in range(len(mock_dataset)):
            X, y = mock_dataset[i]
            assert X.shape == (N_CHANNELS, N_TIMES)

    def test_X_dtype_is_float32(self, mock_dataset):
        X, _ = mock_dataset[0]
        assert X.dtype == torch.float32

    def test_y_dtype_is_long(self, mock_dataset):
        _, y = mock_dataset[0]
        assert y.dtype == torch.long


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Properties Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestProperties:

    def test_n_channels(self, mock_dataset):
        assert mock_dataset.n_channels == N_CHANNELS

    def test_n_times(self, mock_dataset):
        assert mock_dataset.n_times == N_TIMES

    def test_n_classes(self, mock_dataset):
        # fake y has values [1, 2] → 2 classes
        assert mock_dataset.n_classes == 2

    def test_class_labels_are_strings(self, mock_dataset):
        labels = mock_dataset.class_labels
        assert isinstance(labels, list)
        assert all(isinstance(l, str) for l in labels)

    def test_X_shape_full(self, mock_dataset):
        assert mock_dataset.X.shape == (N_EPOCHS, N_CHANNELS, N_TIMES)

    def test_y_shape_full(self, mock_dataset):
        assert mock_dataset.y.shape == (N_EPOCHS,)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DataLoader Compatibility Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataLoaderCompatibility:

    def test_dataloader_iterates(self, mock_dataset):
        from torch.utils.data import DataLoader
        loader = DataLoader(mock_dataset, batch_size=2, shuffle=False)
        batches = list(loader)
        assert len(batches) == N_EPOCHS // 2

    def test_batch_X_shape(self, mock_dataset):
        from torch.utils.data import DataLoader
        loader = DataLoader(mock_dataset, batch_size=2, shuffle=False)
        X_batch, y_batch = next(iter(loader))
        assert X_batch.shape == (2, N_CHANNELS, N_TIMES)

    def test_batch_y_shape(self, mock_dataset):
        from torch.utils.data import DataLoader
        loader = DataLoader(mock_dataset, batch_size=2, shuffle=False)
        X_batch, y_batch = next(iter(loader))
        assert y_batch.shape == (2,)

    def test_shuffle_works(self, mock_dataset):
        from torch.utils.data import DataLoader
        loader = DataLoader(mock_dataset, batch_size=N_EPOCHS, shuffle=True)
        X_batch, y_batch = next(iter(loader))
        assert X_batch.shape[0] == N_EPOCHS

    def test_random_split(self, mock_dataset):
        from torch.utils.data import random_split
        n = len(mock_dataset)
        n_train = int(0.8 * n)
        n_val = n - n_train
        train_set, val_set = random_split(mock_dataset, [n_train, n_val])
        assert len(train_set) == n_train
        assert len(val_set) == n_val


# ═══════════════════════════════════════════════════════════════════════════════
# 5. _load_all Error Handling Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadAllErrorHandling:

    def test_raises_when_no_epochs_extracted(self, mock_bids_root):
        """Should raise RuntimeError if no epochs can be loaded."""
        with patch(
            "neurobids_flow.torch_dataset.NBIDSFDataset"
        ) as MockDataset:
            mock_instance = MagicMock()
            mock_instance.subject_list = [1]
            mock_instance._sessions = ["01"]

            # Raw with NO matching annotations
            mock_raw = MagicMock()
            mock_raw.annotations.description = []
            mock_instance._get_single_subject_data.return_value = {
                "session_01": {"run_0": mock_raw}
            }
            MockDataset.return_value = mock_instance

            with pytest.raises(RuntimeError, match="No epochs could be extracted"):
                NeuroBIDSFlowTorchDataset(bids_root=str(mock_bids_root))