"""
Tests for NeuroBIDS-Flow SSVEPFlow (Target 2)
Covers: CCA, FBCCA, TRCA, Evaluator, Config, Pipeline, Visualizer, Benchmark
"""

from __future__ import annotations

import json
from os import path
import tempfile
from pathlib import Path

from matplotlib import path
import numpy as np
import pytest
import yaml
import matplotlib

from neurobids_flow.ssvep.config import SSVEPConfig, load_ssvep_config
matplotlib.use("Agg")  # non-interactive backend for tests


# ── Shared fixtures ────────────────────────────────────────────────────────────
SFREQ = 256.0
STIM_FREQS = [6.0, 8.0, 10.0, 12.0]
N_CLASSES = len(STIM_FREQS)
N_CH = 8
N_TIMES = int(SFREQ * 2)   # 2 seconds
N_EPOCHS = 20               # 5 per class


def _make_X_y(
    n_epochs: int = N_EPOCHS,
    n_ch: int = N_CH,
    n_times: int = N_TIMES,
    n_classes: int = N_CLASSES,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic (X, y) — small SSVEP-like dataset."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, n_times / SFREQ, n_times)
    X = rng.normal(0, 1e-6, (n_epochs, n_ch, n_times))
    y = np.array([i % n_classes for i in range(n_epochs)])
    # Add weak SSVEP signal so classifiers can latch onto something
    for i, label in enumerate(y):
        freq = STIM_FREQS[label]
        X[i, -1] += 5e-6 * np.sin(2 * np.pi * freq * t)
        X[i, -2] += 3e-6 * np.sin(2 * np.pi * freq * t)
    return X, y


@pytest.fixture
def Xy():
    return _make_X_y()


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ══════════════════════════════════════════════════════════════════════════════
# 1. CCA
# ══════════════════════════════════════════════════════════════════════════════
class TestCCA:
    def test_import(self):
        from neurobids_flow.ssvep.cca import CCA
        assert CCA is not None

    def test_init(self):
        from neurobids_flow.ssvep.cca import CCA
        clf = CCA(stim_freqs=STIM_FREQS, sfreq=SFREQ)
        assert clf.n_harmonics == 3
        assert len(clf.stim_freqs) == N_CLASSES

    def test_build_reference_shape(self):
        from neurobids_flow.ssvep.cca import CCA
        clf = CCA(stim_freqs=STIM_FREQS, sfreq=SFREQ, n_harmonics=3)
        ref = clf._build_reference(6.0, N_TIMES)
        assert ref.shape == (6, N_TIMES)   # 2 * n_harmonics rows

    def test_predict_shape(self, Xy):
        from neurobids_flow.ssvep.cca import CCA
        X, y = Xy
        clf = CCA(stim_freqs=STIM_FREQS, sfreq=SFREQ)
        preds = clf.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_values_in_range(self, Xy):
        from neurobids_flow.ssvep.cca import CCA
        X, y = Xy
        clf = CCA(stim_freqs=STIM_FREQS, sfreq=SFREQ)
        preds = clf.predict(X)
        assert set(preds).issubset(set(range(N_CLASSES)))

    def test_predict_proba_shape(self, Xy):
        from neurobids_flow.ssvep.cca import CCA
        X, y = Xy
        clf = CCA(stim_freqs=STIM_FREQS, sfreq=SFREQ)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), N_CLASSES)

    def test_score_returns_float(self, Xy):
        from neurobids_flow.ssvep.cca import CCA
        X, y = Xy
        clf = CCA(stim_freqs=STIM_FREQS, sfreq=SFREQ)
        acc = clf.score(X, y)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_repr(self):
        from neurobids_flow.ssvep.cca import CCA
        clf = CCA(stim_freqs=STIM_FREQS, sfreq=SFREQ)
        assert "CCA" in repr(clf)

    def test_single_epoch(self):
        from neurobids_flow.ssvep.cca import CCA
        X, y = _make_X_y(n_epochs=1)
        clf = CCA(stim_freqs=STIM_FREQS, sfreq=SFREQ)
        preds = clf.predict(X)
        assert len(preds) == 1


# ══════════════════════════════════════════════════════════════════════════════
# 2. FBCCA
# ══════════════════════════════════════════════════════════════════════════════
class TestFBCCA:
    def test_import(self):
        from neurobids_flow.ssvep.fbcca import FBCCA
        assert FBCCA is not None

    def test_init_default_subbands(self):
        from neurobids_flow.ssvep.fbcca import FBCCA, DEFAULT_SUBBANDS
        clf = FBCCA(stim_freqs=STIM_FREQS, sfreq=SFREQ)
        assert len(clf.subbands) == len(DEFAULT_SUBBANDS)

    def test_weights_shape(self):
        from neurobids_flow.ssvep.fbcca import FBCCA
        clf = FBCCA(stim_freqs=STIM_FREQS, sfreq=SFREQ)
        assert len(clf._weights) == len(clf.subbands)

    def test_weights_positive(self):
        from neurobids_flow.ssvep.fbcca import FBCCA
        clf = FBCCA(stim_freqs=STIM_FREQS, sfreq=SFREQ)
        assert (clf._weights > 0).all()

    def test_bandpass_preserves_shape(self):
        from neurobids_flow.ssvep.fbcca import FBCCA
        clf = FBCCA(stim_freqs=STIM_FREQS, sfreq=SFREQ)
        X = np.random.randn(N_CH, N_TIMES)
        out = clf._bandpass(X, 6, 40)
        assert out.shape == X.shape

    def test_predict_shape(self, Xy):
        from neurobids_flow.ssvep.fbcca import FBCCA
        X, y = Xy
        clf = FBCCA(stim_freqs=STIM_FREQS, sfreq=SFREQ)
        preds = clf.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_proba_shape(self, Xy):
        from neurobids_flow.ssvep.fbcca import FBCCA
        X, y = Xy
        clf = FBCCA(stim_freqs=STIM_FREQS, sfreq=SFREQ)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), N_CLASSES)

    def test_score_returns_float(self, Xy):
        from neurobids_flow.ssvep.fbcca import FBCCA
        X, y = Xy
        clf = FBCCA(stim_freqs=STIM_FREQS, sfreq=SFREQ)
        acc = clf.score(X, y)
        assert 0.0 <= acc <= 1.0

    def test_custom_subbands(self, Xy):
        from neurobids_flow.ssvep.fbcca import FBCCA
        X, y = Xy
        clf = FBCCA(stim_freqs=STIM_FREQS, sfreq=SFREQ, subbands=[(6, 40), (14, 40)])
        preds = clf.predict(X)
        assert len(preds) == len(X)

    def test_repr(self):
        from neurobids_flow.ssvep.fbcca import FBCCA
        clf = FBCCA(stim_freqs=STIM_FREQS, sfreq=SFREQ)
        assert "FBCCA" in repr(clf)


# ══════════════════════════════════════════════════════════════════════════════
# 3. TRCA
# ══════════════════════════════════════════════════════════════════════════════
class TestTRCA:
    def test_import(self):
        from neurobids_flow.ssvep.trca import TRCA
        assert TRCA is not None

    def test_not_fitted_raises(self):
        from neurobids_flow.ssvep.trca import TRCA
        clf = TRCA(stim_freqs=STIM_FREQS, sfreq=SFREQ)
        X, _ = _make_X_y(n_epochs=4)
        with pytest.raises(RuntimeError):
            clf.predict(X)

    def test_fit_returns_self(self, Xy):
        from neurobids_flow.ssvep.trca import TRCA
        X, y = Xy
        clf = TRCA(stim_freqs=STIM_FREQS, sfreq=SFREQ)
        result = clf.fit(X, y)
        assert result is clf

    def test_fitted_flag(self, Xy):
        from neurobids_flow.ssvep.trca import TRCA
        X, y = Xy
        clf = TRCA(stim_freqs=STIM_FREQS, sfreq=SFREQ)
        clf.fit(X, y)
        assert clf._fitted is True

    def test_filters_stored_per_class(self, Xy):
        from neurobids_flow.ssvep.trca import TRCA
        X, y = Xy
        clf = TRCA(stim_freqs=STIM_FREQS, sfreq=SFREQ)
        clf.fit(X, y)
        assert len(clf._filters) == N_CLASSES
        assert len(clf._templates) == N_CLASSES

    def test_predict_shape(self, Xy):
        from neurobids_flow.ssvep.trca import TRCA
        X, y = Xy
        clf = TRCA(stim_freqs=STIM_FREQS, sfreq=SFREQ)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_values_in_range(self, Xy):
        from neurobids_flow.ssvep.trca import TRCA
        X, y = Xy
        clf = TRCA(stim_freqs=STIM_FREQS, sfreq=SFREQ)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert set(preds).issubset(set(range(N_CLASSES)))

    def test_predict_proba_shape(self, Xy):
        from neurobids_flow.ssvep.trca import TRCA
        X, y = Xy
        clf = TRCA(stim_freqs=STIM_FREQS, sfreq=SFREQ)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), N_CLASSES)

    def test_score_returns_float(self, Xy):
        from neurobids_flow.ssvep.trca import TRCA
        X, y = Xy
        clf = TRCA(stim_freqs=STIM_FREQS, sfreq=SFREQ)
        clf.fit(X, y)
        acc = clf.score(X, y)
        assert 0.0 <= acc <= 1.0

    def test_ensemble_false(self, Xy):
        from neurobids_flow.ssvep.trca import TRCA
        X, y = Xy
        clf = TRCA(stim_freqs=STIM_FREQS, sfreq=SFREQ, ensemble=False)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert len(preds) == len(X)

    def test_repr_not_fitted(self):
        from neurobids_flow.ssvep.trca import TRCA
        clf = TRCA(stim_freqs=STIM_FREQS, sfreq=SFREQ)
        assert "not fitted" in repr(clf)

    def test_repr_fitted(self, Xy):
        from neurobids_flow.ssvep.trca import TRCA
        X, y = Xy
        clf = TRCA(stim_freqs=STIM_FREQS, sfreq=SFREQ)
        clf.fit(X, y)
        assert "fitted" in repr(clf)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Evaluator
# ══════════════════════════════════════════════════════════════════════════════
class TestEvaluator:
    def test_import(self):
        from neurobids_flow.ssvep.evaluator import SSVEPEvaluator
        assert SSVEPEvaluator is not None

    def test_itr_bpt_chance(self):
        from neurobids_flow.ssvep.evaluator import itr_bits_per_trial
        bpt = itr_bits_per_trial(n_classes=4, accuracy=0.25)
        assert bpt == pytest.approx(0.0, abs=0.01)

    def test_itr_bpt_perfect(self):
        from neurobids_flow.ssvep.evaluator import itr_bits_per_trial
        bpt = itr_bits_per_trial(n_classes=4, accuracy=1.0)
        assert bpt == pytest.approx(2.0, abs=0.01)   # log2(4) = 2

    def test_itr_bpm_positive(self):
        from neurobids_flow.ssvep.evaluator import itr_bits_per_minute
        bpm = itr_bits_per_minute(4, 0.8, epoch_duration=2.0, gap_duration=0.5)
        assert bpm >= 0.0

    def test_confusion_matrix_shape(self):
        from neurobids_flow.ssvep.evaluator import confusion_matrix
        y_true = np.array([0, 1, 2, 3, 0, 1])
        y_pred = np.array([0, 1, 2, 3, 1, 0])
        cm = confusion_matrix(y_true, y_pred, n_classes=4)
        assert cm.shape == (4, 4)

    def test_confusion_matrix_diagonal(self):
        from neurobids_flow.ssvep.evaluator import confusion_matrix
        y = np.array([0, 1, 2, 3])
        cm = confusion_matrix(y, y, n_classes=4)
        assert np.all(cm == np.eye(4))

    def test_precision_recall_f1_perfect(self):
        from neurobids_flow.ssvep.evaluator import confusion_matrix, precision_recall_f1
        y = np.array([0, 1, 2, 3])
        cm = confusion_matrix(y, y, n_classes=4)
        prec, rec, f1 = precision_recall_f1(cm)
        assert np.allclose(prec, 1.0)
        assert np.allclose(rec, 1.0)
        assert np.allclose(f1, 1.0)

    def test_evaluate_cca_returns_result(self, Xy):
        from neurobids_flow.ssvep.evaluator import SSVEPEvaluator
        from neurobids_flow.ssvep.cca import CCA
        X, y = Xy
        ev = SSVEPEvaluator(n_classes=N_CLASSES, epoch_duration=2.0)
        clf = CCA(stim_freqs=STIM_FREQS, sfreq=SFREQ)
        result = ev.evaluate(clf, X, y, method_name="CCA", run_cv=False)
        assert result.accuracy >= 0.0
        assert result.itr_bpm >= 0.0
        assert result.confusion_matrix.shape == (N_CLASSES, N_CLASSES)

    def test_evaluate_with_cv(self, Xy):
        from neurobids_flow.ssvep.evaluator import SSVEPEvaluator
        from neurobids_flow.ssvep.cca import CCA
        X, y = Xy
        ev = SSVEPEvaluator(n_classes=N_CLASSES, epoch_duration=2.0, n_splits=3)
        clf = CCA(stim_freqs=STIM_FREQS, sfreq=SFREQ)
        result = ev.evaluate(clf, X, y, run_cv=True)
        assert len(result.cv_fold_accuracies) >= 2

    def test_eval_result_to_dict(self, Xy):
        from neurobids_flow.ssvep.evaluator import SSVEPEvaluator
        from neurobids_flow.ssvep.cca import CCA
        X, y = Xy
        ev = SSVEPEvaluator(n_classes=N_CLASSES)
        clf = CCA(stim_freqs=STIM_FREQS, sfreq=SFREQ)
        result = ev.evaluate(clf, X, y, run_cv=False)
        d = result.to_dict()
        assert "accuracy" in d
        assert "itr_bpm" in d

    def test_print_report_runs(self, Xy, capsys):
        from neurobids_flow.ssvep.evaluator import SSVEPEvaluator
        from neurobids_flow.ssvep.cca import CCA
        X, y = Xy
        ev = SSVEPEvaluator(n_classes=N_CLASSES)
        clf = CCA(stim_freqs=STIM_FREQS, sfreq=SFREQ)
        result = ev.evaluate(clf, X, y, run_cv=False)
        ev.print_report(result, stim_freqs=STIM_FREQS)
        captured = capsys.readouterr()
        assert "Accuracy" in captured.out


# ══════════════════════════════════════════════════════════════════════════════
# 5. Config
# ══════════════════════════════════════════════════════════════════════════════
class TestConfig:
    def test_import(self):
        from neurobids_flow.ssvep.config import SSVEPConfig
        assert SSVEPConfig is not None

    def test_default_stim_freqs(self):
        from neurobids_flow.ssvep.config import SSVEPConfig
        cfg = SSVEPConfig()
        assert cfg.stim_freqs == [6.0, 8.0, 10.0, 12.0]

    def test_default_methods(self):
        from neurobids_flow.ssvep.config import SSVEPConfig
        cfg = SSVEPConfig()
        assert "cca" in cfg.methods
        assert "fbcca" in cfg.methods
        assert "trca" in cfg.methods

    def test_to_dict(self):
        from neurobids_flow.ssvep.config import SSVEPConfig
        cfg = SSVEPConfig()
        d = cfg.to_dict()
        assert "stim_freqs" in d
        assert "methods" in d

    def test_save_and_load(self, tmp_dir):
        from neurobids_flow.ssvep.config import SSVEPConfig, load_ssvep_config
        cfg = SSVEPConfig()
        cfg.stim_freqs = [6.0, 8.0]
        path = tmp_dir / "test_cfg.yaml"
        # Convert to dict manually stripping tuples
        import yaml
        d = cfg.to_dict()
        def _fix(obj):
            if isinstance(obj, tuple):
                return list(obj)
            if isinstance(obj, dict):
                return {k: _fix(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_fix(v) for v in obj]
            return obj
        with open(path, "w") as f:
            yaml.dump(_fix(d), f)
        loaded = load_ssvep_config(path)
        assert loaded.stim_freqs == [6.0, 8.0]

    def test_load_missing_file_raises(self, tmp_dir):
        from neurobids_flow.ssvep.config import load_ssvep_config
        with pytest.raises(FileNotFoundError):
            load_ssvep_config(tmp_dir / "nonexistent.yaml")

    def test_from_dict_partial(self):
        from neurobids_flow.ssvep.config import SSVEPConfig
        cfg = SSVEPConfig.from_dict({"stim_freqs": [6.0, 8.0], "task": "test_task"})
        assert cfg.stim_freqs == [6.0, 8.0]
        assert cfg.task == "test_task"

    def test_filter_config_defaults(self):
        from neurobids_flow.ssvep.config import SSVEPConfig
        cfg = SSVEPConfig()
        assert cfg.filters.lowcut == 1.0
        assert cfg.filters.highcut == 40.0

    def test_epoch_config_defaults(self):
        from neurobids_flow.ssvep.config import SSVEPConfig
        cfg = SSVEPConfig()
        assert cfg.epochs.tmin == 0.0
        assert cfg.epochs.tmax == 2.0


# ══════════════════════════════════════════════════════════════════════════════
# 6. Visualizer (no display — just check files are created)
# ══════════════════════════════════════════════════════════════════════════════
class TestVisualizer:
    def _make_results(self):
        from neurobids_flow.ssvep.evaluator import SSVEPEvaluator
        from neurobids_flow.ssvep.cca import CCA
        from neurobids_flow.ssvep.fbcca import FBCCA
        X, y = _make_X_y()
        ev = SSVEPEvaluator(n_classes=N_CLASSES, epoch_duration=2.0)
        results = {}
        for name, clf in [("CCA", CCA(STIM_FREQS, SFREQ)), ("FBCCA", FBCCA(STIM_FREQS, SFREQ))]:
            results[name] = ev.evaluate(clf, X, y, method_name=name, run_cv=False)
        return results, X

    def test_import(self):
        from neurobids_flow.ssvep.visualizer import SSVEPVisualizer
        assert SSVEPVisualizer is not None

    def test_init_creates_output_dir(self, tmp_dir):
        from neurobids_flow.ssvep.visualizer import SSVEPVisualizer
        out = tmp_dir / "plots"
        viz = SSVEPVisualizer(stim_freqs=STIM_FREQS, output_dir=out, show=False)
        assert out.exists()

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("matplotlib"),
        reason="matplotlib not installed"
    )
    def test_plot_accuracy_itr_saves_file(self, tmp_dir):
        from neurobids_flow.ssvep.visualizer import SSVEPVisualizer
        results, _ = self._make_results()
        viz = SSVEPVisualizer(stim_freqs=STIM_FREQS, output_dir=tmp_dir, show=False)
        path = viz.plot_accuracy_itr(results, filename="test_acc.png")
        assert path is not None and path.exists()

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("matplotlib"),
        reason="matplotlib not installed"
    )
    def test_plot_confusion_matrix_saves_file(self, tmp_dir):
        from neurobids_flow.ssvep.visualizer import SSVEPVisualizer
        results, _ = self._make_results()
        viz = SSVEPVisualizer(stim_freqs=STIM_FREQS, output_dir=tmp_dir, show=False)
        path = viz.plot_confusion_matrix(results["CCA"], method_name="CCA", filename="test_cm.png")
        assert path is not None and path.exists()

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("matplotlib"),
        reason="matplotlib not installed"
    )
    def test_plot_psd_saves_file(self, tmp_dir):
        from neurobids_flow.ssvep.visualizer import SSVEPVisualizer
        _, X = self._make_results()
        viz = SSVEPVisualizer(stim_freqs=STIM_FREQS, output_dir=tmp_dir, show=False)
        path = viz.plot_psd(X, sfreq=SFREQ, filename="test_psd.png")
        assert path is not None and path.exists()

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("matplotlib"),
        reason="matplotlib not installed"
    )
    def test_plot_cv_folds_saves_file(self, tmp_dir):
        from neurobids_flow.ssvep.evaluator import SSVEPEvaluator
        from neurobids_flow.ssvep.cca import CCA
        from neurobids_flow.ssvep.visualizer import SSVEPVisualizer
        X, y = _make_X_y()
        ev = SSVEPEvaluator(n_classes=N_CLASSES, n_splits=3)
        results = {"CCA": ev.evaluate(CCA(STIM_FREQS, SFREQ), X, y, run_cv=True)}
        viz = SSVEPVisualizer(stim_freqs=STIM_FREQS, output_dir=tmp_dir, show=False)
        path = viz.plot_cv_folds(results, filename="test_cv.png")
        assert path is not None and path.exists()


# ══════════════════════════════════════════════════════════════════════════════
# 7. Benchmark
# ══════════════════════════════════════════════════════════════════════════════
class TestBenchmark:
    def _make_bids_root(self, tmp_dir: Path) -> Path:
        """Create a minimal fake BIDS structure for benchmark tests."""
        bids = tmp_dir / "bids"
        for sub in ["01", "02"]:
            eeg_dir = bids / f"sub-{sub}" / "ses-01" / "eeg"
            eeg_dir.mkdir(parents=True)
            sidecar = {
                "Manufacturer": "Brain Products",
                "SamplingFrequency": 256.0,
                "TaskName": "ssvep",
            }
            with open(eeg_dir / f"sub-{sub}_ses-01_task-ssvep_eeg.json", "w") as f:
                json.dump(sidecar, f)
        return bids

    def test_import(self):
        from neurobids_flow.ssvep.benchmark import SSVEPBenchmark
        assert SSVEPBenchmark is not None

    def test_init(self, tmp_dir):
        from neurobids_flow.ssvep.benchmark import SSVEPBenchmark
        bm = SSVEPBenchmark(
            bids_root=str(tmp_dir),
            stim_freqs=STIM_FREQS,
            output_dir=str(tmp_dir / "out"),
        )
        assert bm.stim_freqs == STIM_FREQS

    def test_discover_subjects(self, tmp_dir):
        from neurobids_flow.ssvep.benchmark import SSVEPBenchmark
        bids = self._make_bids_root(tmp_dir)
        bm = SSVEPBenchmark(bids_root=str(bids), stim_freqs=STIM_FREQS)
        pairs = bm._discover_subjects()
        assert len(pairs) == 2
        assert ("01", "01") in pairs
        assert ("02", "01") in pairs

    def test_discover_subjects_empty(self, tmp_dir):
        from neurobids_flow.ssvep.benchmark import SSVEPBenchmark
        bm = SSVEPBenchmark(bids_root=str(tmp_dir), stim_freqs=STIM_FREQS)
        pairs = bm._discover_subjects()
        assert pairs == []

    def test_get_device_from_sidecar(self, tmp_dir):
        from neurobids_flow.ssvep.benchmark import SSVEPBenchmark
        bids = self._make_bids_root(tmp_dir)
        bm = SSVEPBenchmark(bids_root=str(bids), stim_freqs=STIM_FREQS)
        device = bm._get_device(str(bids), "01", "01", "ssvep")
        assert device == "Brain Products"

    def test_get_device_missing_sidecar(self, tmp_dir):
        from neurobids_flow.ssvep.benchmark import SSVEPBenchmark
        bm = SSVEPBenchmark(bids_root=str(tmp_dir), stim_freqs=STIM_FREQS)
        device = bm._get_device(str(tmp_dir), "99", "01", "ssvep")
        assert device == "Unknown Device"

    def test_output_dir_created(self, tmp_dir):
        from neurobids_flow.ssvep.benchmark import SSVEPBenchmark
        out = tmp_dir / "bench_out"
        SSVEPBenchmark(bids_root=str(tmp_dir), output_dir=str(out))
        assert out.exists()

    def test_subject_result_to_dict(self):
        from neurobids_flow.ssvep.benchmark import SubjectBenchmarkResult
        r = SubjectBenchmarkResult(
            subject="01", session="01", device="Brain Products",
            n_channels=8, n_epochs=20, cca_acc=0.55
        )
        d = r.to_dict()
        assert d["subject"] == "01"
        assert d["cca_acc"] == 0.55


# ══════════════════════════════════════════════════════════════════════════════
# 8. SSVEPFlow __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
class TestSSVEPInit:
    def test_all_exports_importable(self):
        from neurobids_flow.ssvep import (
            CCA, FBCCA, TRCA,
            SSVEPEvaluator, EvalResult,
            itr_bits_per_trial, itr_bits_per_minute,
            SSVEPConfig, load_ssvep_config,
            SSVEPPipeline, SSVEPVisualizer,
            SSVEPBenchmark, SSVEPPreprocessor,
        )
        assert all([
            CCA, FBCCA, TRCA,
            SSVEPEvaluator, EvalResult,
            itr_bits_per_trial, itr_bits_per_minute,
            SSVEPConfig, load_ssvep_config,
            SSVEPPipeline, SSVEPVisualizer,
            SSVEPBenchmark, SSVEPPreprocessor,
        ])