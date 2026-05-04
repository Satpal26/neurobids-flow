# NeuroBIDS-Flow ![CI](https://github.com/Satpal26/neurobids-flow/actions/workflows/ci.yml/badge.svg)

**Interoperable passive BCI workflows across consumer EEG sources through BIDS-EEG-based harmonization.**

Consumer EEG platforms — Muse 2, Emotiv EPOC+, OpenBCI Cyton — produce structurally incompatible files that block cross-device research. NeuroBIDS-Flow solves this by converting raw EEG from 5 hardware formats into BIDS-EEG + HED-annotated datasets, then running a full SSVEP BCI pipeline on top — all from a single CLI command.

---

## 🗺️ Project Roadmap

| Target | Module | Status |
|--------|--------|--------|
| **Target 1** | NeuroBIDS-Flow — BIDS-EEG Conversion + ML Bridge | ✅ Complete |
| **Target 2** | SSVEPFlow — End-to-end SSVEP BCI Framework | ✅ Complete |
| **Target 3** | Novel SSVEP Method on lab dataset | 🔄 Planned |

---

## ✨ Features

### Target 1 — NeuroBIDS-Flow (BIDS Conversion)
- **5 hardware plugins** — BrainProducts, Neuroscan, OpenBCI, Muse 2, Emotiv EPOC+
- **BIDS-EEG + HED** semantic annotation (v1.1.0)
- **EventHarmonizer** ★ novel cross-device event normalization
- **MOABB wrapper** (`NBIDSFDataset`) — bridges to PyTorch / TF / sklearn
- **PyTorch Dataset** (`NeuroBIDSFlowTorchDataset`) — DataLoader-ready epochs
- **ML pipeline** — CSP+LDA (0.526) + EEGNet (0.545) cross-device evaluation
- **Full pipeline demo** — Raw EEG → BIDS+HED → ML results in 17.4s
- **YAML config** + CLI + BIDS Validator

### Target 2 — SSVEPFlow (SSVEP BCI Framework)
- **CCA** — training-free canonical correlation analysis baseline
- **FBCCA** — filter bank CCA with 5 sub-bands and weighted scores
- **TRCA / eTRCA** — task-related component analysis with ensemble mode
- **Evaluator** — accuracy, ITR (bits/trial + bits/min), confusion matrix, CV
- **Visualizer** — accuracy/ITR bar charts, PSD plots, confusion matrix heatmaps
- **Benchmark** — cross-device/cross-subject evaluation with JSON export
- **YAML config** — fully configurable pipeline via `ssvep_config.yaml`
- **End-to-end pipeline** — BIDS → preprocess → classify → evaluate → plot

---

## 📊 Test Coverage

| Module | Tests |
|--------|-------|
| Hardware plugins (5 devices) | 29 |
| MOABB wrapper | 30 |
| PyTorch Dataset | 29 |
| ML pipeline (CSP+LDA, EEGNet, cross-device) | 32 |
| SSVEPFlow (CCA, FBCCA, TRCA, evaluator, config, visualizer, benchmark) | 66 |
| **Total** | **186 ✅** |

---

## 🚀 Quick Start

```bash
pip install neurobids-flow
```

### Target 1 — Convert raw EEG to BIDS

```bash
# Single file
neurobids-flow convert \
  --file sample_data/generated/sample_brainproducts.vhdr \
  --bids-root ./bids_output \
  --subject 01 --session 01 --task workload

# Full pipeline demo (all 5 devices → BIDS → ML in ~17s)
python src/neurobids_flow/pipeline_demo.py
```

### Target 2 — Run SSVEP pipeline

```bash
# Run CCA + FBCCA + TRCA on BIDS output
python src/neurobids_flow/ssvep/pipeline.py \
  --bids-root ./bids_output \
  --freqs 6.0 8.0 10.0 12.0

# Cross-device benchmark
python src/neurobids_flow/ssvep/benchmark.py \
  --bids-root ./bids_output
```

### Python API

```python
# Target 1 — BIDS conversion
from neurobids_flow import EEGConverter, load_config
cfg = load_config("configs/default_config.yaml")
converter = EEGConverter(cfg)
converter.convert("sample.vhdr", bids_root="./bids_output", subject="01")

# Target 1 — ML pipeline
from neurobids_flow import NBIDSFDataset
dataset = NBIDSFDataset(dataset_name="MyStudy", bids_root="./bids_output")
X, y, metadata = dataset.get_data()

# Target 2 — SSVEP classification
from neurobids_flow.ssvep import CCA, FBCCA, TRCA, SSVEPEvaluator

clf = FBCCA(stim_freqs=[6.0, 8.0, 10.0, 12.0], sfreq=256.0)
preds = clf.predict(X)  # X: (n_epochs, n_channels, n_times)

ev = SSVEPEvaluator(n_classes=4, epoch_duration=2.0)
result = ev.evaluate(clf, X, y)
print(f"Accuracy: {result.accuracy:.3f}  ITR: {result.itr_bpm:.1f} bits/min")

# Target 2 — Full pipeline
from neurobids_flow.ssvep import SSVEPPipeline, SSVEPConfig
cfg = SSVEPConfig()
cfg.bids_root = "./bids_output"
cfg.stim_freqs = [6.0, 8.0, 10.0, 12.0]
pipe = SSVEPPipeline(cfg)
results = pipe.run()
```

---

## 🏗️ Project Structure

```
neurobids-flow/
├── src/neurobids_flow/
│   ├── plugins/                    # Target 1 — Hardware plugins
│   │   ├── brainproducts.py        # BrainProducts ActiChamp (.vhdr)
│   │   ├── neuroscan.py            # Neuroscan (.cnt)
│   │   ├── openbci.py              # OpenBCI Cyton (.txt)
│   │   ├── muse.py                 # Muse 2 (.csv / .xdf)
│   │   └── emotiv.py               # Emotiv EPOC+ (.edf)
│   ├── core/
│   │   ├── converter.py            # Main BIDS converter
│   │   ├── harmonizer.py           # EventHarmonizer ★
│   │   └── config.py               # YAML config loader
│   ├── moabb_wrapper.py            # NBIDSFDataset (MOABB bridge)
│   ├── torch_dataset.py            # NeuroBIDSFlowTorchDataset
│   ├── sklearn_pipeline.py         # CSP + LDA baseline
│   ├── braindecode_pipeline.py     # EEGNet (Braindecode)
│   ├── cross_device_eval.py        # Cross-device evaluation table
│   ├── splits.py                   # Reproducible train/val/test splits
│   ├── pipeline_demo.py            # Full end-to-end demo
│   └── ssvep/                      # Target 2 — SSVEPFlow
│       ├── preprocessor.py         # BIDS loader + filter + epoch
│       ├── cca.py                  # CCA classifier
│       ├── fbcca.py                # Filter Bank CCA
│       ├── trca.py                 # TRCA / eTRCA
│       ├── evaluator.py            # ITR, accuracy, confusion matrix
│       ├── config.py               # YAML config dataclasses
│       ├── pipeline.py             # End-to-end orchestrator
│       ├── visualizer.py           # Result plots
│       └── benchmark.py            # Cross-device benchmark
├── tests/
│   ├── test_plugins.py             # 29 plugin tests
│   ├── test_moabb_wrapper.py       # 30 MOABB tests
│   ├── test_torch_dataset.py       # 29 PyTorch tests
│   ├── test_ml_pipeline.py         # 32 ML pipeline tests
│   └── test_ssvep.py               # 66 SSVEPFlow tests
├── sample_data/
│   ├── generate_samples.py         # Synthetic EEG generator (60s, 20 events)
│   └── generated/                  # 5 sample files (one per device)
├── configs/
│   ├── default_config.yaml         # Target 1 BIDS config
│   ├── splits.json                 # Reproducible subject splits
│   └── ssvep_config.yaml           # Target 2 SSVEP config
└── docs/
    └── architecture_v3.png         # System architecture diagram
```

---

## 🔌 Supported Hardware

| Device | Format | Channels | Plugin |
|--------|--------|----------|--------|
| BrainProducts ActiChamp | `.vhdr` + `.vmrk` + `.eeg` | Up to 256 | `brainproducts.py` |
| Neuroscan SynAmps | `.cnt` | Up to 64 | `neuroscan.py` |
| OpenBCI Cyton | `.txt` | 8–16 | `openbci.py` |
| Muse 2 | `.csv` / `.xdf` | 4 | `muse.py` |
| Emotiv EPOC+ | `.edf` | 14 | `emotiv.py` |

---

## 📈 Results (Synthetic Data)

### Target 1 — Cross-Device ML Evaluation

| Method | Accuracy | ITR (bits/min) |
|--------|----------|----------------|
| CSP + LDA | 0.526 ± 0.114 | — |
| EEGNet (Braindecode) | 0.545 | — |
| Full pipeline runtime | — | 17.4s |

> Results on synthetic data — expect near-chance. Real passive BCI recordings will show above-chance performance.

### Target 2 — SSVEP Classification (4-class)

| Method | Description | Requires Training |
|--------|-------------|-------------------|
| CCA | Canonical Correlation Analysis | ❌ No |
| FBCCA | Filter Bank CCA (5 sub-bands) | ❌ No |
| TRCA / eTRCA | Task-Related Component Analysis | ✅ Yes |

---

## 🛠️ Built With

- [MNE-Python](https://mne.tools/) — EEG processing
- [MNE-BIDS](https://mne.tools/mne-bids/) — BIDS-EEG conversion
- [HED Tools](https://www.hedtags.org/) — Semantic annotation
- [MOABB](https://neurotechx.github.io/moabb/) — BCI benchmark framework
- [PyTorch](https://pytorch.org/) — Deep learning
- [Braindecode](https://braindecode.org/) — EEGNet
- [scikit-learn](https://scikit-learn.org/) — CSP + LDA
- [SciPy](https://scipy.org/) — Signal processing (FBCCA filters)
- [Matplotlib](https://matplotlib.org/) — Visualizations

---

## 📄 Paper

> **"Interoperable Passive BCI Workflows across Consumer EEG Sources through BIDS-EEG-Based Harmonization"**

---

## 👤 Author

**Satpal** — Remote Research Intern, BCI Lab, NTU Singapore
Supervisor: Prof. Aung Aung Phyo Wai

---

## 📜 License

MIT License