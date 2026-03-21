# Changelog

All notable changes to NeuroBIDS-Flow are documented here.

---

## [1.2.0] — 2026-03-20

### MOABB Dataset Wrapper & ML Framework Bridge

- **`moabb_wrapper.py`** — new `NBIDSFDataset` class wrapping NeuroBIDS-Flow BIDS output as a fully MOABB-compatible dataset
- Auto-detects subjects (`sub-XX`) and sessions (`ses-XX`) directly from BIDS root folder structure
- Filters `bids_path.match()` results to EEG-only files (`.vhdr`, `.edf`, `.bdf`, `.set`, `.fif`) — skips `.tsv` / `.json` sidecars
- `NEUROBIDS_EVENTS` dictionary maps all passive BCI `trial_type` values (resting, workload, emotion) to integer labels for ML frameworks
- Compatible with **PyTorch** (Braindecode / TorchEEG), **TensorFlow** (Keras), and **Scikit-learn** — no data duplication or reformatting
- `read_raw_bids` automatically attaches `events.tsv` annotations and HED strings from `events.json` to MNE Raw objects
- **Bug fix** — `EEGConverter` updated to use `AppConfig` attribute access (`.recording.task`, `.dataset.name`, etc.) instead of `.get()` dict calls
- `moabb>=1.1.0` and `filelock>=3.12.0` added to package dependencies
- Tests expanded: 29 → **59** (30 new MOABB wrapper tests across 5 test classes)

---

## [1.1.0] — 2026-03-16

### HED Semantic Annotation Integration

- **EventHarmonizer** extended to support dual YAML config format — simple (`"1": "rest_open"`) and extended (`"1": {trial_type, hed}`)
- **`write_events_json()`** — new method writes BIDS-compliant `events.json` sidecar with full HED string dictionary keyed by `trial_type`
- **`has_hed()`** — new method returns True if any HED strings are configured
- **`dataset_description.py`** — `inject_hed=True` now automatically adds `"HEDVersion": "8.2.0"` to `dataset_description.json`
- **`GeneratedBy`** attribution block added to `dataset_description.json`
- **`default_config.yaml`** — all passive BCI events now include full HED strings (resting state, workload, emotion)
- **`hedtools>=0.5.0`** added to package dependencies
- Tests expanded: 19 → 29 (7 new HED tests + 3 dataset description tests)
- Output is now fully FAIR-compliant BIDS-EEG with semantic HED annotation

---
## [1.2.0] — 2026-03-21

### ML/DL Bridge Layer

- **`moabb_wrapper.py`** — `NBIDSFDataset` MOABB-compatible wrapper bridging BIDS+HED output to PyTorch/TF/sklearn pipelines
- **`torch_dataset.py`** — `NeuroBIDSFlowTorchDataset` PyTorch Dataset wrapper with DataLoader support, shape `(epochs, channels, times)`
- **`__init__.py`** — top-level exports added for `NBIDSFDataset`, `NeuroBIDSFlowTorchDataset`, `EEGConverter`, `EventHarmonizer`, `load_config`
- `moabb>=1.1.0`, `filelock>=3.12.0`, `torch>=2.0.0` added to dependencies
- Fixed `converter.py` — `AppConfig.get()` replaced with proper attribute access
- Tests expanded: 29 → 88 (30 MOABB tests + 29 PyTorch tests)
- Version bumped to 1.2.0


## [1.0.0] — 2026-03-13

### Initial Release

First stable release of NeuroBIDS-Flow — a modular graphical framework for standardizing multi-source EEG recordings to BIDS-EEG format.

#### Core Framework
- Plugin-based hardware abstraction layer with a unified `BaseHardwarePlugin` interface
- `EEGConverter` pipeline orchestrator — auto-detects hardware, dispatches to correct plugin, writes BIDS output
- YAML-based configuration layer — dataset-specific event mapping without source-code changes
- BIDS validator integration — MNE-BIDS validation with fallback to bids-validator (Node.js)

#### Hardware Plugins (5 supported)
- `BrainProductsPlugin` — BrainProducts ActiChamp Plus (`.vhdr` / `.vmrk` / `.eeg`)
- `NeuroscanPlugin` — Neuroscan NuAmps 40ch (`.cnt`)
- `OpenBCIPlugin` — OpenBCI Cyton 8-channel (`.txt`) with custom CSV parser
- `MusePlugin` — InteraXon Muse 2 (`.csv` Mind Monitor + `.xdf` MuseLSL) with auto-format detection
- `EmotivPlugin` — Emotiv EPOC+ 14-channel (`.edf`) with channel fingerprinting

#### EventHarmonizer
- Normalizes 5 raw event marker formats into unified BIDS-compliant `events.tsv`
- Supported input formats: TTL triggers, numerical IDs, LSL markers, software strings, EDF annotations
- Output columns: `onset | duration | trial_type | original_value | trigger_source`
- YAML-driven event mapping — no hardcoded trigger codes

#### GUI Frontend (Dear PyGui)
- Node-based pipeline editor — drag-and-drop canvas with visual pin connections
- EEG Signal Preview tab — renders first 8 channels, first 10 seconds, all 5 formats
- YAML Config panel — dataset name, task, event mapping, power line frequency
- Execution console — timestamped logs, progress bar, error reporting
- Windows-compatible — all labels use text, no emoji characters

#### CLI
- `neurobids-flow convert` — single-file conversion with subject/session/task flags
- Config auto-loaded from `configs/default_config.yaml`

#### Sample Data
- `sample_data/generate_samples.py` — generates valid synthetic EEG files for all 5 formats
- Realistic SSVEP signal — 10 Hz alpha + 6 Hz / 8 Hz stimulus bursts, 3 events per file

#### Testing
- 19 unit tests — all passing
- End-to-end BIDS conversion validated across 4 formats (BrainProducts, OpenBCI, Muse CSV, Emotiv)
- All outputs pass MNE-BIDS validation (BIDS v1.9.0)

---

## Roadmap

### [1.3.0] — Planned
- PyTorch Dataset class — direct `torch.utils.data.Dataset` wrapper for TorchEEG
- Pre-defined train/val/test splits JSON for reproducible ML benchmarking
- MOABB PR submission — contribute `NBIDSFDataset` to official MOABB repository

### [2.0.0] — Planned
- SSVEP paradigm-specific processing pipeline
- Motor imagery and passive BCI support
- Integration with NTU BCI Lab dataset pipeline