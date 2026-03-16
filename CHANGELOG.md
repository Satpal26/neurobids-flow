
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

# Changelog

All notable changes to NeuroBIDS-Flow are documented here.

---

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

### [1.1.0] — Planned
- Additional hardware plugins (g.tec, Biosemi)
- Multi-file batch conversion via CLI
- GUI node state save/load

### [2.0.0] — Planned
- SSVEP paradigm-specific processing pipeline
- Motor imagery and passive BCI support
- Integration with NTU BCI Lab dataset pipeline