# NeuroBIDS-Flow

![CI](https://github.com/Satpal26/neurobids-flow/actions/workflows/ci.yml/badge.svg)

**Interoperable passive BCI workflows across consumer EEG sources through BIDS-EEG-based harmonization.**

Consumer EEG platforms — Muse 2, Emotiv EPOC+, OpenBCI Cyton — produce structurally incompatible output formats, making cross-device passive BCI research practically infeasible. NeuroBIDS-Flow solves this by converting heterogeneous consumer EEG recordings into a unified BIDS-EEG representation through a modular graphical framework with automated event harmonization.

---

## Quickstart

### 1 — Install

```bash
git clone https://github.com/Satpal26/neurobids-flow.git
cd neurobids-flow
uv pip install -e ".[dev]"
```

### 2 — Generate sample EEG files

```bash
python sample_data/generate_samples.py
```

Creates one valid sample file per supported format in `sample_data/generated/`.

### 3 — Run a conversion

```bash
# OpenBCI Cyton
neurobids-flow convert \
  --file sample_data/generated/sample_openbci.txt \
  --bids-root ./bids_output --subject 01 --session 01 --task workload

# Muse 2 (Mind Monitor CSV)
neurobids-flow convert \
  --file sample_data/generated/sample_muse.csv \
  --bids-root ./bids_output --subject 02 --session 01 --task workload

# Emotiv EPOC+
neurobids-flow convert \
  --file sample_data/generated/sample_emotiv.edf \
  --bids-root ./bids_output --subject 03 --session 01 --task workload
```

### 4 — Check output

```bash
ls bids_output/sub-01/ses-01/eeg/
```

### 5 — Run tests

```bash
uv run pytest tests/ -v
```

### 6 — Launch GUI (optional)

```bash
python neurobids_gui.py
```

---

## Supported Hardware

| Device | File Format | Type | Status |
|---|---|---|---|
| InteraXon Muse 2 (Mind Monitor) | .csv | Consumer | ✅ Done |
| InteraXon Muse 2 (MuseLSL) | .xdf | Consumer | ✅ Done |
| Emotiv EPOC+ | .edf | Consumer | ✅ Done |
| OpenBCI Cyton | .txt | Consumer | ✅ Done |
| BrainProducts ActiChamp Plus | .vhdr / .vmrk / .eeg | Research | ✅ Done |
| Neuroscan NuAmps | .cnt | Research | ✅ Done |

---

## System Architecture

```mermaid
graph TB
    subgraph GUI["GUI Layer (Optional)"]
        G1["Dear PyGui Node Editor"]
        G2["EEG Signal Preview"]
        G3["Execution Console"]
    end
    subgraph INPUT["Consumer EEG Sources"]
        A3[".txt — OpenBCI Cyton"]
        A4[".csv / .xdf — Muse 2"]
        A5[".edf — Emotiv EPOC+"]
    end
    subgraph PLUGINS["Hardware Plugin Layer"]
        B3["OpenBCIPlugin"]
        B4["MusePlugin"]
        B5["EmotivPlugin"]
    end
    subgraph CORE["Core Engine"]
        C1["EEGConverter\nauto-detect + orchestrate"]
        C2["EventHarmonizer\nnormalize all marker formats"]
        C3["YAML Config Loader\nno-code event mapping"]
        C4["BIDS Validator\nMNE-BIDS validation"]
    end
    subgraph OUTPUT["BIDS-EEG Output"]
        D1["sub-XX/eeg/*_eeg.*"]
        D2["sub-XX/eeg/*_events.tsv"]
        D3["sub-XX/eeg/*_eeg.json"]
        D4["sub-XX/eeg/*_channels.tsv"]
        D5["dataset_description.json"]
    end
    G1 --> C1
    A3 --> B3
    A4 --> B4
    A5 --> B5
    B3 & B4 & B5 --> C1
    C1 --> C2 --> D2
    C1 --> C3
    C1 --> C4
    C1 --> D1
    C1 --> D3
    C1 --> D4
    C1 --> D5
```

---

## Plugin Detection Flow

```mermaid
flowchart TD
    A["Consumer EEG File"] --> B["EEGConverter.detect_plugin()"]
    B --> C{Check extension + fingerprint}
    C -->|".txt"| F["OpenBCIPlugin"]
    C -->|".xdf or .csv"| G["MusePlugin"]
    C -->|".edf"| H["EmotivPlugin"]
    C -->|".vhdr"| D["BrainProductsPlugin"]
    C -->|".cnt"| E["NeuroscanPlugin"]
    C -->|"unknown"| I["ValueError: No plugin found"]
    D & E & F & G & H --> J["plugin.read_raw()"]
    J --> K["plugin.extract_events()"]
    K --> L["plugin.get_metadata()"]
    L --> M["EEGConverter writes BIDS output"]
```

---

## BIDS Conversion Pipeline

```mermaid
sequenceDiagram
    participant U as User / CLI / GUI
    participant C as EEGConverter
    participant P as Hardware Plugin
    participant EH as EventHarmonizer
    participant M as MNE-BIDS

    U->>C: convert(filepath, bids_root, subject, session, task)
    C->>P: detect(filepath)
    P-->>C: True / False
    C->>P: read_raw(filepath)
    P-->>C: mne.io.BaseRaw object
    C->>P: extract_events(filepath, raw)
    P-->>C: List[EventInfo]
    C->>EH: harmonize(events)
    EH-->>C: Unified BIDS events
    C->>P: get_metadata(filepath)
    P-->>C: HardwareMetadata
    C->>M: write_raw_bids(raw, bids_path, allow_preload=True)
    M-->>C: BIDS folder written
    C-->>U: BIDSPath + validation report
```

---

## EventHarmonizer — Supported Input Formats

Normalizes all consumer EEG marker types into a unified BIDS-compliant `events.tsv`:

| Input Format | Example | Source |
|---|---|---|
| LSL Markers | Lab Streaming Layer stream | Muse XDF |
| Software Strings | `eyes_open`, `workload_high` | Muse CSV, Emotiv |
| EDF Annotations | Annotation-based labels | Emotiv EDF |
| Numerical IDs | `1`, `2`, `99` | OpenBCI marker column |
| TTL Triggers | `S  1`, `S  2` | BrainProducts, Neuroscan |

Output columns: `onset | duration | trial_type | original_value | trigger_source`

---

## HED Semantic Annotation

NeuroBIDS-Flow automatically injects [Hierarchical Event Descriptors (HED)](https://www.hedtags.org) into the BIDS-EEG output when HED strings are defined in your config.

**What gets generated automatically:**

| File | Content |
|---|---|
| `*_events.tsv` | `onset \| duration \| trial_type \| value \| trigger_source` |
| `*_events.json` | HED string dictionary mapped to each `trial_type` |
| `dataset_description.json` | `"HEDVersion": "8.2.0"` injected automatically |

**Example `events.json` sidecar output:**
```json
{
    "trial_type": {
        "HED": {
            "rest_open":      "Sensory-event, (Eyes, Open), Rest",
            "rest_closed":    "Sensory-event, (Eyes, Closed), Rest",
            "cognitive_high": "Cognitive-effort, Task-difficulty/High"
        }
    }
}
```

This makes NeuroBIDS-Flow output fully FAIR-compliant — datasets are ready for cross-device mega-analyses and generalized passive BCI model training without any additional annotation steps.



## YAML Configuration

No source-code changes needed between datasets. Edit `configs/default_config.yaml`:

```yaml
dataset:
  name: "My Passive BCI Study"
  authors: ["Your Name"]
  institution: "Your Institution"

recording:
  task: "workload"
  power_line_freq: 50.0

event_mapping:
  "eyes_open":     "rest_open"
  "eyes_closed":   "rest_closed"
  "workload_low":  "cognitive_low"
  "workload_high": "cognitive_high"
  "1":             "rest_open"
  "2":             "rest_closed"
  "99":            "rest"

output:
  validate_bids: true
  overwrite: true
```

---

## Test Results

```
19 passed in 3.92s
```

All plugins validated. End-to-end BIDS conversion tested across consumer EEG formats — all passing MNE-BIDS validation.

---

## Project Structure

```
neurobids-flow/
    src/neurobids_flow/
        plugins/
            base.py              # abstract plugin interface
            brainproducts.py     # BrainProducts ActiChamp Plus
            neuroscan.py         # Neuroscan NuAmps
            openbci.py           # OpenBCI Cyton
            muse.py              # InteraXon Muse 2
            emotiv.py            # Emotiv EPOC+
        core/
            converter.py         # pipeline orchestrator
            harmonizer.py        # event normalization
            config.py            # YAML config loader
            validator.py         # BIDS validation
        cli.py                   # command line interface
    sample_data/
        generate_samples.py      # generates sample EEG files for all formats
        generated/               # gitignored — generated locally
    configs/
        default_config.yaml      # default configuration
    tests/
        test_plugins.py          # 19 tests
    neurobids_gui.py             # Dear PyGui node-based GUI
```

---

## Built With

- Python 3.11
- [MNE-Python 1.11](https://mne.tools)
- [MNE-BIDS 0.18](https://mne.tools/mne-bids)
- [Dear PyGui](https://github.com/hoffstadt/DearPyGui) — GUI frontend
- [uv](https://github.com/astral-sh/uv) — package manager

---

## Author

Satpal Singh — National Institute of Technology Raipur
