# eeg2bids-unify

A plugin-based Python tool for converting heterogeneous EEG hardware files into BIDS-EEG format.

Built for NTU Singapore BCI Lab — Target 1 of the SSVEP research roadmap.

---

## System Architecture
```mermaid
graph TB
    subgraph INPUT["Input Layer"]
        A1[".vhdr file\nBrainProducts"]
        A2[".cnt file\nNeuroscan"]
        A3[".txt / .bdf\nOpenBCI"]
        A4[".csv / .xdf\nMuse 2"]
        A5[".edf / .csv\nEmotiv EPOC"]
    end

    subgraph PLUGINS["Plugin Layer"]
        B1["BrainProductsPlugin"]
        B2["NeuroscanPlugin"]
        B3["OpenBCIPlugin"]
        B4["MusePlugin"]
        B5["EmotivPlugin"]
    end

    subgraph CORE["Core Engine"]
        C1["EEGConverter\n(auto-detect + dispatch)"]
        C2["EventHarmonizer\n(normalize all event formats)"]
        C3["MetadataBuilder\n(build BIDS sidecar JSON)"]
    end

    subgraph OUTPUT["BIDS Output"]
        D1["sub-01/eeg/*.edf"]
        D2["sub-01/eeg/*_events.tsv"]
        D3["sub-01/eeg/*_eeg.json"]
        D4["dataset_description.json"]
    end

    A1 --> B1
    A2 --> B2
    A3 --> B3
    A4 --> B4
    A5 --> B5

    B1 & B2 & B3 & B4 & B5 --> C1
    C1 --> C2
    C1 --> C3
    C2 --> D2
    C3 --> D3
    C1 --> D1
    C1 --> D4
```

---

## Plugin Detection Flow
```mermaid
flowchart TD
    A["EEG File Input"] --> B["EEGConverter.detect_plugin()"]
    B --> C{Check file extension}

    C -->|".vhdr"| D["BrainProductsPlugin ✅"]
    C -->|".cnt"| E["NeuroscanPlugin ✅"]
    C -->|".txt or .bdf"| F["OpenBCIPlugin 🔄"]
    C -->|".xdf or .csv"| G["MusePlugin 🔄"]
    C -->|".edf or .csv"| H["EmotivPlugin 🔄"]
    C -->|"unknown"| I["ValueError: No plugin found ❌"]

    D & E & F & G & H --> J["plugin.read_raw()"]
    J --> K["plugin.extract_events()"]
    K --> L["plugin.get_metadata()"]
    L --> M["Pass to Core Converter"]
```

---

## BIDS Conversion Pipeline
```mermaid
sequenceDiagram
    participant U as User / CLI
    participant C as EEGConverter
    participant P as Hardware Plugin
    participant M as MNE-BIDS

    U->>C: convert(filepath, bids_root, subject, session, task)
    C->>P: detect(filepath)
    P-->>C: True / False
    C->>P: read_raw(filepath)
    P-->>C: mne.io.BaseRaw object
    C->>P: extract_events(filepath, raw)
    P-->>C: List of EventInfo objects
    C->>P: get_metadata(filepath)
    P-->>C: HardwareMetadata object
    C->>M: BIDSPath(subject, session, task, root)
    C->>M: write_raw_bids(raw, bids_path)
    M-->>C: BIDS folder written
    C-->>U: BIDSPath (success)
```

---

## Supported Hardware

| Hardware | File Format | Status |
|---|---|---|
| BrainProducts ActiChamp Plus | .vhdr / .vmrk / .eeg | ✅ Done |
| Neuroscan NuAmps | .cnt | ✅ Done |
| OpenBCI Cyton | .txt / .bdf | ✅ Done |
| Muse 2 | .csv / .xdf | ✅ Done |
| Emotiv EPOC+ | .edf / .csv | ✅ Done |

## Test Results
```
19 passed in 3.92s
```

## Project Structure
```
src/eeg2bids_unify/
    plugins/
        base.py            # abstract plugin interface
        brainproducts.py   # BrainProducts ActiChamp
        neuroscan.py       # Neuroscan NuAmps
        openbci.py         # OpenBCI Cyton
        muse.py            # InteraXon Muse 2
        emotiv.py          # Emotiv EPOC+
    core/
        converter.py       # main conversion engine
        harmonizer.py      # event normalization
        config.py          # YAML config loader
        validator.py       # BIDS validation
    cli.py                 # command line interface
configs/
    default_config.yaml    # default configuration
tests/
    test_plugins.py        # 19 tests
```


## Built With
- Python 3.11
- MNE-Python 1.11
- MNE-BIDS 0.18
- uv (package manager)

