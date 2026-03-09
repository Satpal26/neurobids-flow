# eeg2bids-unify

A plugin-based Python tool for converting heterogeneous EEG hardware files into BIDS-EEG format.

## Supported Hardware
- ✅ BrainProduct ActiChamp Plus (.vhdr)
- ✅ Neuroscan NuAmps (.cnt)
- 🔄 OpenBCI Cyton (.txt / .bdf) — in progress
- 🔄 Muse 2 (.csv / .xdf) — coming soon
- 🔄 Emotiv EPOC (.csv / .edf) — coming soon

## Built With
- Python 3.11
- MNE-Python
- MNE-BIDS
- uv (package manager)

## Project Structure
```
src/eeg2bids_unify/
    plugins/     # one plugin per hardware device
    core/        # converter engine
    cli.py       # command line interface
```

## Part of NTU Singapore BCI Lab Research
Target 1 of a 3-target research roadmap on EEG standardization and SSVEP analysis.