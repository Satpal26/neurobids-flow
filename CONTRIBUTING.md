# Contributing to NeuroBIDS-Flow

Thank you for your interest in contributing to NeuroBIDS-Flow! This guide explains how to add new hardware plugins, extend event harmonization, and submit contributions.

---

## Adding a New Hardware Plugin

Adding support for a new EEG device requires creating a single Python file implementing the `BaseHardwarePlugin` interface.

### Step 1 — Create the plugin file

Create a new file in `src/neurobids_flow/plugins/`:

```
src/neurobids_flow/plugins/yourdevice.py
```

### Step 2 — Implement the interface

Every plugin must implement exactly 4 methods:

```python
from neurobids_flow.plugins.base import BaseHardwarePlugin, EventInfo, HardwareMetadata
import mne

class YourDevicePlugin(BaseHardwarePlugin):

    def detect(self, filepath: str) -> bool:
        """Return True if this plugin should handle the given file.
        
        Use file extension and/or content fingerprinting.
        Never raise exceptions — always return True or False.
        """
        return filepath.endswith(".yourextension")

    def read_raw(self, filepath: str) -> mne.io.BaseRaw:
        """Parse the file and return an MNE Raw object.
        
        - Set channel types correctly (eeg, stim, misc)
        - Do NOT preload data here
        - Units must be in Volts
        """
        raw = mne.io.read_raw_yourformat(filepath, preload=False)
        return raw

    def extract_events(self, filepath: str, raw: mne.io.BaseRaw) -> list[EventInfo]:
        """Extract stimulus markers and return as EventInfo list.
        
        trigger_source options:
            hardware_ttl  — TTL trigger from hardware
            lsl           — Lab Streaming Layer marker
            software      — software-generated string
            keystroke     — keyboard response
        """
        events = []
        # your extraction logic here
        events.append(EventInfo(
            onset=1.0,
            duration=0.0,
            description="your_marker",
            trigger_source="software"
        ))
        return events

    def get_metadata(self, filepath: str) -> HardwareMetadata:
        """Return device metadata for BIDS sidecar JSON."""
        return HardwareMetadata(
            manufacturer="Your Manufacturer",
            model="Your Model",
            sampling_rate=256.0,
            channel_count=8,
            reference_scheme="CMS/DRL",
            power_line_freq=50.0,
            eeg_ground="DRL",
        )
```

### Step 3 — Register the plugin

Open `src/neurobids_flow/core/converter.py` and add your plugin to the `PLUGINS` list and `FORMAT_MAP`:

```python
from neurobids_flow.plugins.yourdevice import YourDevicePlugin

PLUGINS = [
    BrainProductsPlugin,
    NeuroscanPlugin,
    EmotivPlugin,
    MusePlugin,
    OpenBCIPlugin,
    YourDevicePlugin,   # ← add here
]

FORMAT_MAP = {
    ...
    "YourDevicePlugin": "BrainVision",  # or "EDF" depending on output format
}
```

### Step 4 — Add detection tests

Add a test class to `tests/test_plugins.py`:

```python
class TestYourDevicePlugin:
    def setup_method(self):
        self.plugin = YourDevicePlugin()

    def test_detects_correct_extension(self, tmp_path):
        f = tmp_path / "test.yourextension"
        f.write_bytes(b"\x00" * 10)
        assert self.plugin.detect(str(f)) is True

    def test_rejects_wrong_extension(self, tmp_path):
        f = tmp_path / "test.vhdr"
        f.write_text("[Common Infos]\n")
        assert self.plugin.detect(str(f)) is False
```

### Step 5 — Run tests

```bash
python -m pytest tests/ -v
```

All existing tests must still pass.

---

## Adding HED Strings for a New Device

If your device uses custom event markers, add them to `configs/default_config.yaml`:

```yaml
event_mapping:
  "your_marker":
    trial_type: "descriptive_name"
    hed: "Sensory-event, Visual-presentation"
```

HED strings must follow [HED schema 8.2.0](https://www.hedtags.org). Use the [HED tag search](https://hedtools.org/hed) to find correct tags.

---

## Development Setup

```bash
git clone https://github.com/Satpal26/neurobids-flow.git
cd neurobids-flow
uv pip install -e ".[dev]"
python -m pytest tests/ -v
```

---

## Pull Request Checklist

Before submitting a PR, make sure:

- [ ] All existing 29 tests pass
- [ ] New plugin has at least 2 detection tests
- [ ] `detect()` never raises exceptions
- [ ] Units in `read_raw()` are in Volts
- [ ] Plugin registered in `converter.py`
- [ ] HED strings added to `default_config.yaml` if applicable
- [ ] `CHANGELOG.md` updated

---

## Project Structure

```
src/neurobids_flow/
    plugins/
        base.py          ← BaseHardwarePlugin interface — read this first
        brainproducts.py ← reference implementation
        openbci.py       ← example of custom CSV parser
        muse.py          ← example of dual format detection
        emotiv.py        ← example of content fingerprinting
    core/
        converter.py     ← pipeline orchestrator — register plugins here
        harmonizer.py    ← EventHarmonizer + HED injection
        config.py        ← YAML config loader
        dataset_description.py ← BIDS dataset_description.json writer
```

---

## Questions

Open an issue on GitHub or contact the maintainer via the repository.