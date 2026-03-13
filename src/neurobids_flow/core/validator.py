# validator.py
# BIDS Validator integration
# Checks that the output folder is a valid BIDS dataset
# Uses bids-validator via subprocess (Node.js based)
# Falls back to MNE-BIDS validation if bids-validator not installed

import subprocess
import shutil
from pathlib import Path
import mne_bids


def validate_bids(bids_root: str) -> bool:
    """
    Validate a BIDS dataset.
    First tries official bids-validator (Node.js).
    Falls back to MNE-BIDS built-in validator if not available.
    Returns True if valid, False if not.
    """
    bids_root = Path(bids_root)

    if not bids_root.exists():
        print(f"[validator] ERROR: BIDS root does not exist: {bids_root}")
        return False

    print(f"[validator] Validating BIDS dataset at: {bids_root}")

    # Try official bids-validator first (Node.js)
    if shutil.which("bids-validator"):
        return _validate_with_bids_validator(bids_root)
    else:
        print("[validator] bids-validator not found — using MNE-BIDS validator")
        return _validate_with_mne_bids(bids_root)


def _validate_with_bids_validator(bids_root: Path) -> bool:
    """Run official Node.js bids-validator."""
    try:
        result = subprocess.run(
            ["bids-validator", str(bids_root), "--json"],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            print("[validator] BIDS validation PASSED ✅")
            return True
        else:
            print("[validator] BIDS validation FAILED ❌")
            print(result.stdout[:500])
            return False
    except subprocess.TimeoutExpired:
        print("[validator] Validation timed out")
        return False
    except Exception as e:
        print(f"[validator] Validation error: {e}")
        return False


def _validate_with_mne_bids(bids_root: Path) -> bool:
    """Run MNE-BIDS built-in validation."""
    try:
        report = mne_bids.make_report(bids_root)
        print("[validator] MNE-BIDS validation PASSED ✅")
        print(f"[validator] Report:\n{report}")
        return True
    except Exception as e:
        print(f"[validator] MNE-BIDS validation FAILED ❌: {e}")
        return False
