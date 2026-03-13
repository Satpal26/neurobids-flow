# dataset_description.py
# Generates dataset_description.json — required for valid BIDS dataset
# BIDS spec requires this file at the root of every BIDS dataset

import json
from pathlib import Path
from .config import AppConfig


def generate_dataset_description(bids_root: str, config: AppConfig):
    """
    Creates dataset_description.json at BIDS root.
    This is required by BIDS spec — without it the dataset is invalid.
    """

    description = {
        "Name": config.dataset.name,
        "BIDSVersion": "1.9.0",
        "DatasetType": "raw",
        "Authors": config.dataset.authors if config.dataset.authors else ["Unknown"],
        "InstitutionName": config.dataset.institution,
        "EthicsApprovals": [config.dataset.ethics_approval] if config.dataset.ethics_approval else [],
        "GeneratedBy": [
            {
                "Name": "eeg2bids-unify",
                "Version": "0.1.0",
                "CodeURL": "https://github.com/Satpal26/eeg2bids-unify"
            }
        ],
        "HowToAcknowledge": "Please cite eeg2bids-unify when using this dataset."
    }

    output_path = Path(bids_root) / "dataset_description.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(description, f, indent=2)

    print(f"[dataset] Written dataset_description.json to {output_path}")
    return output_path
