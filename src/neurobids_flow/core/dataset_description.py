# dataset_description.py
# Generates BIDS-compliant dataset_description.json
# Injects HEDVersion when HED strings are present in the config
# NeuroBIDS-Flow | NTU Singapore BCI Lab

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional


# HED schema version used by NeuroBIDS-Flow
HED_SCHEMA_VERSION = "8.2.0"


@dataclass
class DatasetDescription:
    """BIDS dataset_description.json fields."""
    name: str
    authors: list[str] = field(default_factory=list)
    institution: str = ""
    ethics_approval: str = ""
    hed_version: Optional[str] = None  # set to HED_SCHEMA_VERSION when HED is used


def write_dataset_description(
    bids_root: str,
    description: DatasetDescription,
    inject_hed: bool = False,
) -> str:
    """
    Write BIDS-compliant dataset_description.json to bids_root.

    Parameters
    ----------
    bids_root : str
        Root directory of the BIDS dataset.
    description : DatasetDescription
        Dataset metadata.
    inject_hed : bool
        If True, injects HEDVersion into the JSON.
        Set to True when EventHarmonizer has HED strings configured.

    Returns
    -------
    str
        Path to the written file.
    """
    os.makedirs(bids_root, exist_ok=True)
    filepath = os.path.join(bids_root, "dataset_description.json")

    # Load existing file if present (avoid overwriting fields set by mne-bids)
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            doc = json.load(f)
    else:
        doc = {}

    # Required BIDS fields
    doc["Name"] = description.name
    doc["BIDSVersion"] = "1.9.0"

    # Optional fields
    if description.authors:
        doc["Authors"] = description.authors
    if description.institution:
        doc["InstitutionName"] = description.institution
    if description.ethics_approval:
        doc["EthicsApprovals"] = [description.ethics_approval]

    # HED injection — only when harmonizer has HED strings
    if inject_hed:
        doc["HEDVersion"] = HED_SCHEMA_VERSION
    elif "HEDVersion" not in doc:
        # Don't add it if not needed, don't remove it if already there
        pass

    # NeuroBIDS-Flow attribution
    doc["GeneratedBy"] = [{
        "Name": "NeuroBIDS-Flow",
        "Version": "1.0.0",
        "Description": (
            "Modular graphical framework for standardizing multi-source "
            "EEG recordings to BIDS-EEG with HED semantic annotation."
        ),
        "CodeURL": "https://github.com/Satpal26/neurobids-flow"
    }]

    with open(filepath, "w") as f:
        json.dump(doc, f, indent=4)

    return filepath