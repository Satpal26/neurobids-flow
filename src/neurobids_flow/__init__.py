# NeuroBIDS-Flow
# Interoperable passive BCI workflows across consumer EEG sources
# through BIDS-EEG-based harmonization

from .core.converter import EEGConverter
from .core.harmonizer import EventHarmonizer
from .core.config import load_config
from .moabb_wrapper import NBIDSFDataset
from .torch_dataset import NeuroBIDSFlowTorchDataset

__version__ = "1.2.0"
__all__ = [
    "EEGConverter",
    "EventHarmonizer",
    "load_config",
    "NBIDSFDataset",
    "NeuroBIDSFlowTorchDataset",
]
