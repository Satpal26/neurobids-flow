"""
NeuroBIDS-Flow — SSVEPFlow (Target 2)
======================================
End-to-end SSVEP BCI pipeline built on top of NeuroBIDS-Flow BIDS output.

Modules:
    preprocessor  — filtering, epoching, baseline correction
    cca           — Canonical Correlation Analysis (baseline)
    fbcca         — Filter Bank CCA
    trca          — Task-Related Component Analysis (eTRCA)
    evaluator     — accuracy, ITR, confusion matrix, CV
    config        — YAML config loader / dataclasses
    pipeline      — end-to-end orchestrator
    visualizer    — result plots (accuracy, ITR, PSD, confusion matrix)
    benchmark     — cross-device / cross-subject benchmarking
"""

from .cca import CCA
from .fbcca import FBCCA
from .trca import TRCA
from .evaluator import SSVEPEvaluator, EvalResult, itr_bits_per_trial, itr_bits_per_minute
from .config import SSVEPConfig, load_ssvep_config
from .pipeline import SSVEPPipeline
from .visualizer import SSVEPVisualizer
from .benchmark import SSVEPBenchmark
from .preprocessor import SSVEPPreprocessor

__all__ = [
    "CCA",
    "FBCCA",
    "TRCA",
    "SSVEPEvaluator",
    "EvalResult",
    "itr_bits_per_trial",
    "itr_bits_per_minute",
    "SSVEPConfig",
    "load_ssvep_config",
    "SSVEPPipeline",
    "SSVEPVisualizer",
    "SSVEPBenchmark",
    "SSVEPPreprocessor",
]