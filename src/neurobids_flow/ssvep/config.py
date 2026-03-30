"""
NeuroBIDS-Flow — SSVEP Config
==============================
YAML-based configuration for the SSVEPFlow pipeline.
Provides defaults and validation for all pipeline parameters.

Usage:
    from neurobids_flow.ssvep.config import SSVEPConfig, load_ssvep_config
    cfg = load_ssvep_config("configs/ssvep_config.yaml")
    print(cfg.stim_freqs)

    # Or use defaults directly:
    cfg = SSVEPConfig()
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class FilterConfig:
    """Bandpass / notch filter settings."""
    lowcut: float = 1.0
    highcut: float = 40.0
    notch: float = 50.0
    notch_width: float = 1.0
    order: int = 4


@dataclass
class EpochConfig:
    """Epoching settings."""
    tmin: float = 0.0
    tmax: float = 2.0
    baseline_start: float | None = None
    baseline_end: float = 0.0
    reject_peak_to_peak: float | None = None


@dataclass
class CCAConfig:
    """Baseline CCA classifier settings."""
    n_harmonics: int = 3
    n_components: int = 1


@dataclass
class FBCCAConfig:
    """Filter Bank CCA settings."""
    n_harmonics: int = 3
    n_components: int = 1
    filter_order: int = 4
    a: float = 1.25
    b: float = 0.25
    subbands: list[list[float]] = field(default_factory=lambda: [
        [6, 90], [14, 90], [22, 90], [30, 90], [38, 90]
    ])


@dataclass
class TRCAConfig:
    """TRCA / eTRCA classifier settings."""
    n_components: int = 1
    ensemble: bool = True


@dataclass
class EvalConfig:
    """Evaluation settings."""
    n_splits: int = 5
    epoch_duration: float = 2.0
    gap_duration: float = 0.5


@dataclass
class SSVEPConfig:
    """
    Master config for the SSVEPFlow pipeline.

    Parameters
    ----------
    stim_freqs : list[float]
        SSVEP stimulus frequencies in Hz.
    sfreq : float
        Sampling frequency (auto-read from BIDS if None).
    bids_root : str
        Path to BIDS output directory from Target 1.
    task : str
        BIDS task name.
    methods : list[str]
        Classifiers to run — subset of ["cca", "fbcca", "trca"].
    """
    stim_freqs: list[float] = field(default_factory=lambda: [6.0, 8.0, 10.0, 12.0])
    sfreq: float | None = None
    bids_root: str = "./bids_output"
    task: str = "ssvep"
    methods: list[str] = field(default_factory=lambda: ["cca", "fbcca", "trca"])

    filters: FilterConfig = field(default_factory=FilterConfig)
    epochs: EpochConfig = field(default_factory=EpochConfig)
    cca: CCAConfig = field(default_factory=CCAConfig)
    fbcca: FBCCAConfig = field(default_factory=FBCCAConfig)
    trca: TRCAConfig = field(default_factory=TRCAConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict:
        """Convert to plain dict safe for YAML serialization (no tuples)."""
        d = asdict(self)
        def _fix(obj):
            if isinstance(obj, tuple):
                return list(obj)
            if isinstance(obj, dict):
                return {k: _fix(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_fix(v) for v in obj]
            return obj
        return _fix(d)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_dict(cls, d: dict) -> "SSVEPConfig":
        cfg = cls()
        cfg.stim_freqs = d.get("stim_freqs", cfg.stim_freqs)
        cfg.sfreq = d.get("sfreq", cfg.sfreq)
        cfg.bids_root = d.get("bids_root", cfg.bids_root)
        cfg.task = d.get("task", cfg.task)
        cfg.methods = d.get("methods", cfg.methods)

        if "filters" in d:
            cfg.filters = FilterConfig(**{
                k: v for k, v in d["filters"].items()
                if k in FilterConfig.__dataclass_fields__
            })
        if "epochs" in d:
            ep = d["epochs"]
            ep.pop("baseline", None)  # remove old tuple field if present
            cfg.epochs = EpochConfig(**{
                k: v for k, v in ep.items()
                if k in EpochConfig.__dataclass_fields__
            })
        if "cca" in d:
            cfg.cca = CCAConfig(**{
                k: v for k, v in d["cca"].items()
                if k in CCAConfig.__dataclass_fields__
            })
        if "fbcca" in d:
            cfg.fbcca = FBCCAConfig(**{
                k: v for k, v in d["fbcca"].items()
                if k in FBCCAConfig.__dataclass_fields__
            })
        if "trca" in d:
            cfg.trca = TRCAConfig(**{
                k: v for k, v in d["trca"].items()
                if k in TRCAConfig.__dataclass_fields__
            })
        if "eval" in d:
            cfg.eval = EvalConfig(**{
                k: v for k, v in d["eval"].items()
                if k in EvalConfig.__dataclass_fields__
            })
        return cfg


def load_ssvep_config(path: str | Path) -> SSVEPConfig:
    """
    Load SSVEPConfig from a YAML file.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    SSVEPConfig
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f)
    return SSVEPConfig.from_dict(raw or {})


def save_default_config(path: str | Path = "configs/ssvep_config.yaml") -> None:
    """Save a default SSVEPConfig to YAML for the user to customise."""
    cfg = SSVEPConfig()
    cfg.save(path)
    print(f"Default SSVEP config saved to: {path}")


if __name__ == "__main__":
    save_default_config()