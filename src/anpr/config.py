"""Typed configuration loaded from YAML with command-line overrides."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path

import yaml


@dataclass
class VehicleCfg:
    """Vehicle detector settings."""

    weights: str = "yolo26n.pt"
    conf: float = 0.3
    imgsz: int = 640


@dataclass
class PlateCfg:
    """Plate detector settings."""

    enabled: bool = True
    kind: str = "rfdetr"
    weights: str = ""
    conf: float = 0.3


@dataclass
class OcrCfg:
    """OCR settings."""

    openocr_dir: str = ""
    weights: str = ""
    dict_path: str = ""
    two_line_aspect: float = 2.6


@dataclass
class TrackerCfg:
    """ByteTrack settings."""

    high_thr: float = 0.5
    low_thr: float = 0.1
    match_iou: float = 0.3
    max_age: int = 30
    min_hits: int = 2


@dataclass
class GateCfg:
    """Plate crop quality gate."""

    min_width: int = 40
    min_height: int = 12
    min_sharpness: float = 20.0


@dataclass
class ConsensusCfg:
    """When a track is considered settled."""

    settle_reads: int = 5
    settle_support: float = 0.8


@dataclass
class SinkCfg:
    """Output locations; empty string disables a sink."""

    jsonl: str = "out/events.jsonl"
    sqlite: str = ""


@dataclass
class Config:
    """Full pipeline configuration."""

    camera_id: str = "cam0"
    source: str = "0"
    live: bool = False
    realtime: bool = True
    queue_size: int = 2
    max_gap: int = 45
    report_every: float = 10.0
    vehicle: VehicleCfg = field(default_factory=VehicleCfg)
    plate: PlateCfg = field(default_factory=PlateCfg)
    ocr: OcrCfg = field(default_factory=OcrCfg)
    tracker: TrackerCfg = field(default_factory=TrackerCfg)
    gate: GateCfg = field(default_factory=GateCfg)
    consensus: ConsensusCfg = field(default_factory=ConsensusCfg)
    sinks: SinkCfg = field(default_factory=SinkCfg)


def _fill(obj, data: dict, path: str = "") -> None:
    """Copy ``data`` into dataclass ``obj``, rejecting unknown keys."""
    names = {f.name for f in fields(obj)}
    for key, val in data.items():
        if key not in names:
            raise KeyError(f"Unknown config key: {path}{key}")
        cur = getattr(obj, key)
        if is_dataclass(cur):
            _fill(cur, val or {}, f"{path}{key}.")
        else:
            setattr(
                obj, key, type(cur)(val) if cur is not None and val is not None else val
            )


def load_config(path: str | Path | None, overrides: dict | None = None) -> Config:
    """Load a YAML file (optional) and apply dotted overrides.

    Args:
        path: YAML file, or None for pure defaults.
        overrides: Mapping such as ``{"plate.enabled": False, "source": "a.mp4"}``.

    Returns:
        The populated configuration.

    Raises:
        KeyError: If a key in the file or the overrides is unknown. Failing
            loudly beats silently ignoring a typo in a deployment config.
    """
    cfg = Config()
    if path:
        data = yaml.safe_load(Path(path).read_text()) or {}
        _fill(cfg, data)
    for dotted, val in (overrides or {}).items():
        if val is None:
            continue
        parts = dotted.split(".")
        node = cfg
        for p in parts[:-1]:
            node = getattr(node, p)
        _fill(node, {parts[-1]: val}, ".".join(parts[:-1]) + "." if parts[:-1] else "")
    return cfg
