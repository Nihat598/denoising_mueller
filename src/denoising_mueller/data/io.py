"""Loading acquisitions from the HORAO / IMPv2 folder layout.

Layout of one acquisition:
    <acq>/MetaData.json                      (missing in some acquisitions)
    <acq>/To_Process/<wl>_Image_Number_<k>.npy   k = 1..N, uint8 (H, W, 16)
    <acq>/To_Process/<wl>_Image_Number_LQ.npy    = frame 1
    <acq>/To_Process/<wl>_Image_Number_HQ.npy    = 8-frame average from unknown frames: do not use
    <acq>/To_Process/<wl>_Image_Number_SHQ.npy   = floor(mean of 16 frames), when present
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

FRAME_RE = re.compile(r"^(\d+)_Image_Number_(\d+)\.npy$")


def load_paths(config: str | Path = "configs/paths.yaml") -> dict:
    with open(config) as f:
        return yaml.safe_load(f)


@dataclass
class Acquisition:
    path: Path
    wavelength: int = 630
    frame_files: list[Path] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def setting(self) -> str:
        """'OR' (operating room) or 'FF', parsed from the folder name."""
        m = re.search(r"_F_(OR|FF)_", self.name)
        return m.group(1) if m else "?"

    @property
    def n_frames(self) -> int:
        return len(self.frame_files)

    def load_frames(self, dtype=np.float32) -> np.ndarray:
        """Return frames as (N, 16, H, W)."""
        frames = [np.load(f).astype(dtype) for f in self.frame_files]
        return np.stack(frames).transpose(0, 3, 1, 2)

    def load_stored(self, which: str = "HQ", dtype=np.float32) -> np.ndarray:
        """Stored LQ or HQ image as (16, H, W)."""
        f = self.path / "To_Process" / f"{self.wavelength}_Image_Number_{which}.npy"
        return np.load(f).astype(dtype).transpose(2, 0, 1)


def open_acquisition(path: str | Path, wavelength: int = 630) -> Acquisition:
    path = Path(path)
    tp = path / "To_Process"
    frames = []
    for f in tp.glob("*.npy"):
        m = FRAME_RE.match(f.name)
        if m and int(m.group(1)) == wavelength:
            frames.append((int(m.group(2)), f))
    frames = [f for _, f in sorted(frames)]
    md = {}
    if (path / "MetaData.json").exists():
        md = json.loads((path / "MetaData.json").read_text())
    return Acquisition(path=path, wavelength=wavelength, frame_files=frames, metadata=md)


def list_acquisitions(data_root: str | Path, wavelength: int = 630) -> list[Acquisition]:
    """All acquisitions under data_root (any depth), sorted by folder name."""
    root = Path(data_root)
    dirs = sorted({tp.parent for tp in root.rglob("To_Process") if tp.is_dir()}, key=lambda p: p.name)
    return [open_acquisition(p, wavelength) for p in dirs]


def find_acquisition(data_root: str | Path, name: str, wavelength: int = 630) -> Acquisition:
    """Find an acquisition by folder name anywhere under data_root."""
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"data_root does not exist: {root}  (check configs/paths.yaml)")
    direct = root / name
    if (direct / "To_Process").is_dir():
        return open_acquisition(direct, wavelength)
    matches = [tp.parent for tp in root.rglob("To_Process") if tp.parent.name == name]
    if not matches:
        raise FileNotFoundError(f"No acquisition named '{name}' under {root}")
    if len(matches) > 1:
        raise ValueError(f"Several acquisitions named '{name}': {matches}")
    return open_acquisition(matches[0], wavelength)
