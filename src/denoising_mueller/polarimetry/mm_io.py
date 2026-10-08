"""Loading the lab's precomputed MM.npz files.

Path: <acq>/polarimetry/<wl>_Image_Number_<LQ|HQ|SHQ>/<wl>nm/MM.npz
Keys: nM (H, W, 16; float16 in 'light' mode, nM[..., 0] == 1), M11 (H, W), linR, azimuth,
totP, totD, Msk (unreliable for IMPv2 uint8 data: saturation test uses the wrong camera type).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def mm_npz_path(acq_path: str | Path, which: str = "LQ", wavelength: int = 630) -> Path:
    return Path(acq_path) / "polarimetry" / f"{wavelength}_Image_Number_{which}" / f"{wavelength}nm" / "MM.npz"


def load_mm_npz(acq_path: str | Path, which: str = "LQ", wavelength: int = 630) -> dict:
    """Return {'nM': (H, W, 16) float64, 'M11': (H, W) float64, 'maps': {name: (H, W)}}."""
    path = mm_npz_path(acq_path, which, wavelength)
    if not path.exists():
        raise FileNotFoundError(path)
    z = np.load(path)
    maps = {k: z[k].astype(np.float64) for k in ("totD", "linR", "azimuth", "totP") if k in z.files}
    return {"nM": z["nM"].astype(np.float64), "M11": z["M11"].astype(np.float64), "maps": maps, "path": path}


def available_references(acq_path: str | Path, wavelength: int = 630) -> list[str]:
    return [w for w in ("SHQ", "HQ") if mm_npz_path(acq_path, w, wavelength).exists()]
