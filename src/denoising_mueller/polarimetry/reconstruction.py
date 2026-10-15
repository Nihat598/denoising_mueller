"""Intensities -> Mueller matrix, following the lab's IMPv2 pipeline (libmpMuelMat / mmpt).

Ported from the lab's C code (C-libs/mp_comp_MM_AIW.c + compute_M_AIW.c, Moriconi 2022):
- every 16-vector (A, I, W per pixel) is read COLUMN-major ("Components MUST be Transposed!"):
  mat[r, c] = vec[r + 4 c]  ==  vec.reshape(4, 4).T
- M = A^-1 @ I @ W^-1, written back column-major; nM = M / m11.
So the stored nM is the column-major form of M, which is why the textbook matrix is
nM.reshape(4, 4).T (see lu_chipman.to_matrix).

Dark ("Noise") image: the IMPv2 calibration subtracts <wl>_Noise.npy from every calibration
image (libmpMuelMat_TL._load_calib_np_data), and the acquisition GUI subtracts it from the
measurement before computing M (save_data_ids.Calculation_MM_GUI). The offline batch pipeline
(processingmm / utils.get_intensity) does NOT, so stored MM.npz files are not dark-subtracted.
`subtract_dark=True` (default) is the physically consistent choice; False reproduces MM.npz.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np


# ---------- layout helpers ----------

def vec_to_mat(v: np.ndarray) -> np.ndarray:
    """(..., 16) column-major lab vectors -> (..., 4, 4) matrices."""
    return np.asarray(v, dtype=np.float64).reshape(*v.shape[:-1], 4, 4).swapaxes(-1, -2)


def mat_to_vec(m: np.ndarray) -> np.ndarray:
    """(..., 4, 4) matrices -> (..., 16) column-major lab vectors."""
    return np.ascontiguousarray(m.swapaxes(-1, -2)).reshape(*m.shape[:-2], 16)


# ---------- reconstruction ----------

def reconstruct_mm(I: np.ndarray, A: np.ndarray, W: np.ndarray, dark: np.ndarray | None = None,
                   subtract_dark: bool = True) -> dict:
    """I, A, W: (H, W, 16). dark: (H, W, 16) or (H, W) or None.

    Returns dict with 'M' (H, W, 16, unnormalized, lab layout), 'nM' (normalized, lab layout),
    'M11' (H, W) = M[..., 0], and 'I_used' (the intensities after dark subtraction).
    """
    I = np.asarray(I, dtype=np.float64)
    if subtract_dark and dark is not None:
        d = np.asarray(dark, dtype=np.float64)
        I = I - (d[..., None] if d.ndim == 2 else d)
    Am, Im, Wm = vec_to_mat(A), vec_to_mat(I), vec_to_mat(W)
    Mm = np.linalg.solve(Am, Im) @ np.linalg.inv(Wm)       # A^-1 I W^-1
    M = mat_to_vec(Mm)
    M11 = M[..., 0].copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        nM = M / M11[..., None]
    return {"M": M, "nM": nM, "M11": M11, "I_used": I}


def intensities_from_mm(M: np.ndarray, A: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Forward model I = A @ M @ W (lab layout in and out). Used for synthetic tests."""
    return mat_to_vec(vec_to_mat(A) @ vec_to_mat(M) @ vec_to_mat(W))


def saturation_mask(I_raw: np.ndarray, level: float = 255.0) -> np.ndarray:
    """True where any of the 16 raw intensities is saturated."""
    return (np.asarray(I_raw) >= level).any(axis=-1)


# ---------- calibration files ----------

CALIB_NAME_RE = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{6})")


def calibration_wl_dir(calib_dir: str | Path, wavelength: int = 630) -> Path:
    """Accept either the calibration root (<date>/) or its Processed_images/<wl>nm folder."""
    p = Path(calib_dir)
    for cand in (p, p / "Processed_images" / f"{wavelength}nm", p / f"{wavelength}nm"):
        if (cand / "A.npy").exists() and (cand / "W.npy").exists():
            return cand
    raise FileNotFoundError(f"No A.npy/W.npy for {wavelength} nm under {p}")


def load_calibration(calib_dir: str | Path, wavelength: int = 630) -> dict:
    """Return {'A', 'W', 'dark' (or None), 'path'} from an IMPv2 calibration folder."""
    d = calibration_wl_dir(calib_dir, wavelength)
    A = np.load(d / "A.npy").astype(np.float64)
    W = np.load(d / "W.npy").astype(np.float64)
    dark = None
    for name in (f"{wavelength}_Noise.npy", "Noise.npy"):
        for base in (d, d.parent, d.parent.parent):
            if (base / name).exists():
                dark = np.load(base / name).astype(np.float64)
                break
        if dark is not None:
            break
    if A.shape != W.shape or A.shape[-1] != 16:
        raise ValueError(f"A {A.shape} and W {W.shape} must both be (H, W, 16)")
    return {"A": A, "W": W, "dark": dark, "path": d}


def calibration_name_for(acq_path: str | Path) -> str | None:
    """Calibration folder name (YYYY-MM-DD_HHMMSS) recorded with an acquisition, if any.
    Looks in MetaData.json ('Calibration Folder') then MMProcessing.txt."""
    acq_path = Path(acq_path)
    md = acq_path / "MetaData.json"
    if md.exists():
        m = CALIB_NAME_RE.search(json.loads(md.read_text()).get("Calibration Folder", "") or "")
        if m:
            return m.group(1)
    mp = acq_path / "MMProcessing.txt"
    if mp.exists():
        m = CALIB_NAME_RE.search(mp.read_text())
        if m:
            return m.group(1)
    return None


def find_calibration(acq_path: str | Path, calib_root: str | Path) -> Path:
    """The calibration folder for an acquisition: the one named in its metadata if present
    under calib_root, else the most recent calibration taken before the acquisition."""
    calib_root = Path(calib_root)
    named = calibration_name_for(acq_path)
    if named:
        hits = [p for p in calib_root.rglob(named) if p.is_dir()]
        if hits:
            return hits[0]
    m = CALIB_NAME_RE.search(Path(acq_path).name)
    if not m:
        raise FileNotFoundError(f"Cannot date acquisition {acq_path}")
    t_acq = datetime.strptime(m.group(1), "%Y-%m-%d_%H%M%S")
    cands = []
    for p in calib_root.rglob("*"):
        mm = CALIB_NAME_RE.fullmatch(p.name) if p.is_dir() else None
        if mm:
            t = datetime.strptime(mm.group(1), "%Y-%m-%d_%H%M%S")
            if t <= t_acq:
                cands.append((t, p))
    if not cands:
        raise FileNotFoundError(f"No calibration before {t_acq} under {calib_root}"
                                + (f" (metadata names {named})" if named else ""))
    return max(cands)[1]
