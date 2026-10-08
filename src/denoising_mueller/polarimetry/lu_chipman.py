"""Lu-Chipman polar decomposition and polarimetric maps, vectorized NumPy.

Portable replacement for the lab's compiled libmpMuelMat routines (Windows DLL),
following the definitions in mmpt/addons/polarpred/mm/functions/{lu_chipman,polarimetry}.py.
Validated against the maps stored in the lab's MM.npz files (see tests/ and docs/data_notes.md).

Conventions
-----------
- `nM` is the lab's (H, W, 16) array, normalized so nM[..., 0] == 1.
- The lab stores the matrix transposed: the textbook Mueller matrix is
  `nM.reshape(H, W, 4, 4).swapaxes(-1, -2)` (row index = output Stokes component).
- Angles are returned in degrees: linR in [0, 180], azimuth in [0, 180).
"""
from __future__ import annotations

import numpy as np

MAP_NAMES = ("totD", "linR", "azimuth", "totP")


def to_matrix(nM: np.ndarray, transposed_storage: bool = True) -> np.ndarray:
    """(..., 16) lab layout -> (..., 4, 4) textbook Mueller matrices (float64)."""
    M = np.asarray(nM, dtype=np.float64).reshape(*nM.shape[:-1], 4, 4)
    return M.swapaxes(-1, -2) if transposed_storage else M


def from_matrix(M: np.ndarray, transposed_storage: bool = True) -> np.ndarray:
    """(..., 4, 4) textbook matrices -> (..., 16) lab layout."""
    if transposed_storage:
        M = M.swapaxes(-1, -2)
    return M.reshape(*M.shape[:-2], 16)


def normalize(M: np.ndarray) -> np.ndarray:
    m00 = M[..., 0, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        return M / m00[..., None, None]


def decompose(M: np.ndarray):
    """Lu-Chipman decomposition M = M_delta @ M_R @ M_D of normalized matrices (..., 4, 4).

    Returns (MD, MR, Mdelta), each (..., 4, 4).
    """
    shape = M.shape[:-2]
    M = M.reshape(-1, 4, 4)
    n = len(M)
    eye4 = np.broadcast_to(np.eye(4), (n, 4, 4))

    # --- diattenuator ---
    d = M[:, 0, 1:4]                                   # diattenuation vector (first row)
    D = np.linalg.norm(d, axis=-1)
    D_safe = np.clip(D, 0.0, 1.0 - 1e-12)
    D1 = np.sqrt(1.0 - D_safe ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        dhat = np.where(D[:, None] > 0, d / D[:, None], 0.0)
    MD = np.zeros((n, 4, 4))
    MD[:, 0, 0] = 1.0
    MD[:, 0, 1:] = d
    MD[:, 1:, 0] = d
    MD[:, 1:, 1:] = D1[:, None, None] * np.eye(3) + (1 - D1)[:, None, None] * dhat[:, :, None] * dhat[:, None, :]

    finite = np.isfinite(M).all(axis=(1, 2)) & (D < 1.0)
    M0 = np.full((n, 4, 4), np.nan)
    M0[finite] = M[finite] @ np.linalg.inv(MD[finite])
    zeroD = finite & (D == 0)
    M0[zeroD] = M[zeroD]
    MD[zeroD] = eye4[zeroD]

    # --- retarder: orthogonal polar factor of the 3x3 block (as in the lab's torch code) ---
    MR = np.tile(np.eye(4), (n, 1, 1))
    ok = finite & np.isfinite(M0).all(axis=(1, 2))
    m0 = M0[ok, 1:, 1:]
    U, _, Vt = np.linalg.svd(m0)
    S = np.tile(np.eye(3), (len(m0), 1, 1))
    S[np.linalg.det(M[ok]) < 0, 2, 2] = -1.0
    MR[ok, 1:, 1:] = U @ S @ Vt
    MR[~ok] = np.nan

    # --- depolarizer ---
    Mdelta = M0 @ MR.swapaxes(-1, -2)

    return (MD.reshape(*shape, 4, 4), MR.reshape(*shape, 4, 4), Mdelta.reshape(*shape, 4, 4))


def _acos_deg(x):
    return np.degrees(np.arccos(np.clip(x, -1.0, 1.0)))


def _rot(theta_deg):
    """Circular retarder (rotation) matrices for optical rotation theta (degrees)."""
    t = np.radians(theta_deg)
    c, s = np.cos(2 * t), np.sin(2 * t)
    R = np.zeros(t.shape + (4, 4))
    R[..., 0, 0] = 1
    R[..., 1, 1] = c
    R[..., 1, 2] = -s
    R[..., 2, 1] = s
    R[..., 2, 2] = c
    R[..., 3, 3] = 1
    return R


def retardance_params(MR: np.ndarray, tol: float = 1e-9):
    """Linear retardance and azimuth (degrees) with the LIN-CIR split used in HORAO papers."""
    arg = 0.5 * (MR[..., 1, 1] + MR[..., 2, 2] + MR[..., 3, 3]) - 0.5
    totR = _acos_deg(arg)

    cir = _acos_deg(MR[..., 3, 3])
    MRC = _rot(cir / 2)

    def _cir_temp(X):
        return np.degrees(np.arctan2(X[..., 2, 1] - X[..., 1, 2], X[..., 2, 2] + X[..., 1, 1]))

    MRL = MR @ MRC.swapaxes(-1, -2)
    alt = MR @ MRC
    swap = np.abs(_cir_temp(MRL)) > np.abs(cir)
    MRL[swap] = alt[swap]
    small = totR < tol
    MRL[small] = MR[small]

    Mn = (MRL[..., 1, 1] + MRL[..., 2, 2]) ** 2 + (MRL[..., 2, 1] - MRL[..., 1, 2]) ** 2
    linR = _acos_deg(np.sqrt(Mn) - 1)

    # Azimuth as computed by the lab's compiled libmpMuelMat (matched to 0.0014 deg median on
    # stored maps): half-angle of the retardance vector (a1, a2), rotated by 90 deg and
    # corrected by half the optical rotation psi of MR.
    R = np.radians(totR)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = 1.0 / (2.0 * np.sin(R))
    a1 = k * (MR[..., 2, 3] - MR[..., 3, 2])
    a2 = k * (MR[..., 3, 1] - MR[..., 1, 3])
    psi = 0.5 * np.degrees(np.arctan2(MR[..., 2, 1] - MR[..., 1, 2], MR[..., 1, 1] + MR[..., 2, 2]))
    azimuth = np.mod(0.5 * np.degrees(np.arctan2(a2, a1)) + 90.0 + psi / 2.0, 180.0)
    return linR, azimuth


def polarimetric_maps(nM: np.ndarray, transposed_storage: bool = True) -> dict:
    """nM (H, W, 16) -> dict of maps: totD, linR (deg), azimuth (deg), totP, plus validity masks."""
    M = normalize(to_matrix(nM, transposed_storage))
    MD, MR, Mdelta = decompose(M)

    totD = np.linalg.norm(MD[..., 0, 1:4], axis=-1)
    linR, azimuth = retardance_params(MR)
    diag = np.abs(np.stack([Mdelta[..., 1, 1], Mdelta[..., 2, 2], Mdelta[..., 3, 3]], -1))
    with np.errstate(invalid="ignore"):
        totP = 1.0 - np.mean(diag, axis=-1)

    return {"totD": totD, "linR": linR, "azimuth": azimuth, "totP": totP,
            "valid": physical_mask(M)}


def physical_mask(M: np.ndarray, tol: float = 1e-4, require_det: bool = False) -> np.ndarray:
    """True where M is physically realizable: the Cloude coherency matrix has no eigenvalue
    below -tol. M: (..., 4, 4) normalized.

    `require_det=True` also demands det(M) > 0, as the lab's Msk does. That extra test rejects
    strongly depolarizing pixels (det ~ 0, sign set by noise), so it is off by default.
    """
    H = coherency_matrix(np.nan_to_num(M))
    ok = (np.linalg.eigvalsh(H).min(axis=-1) >= -tol) & np.isfinite(M).all(axis=(-1, -2))
    if require_det:
        ok &= np.linalg.det(np.nan_to_num(M)) > 0
    return ok


# Pauli basis for the Cloude coherency matrix: H = 1/4 * sum_ij m_ij (sigma_i kron sigma_j*)
_PAULI = np.array([[[1, 0], [0, 1]],
                   [[1, 0], [0, -1]],
                   [[0, 1], [1, 0]],
                   [[0, -1j], [1j, 0]]], dtype=complex)
_BASIS = np.array([[np.kron(_PAULI[i], _PAULI[j].conj()) for j in range(4)] for i in range(4)])  # (4,4,4,4)


def coherency_matrix(M: np.ndarray) -> np.ndarray:
    """Cloude coherency matrix (..., 4, 4) Hermitian; its eigenvalues are >= 0 for physical M."""
    return 0.25 * np.einsum("...ij,ijkl->...kl", M.astype(complex), _BASIS)
