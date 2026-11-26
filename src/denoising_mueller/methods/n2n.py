"""Noise2Noise (Lehtinen et al., ICML 2018) for imaging Mueller polarimetry: network, Mueller algebra in torch,
checkpoint loading and inference.

Why this is a valid Noise2Noise setup
-------------------------------------
Noise2Noise trains f(x_i) -> x_j with an L2 loss, where x_i and x_j are two noisy observations of the same scene
whose noise is independent and has zero mean around the same clean signal s: E[x_j | x_i] = s. The minimizer of
E||f(x_i) - x_j||^2 is then the same as with clean targets. Our observations are the 16 registered raw frames of a
static acquisition (shot + read noise, independent from frame to frame). Two levels are possible:

  level "intensity": x = one raw frame, 16 intensities in counts, dark NOT subtracted (exactly the input of the
                     classical intensity path and of PDDN; the dark offset is a constant, so E[x_j] = s + dark).
  level "mm":        x = unnormalized Mueller matrix of one frame, M = A^-1 (I - dark) W^-1 (lab layout, 16
                     elements). M is LINEAR in I, so its frame-to-frame noise is still zero-mean and independent
                     between frames: a valid Noise2Noise target.

Never use as targets: the normalized matrix nM = M / m11, or the polarimetric maps (retardance, depolarization,
diattenuation, azimuth). They are nonlinear functions of the data, so their noise is biased (E[nM_j] != nM_clean)
and the network would learn that bias (the same reason Lehtinen et al. could not tone-map their targets).

Pixels where a frame is saturated are excluded from the loss (clipping removes part of the noise distribution,
so E[x_j] != s there; Lehtinen et al. section 3.2). Registration wrap-around strips are excluded too.

Network: the U-Net of Lehtinen et al. (appendix: 5 poolings, 48 features, leaky ReLU 0.1, no normalization),
with 16 input/output channels and a residual connection (the network predicts the correction to its input).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:                                  # the rest of the package does not need torch
    torch = None
    nn = None

LEVELS = ("intensity", "mm")


# ----------------------------------------------------------------------------------------------- network
if nn is not None:
    class _Conv(nn.Sequential):
        def __init__(self, cin, cout):
            super().__init__(nn.Conv2d(cin, cout, 3, padding=1), nn.LeakyReLU(0.1, inplace=True))

    class UNet(nn.Module):
        """U-Net of Noise2Noise (Lehtinen et al. 2018, appendix A.1), in/out = `channels` (16 here).
        Input height and width must be multiples of 32 (5 poolings); N2NModel pads automatically."""

        def __init__(self, channels: int = 16, width: int = 48, residual: bool = True):
            super().__init__()
            w, w2 = width, 2 * width
            self.residual = residual
            self.enc0 = nn.Sequential(_Conv(channels, w), _Conv(w, w))
            self.enc = nn.ModuleList([_Conv(w, w) for _ in range(5)])          # enc_conv2..6
            self.dec5 = nn.Sequential(_Conv(2 * w, w2), _Conv(w2, w2))
            self.dec = nn.ModuleList([nn.Sequential(_Conv(w2 + w, w2), _Conv(w2, w2)) for _ in range(3)])
            self.dec1 = nn.Sequential(_Conv(w2 + channels, 64), _Conv(64, 32))
            self.out = nn.Conv2d(32, channels, 3, padding=1)                    # linear output

        def forward(self, x):
            skips = [x]
            h = F.max_pool2d(self.enc0(x), 2)                                   # pool1
            skips.append(h)
            for k, conv in enumerate(self.enc):
                h = conv(h)
                if k < 4:
                    h = F.max_pool2d(h, 2)                                      # pool2..pool5
                    skips.append(h)
            # h: enc_conv6 output at 1/32 resolution; skips = [x, pool1, pool2, pool3, pool4, pool5]
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            h = self.dec5(torch.cat([h, skips[4]], 1))
            for k, dec in enumerate(self.dec):                                  # concat pool3, pool2, pool1
                h = F.interpolate(h, scale_factor=2, mode="nearest")
                h = dec(torch.cat([h, skips[3 - k]], 1))
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            h = self.out(self.dec1(torch.cat([h, skips[0]], 1)))
            return x + h if self.residual else h


def pad_to_multiple(x, m: int = 32):
    """Reflect-pad a (B, C, H, W) tensor on the bottom/right to multiples of m. Returns (padded, (H, W))."""
    h, w = x.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, (h, w)


# ----------------------------------------------------------------------------------- Mueller algebra (torch)
def vec_to_mat_t(v):
    """(B, 16, H, W) lab vectors (column-major: v[r + 4c] = m[r, c]) -> (B, 4, 4, H, W) with m[:, r, c]."""
    b, _, h, w = v.shape
    return v.reshape(b, 4, 4, h, w).transpose(1, 2)


def mat_to_vec_t(m):
    b, _, _, h, w = m.shape
    return m.transpose(1, 2).reshape(b, 16, h, w)


def mm_from_intensities_t(I, Ainv, Winv, dark=None):
    """M = A^-1 (I - dark) W^-1 per pixel, lab layout in and out.
    I: (B, 16, H, W) raw counts; Ainv, Winv: (B, 4, 4, H, W) inverse matrices; dark: (B, 16, H, W) or None."""
    if dark is not None:
        I = I - dark
    Im = vec_to_mat_t(I)
    M = torch.einsum("brkhw,bkchw->brchw", Ainv, Im)                           # A^-1 I
    M = torch.einsum("brkhw,bkchw->brchw", M, Winv)                             # (A^-1 I) W^-1
    return mat_to_vec_t(M)


def normalized_t(M):
    """nM = M / m11 (element 0 of the lab layout), as in reconstruct_mm (no clamping: dark pixels are masked)."""
    return M / M[:, :1]


def inverse_mats(X: np.ndarray) -> np.ndarray:
    """(H, W, 16) lab vectors -> (4, 4, H, W) float32 inverse matrices m[r, c, y, x]."""
    from ..polarimetry.reconstruction import vec_to_mat
    inv = np.linalg.inv(vec_to_mat(X))                                          # (H, W, 4, 4)
    return np.ascontiguousarray(np.moveaxis(inv, (2, 3), (0, 1))).astype(np.float32)


# ------------------------------------------------------------------------------------------- inference
class N2NModel:
    """Trained Noise2Noise denoiser.  model = N2NModel("runs/n2n/slices_test_intensity/best.pt")
    out, seconds = model.denoise(x)   with x (H, W, 16): raw counts (level intensity) or unnormalized M (level mm).
    """

    def __init__(self, checkpoint: str | Path, device: str | None = None):
        if torch is None:
            raise ImportError("Noise2Noise needs PyTorch")
        ck = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        self.path = Path(checkpoint)
        self.level = ck["level"]
        self.scale = float(ck["scale"])
        self.meta = ck.get("meta", {})
        self.split = ck["split"]                      # {"train": [...], "val": [...], "test": [...]} acquisition names
        self.iteration = ck.get("iteration")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = UNet(16, ck["width"], ck["residual"])
        self.net.load_state_dict(ck["state_dict"])
        self.net.to(self.device).eval()

    def check_not_trained_on(self, name: str):
        """Leak guard: refuse to evaluate on an acquisition the network was trained or validated on."""
        if name in self.split["train"] or name in self.split["val"]:
            raise RuntimeError(f"{name} was used for training/validation of {self.path}: not a valid test")

    def forward(self, x):
        """x: (B, 16, H, W) tensor in data units on self.device -> same shape, data units."""
        with torch.no_grad():
            xp, (h, w) = pad_to_multiple(x / self.scale, 32)
            return self.net(xp)[..., :h, :w] * self.scale

    def denoise(self, x: np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(np.moveaxis(x, -1, 0), dtype=np.float32))[None].to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = self.forward(t)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        return np.moveaxis(y[0].cpu().numpy().astype(np.float64), 0, -1), dt


_CACHE: dict = {}


def get_model(checkpoint, device=None) -> N2NModel:
    """Load once and reuse; one warm-up pass so one-off GPU costs are not timed on the first image."""
    key = (str(checkpoint), device)
    if key not in _CACHE:
        m = N2NModel(checkpoint, device)
        m.denoise(np.random.default_rng(0).uniform(0, 100, (512, 608, 16)))
        _CACHE[key] = m
    return _CACHE[key]
