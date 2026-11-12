"""PDDN (Polarimetric Denoising Diffusion Network, Moriconi et al. 2024): inference wrapper, no training.

Uses the lab's own implementation (fastPDDN.py = newest libmpMPIdenoisePDDN, from the fastMMPT / HORAO code) for the network and the
overlap-and-add tiling, but builds the model directly so it runs on any torch device (the lab's MPI_PDDN class
hard-codes .cuda()). Behaviour matches the lab's IMPv2 pipeline (mmpt.processingmm, IMPv2 branch):
  - input: the 16 raw intensities of one frame, (H, W, 16), NOT dark-subtracted, no variance stabilisation;
  - per-image min/max scaling to [-1, 1], tiles of `patch_size` with overlap `hop`, Hann-weighted overlap-add;
  - reverse diffusion from `tstr` to 0 (tstr = 1: single deterministic step, the lab's default);
  - the network weights stored under 'model' in the checkpoint (as the lab's loader), or the EMA copy if asked.

Requires torch, einops, ema_pytorch (imported by the lab library), and the path to libmpMPIdenoisePDDN.py.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import time
from pathlib import Path

import numpy as np


def _import_lab_library(lib_path: str | Path):
    """Import libmpMPIdenoisePDDN.py from a file or from the folder that contains it."""
    lib_path = Path(lib_path)
    if lib_path.is_dir():
        for name in ("fastPDDN.py", "libmpMPIdenoisePDDN.py"):
            if (lib_path / name).is_file():
                lib_path = lib_path / name
                break
    if not lib_path.is_file():
        raise FileNotFoundError(f"PDDN library not found at {lib_path} (set pddn_lib in configs/paths.yaml)")
    spec = importlib.util.spec_from_file_location("libmpMPIdenoisePDDN", lib_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["libmpMPIdenoisePDDN"] = mod
    spec.loader.exec_module(mod)
    return mod


class PDDN:
    """model = PDDN(weights, lib_path, patch_size=128); I_den, seconds = model.denoise(I_raw, tstr=1)"""

    def __init__(self, weights: str | Path, lib_path: str | Path, patch_size: int = 128, device: str | None = None,
                 use_ema: bool = False, timesteps: int = 1000):
        import torch
        self.torch = torch
        lib = _import_lab_library(lib_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        unet = lib.MPI_PDDN_Unet(dim=64, channels=16, dim_mults=(1, 2, 4, 8))
        self.diffusion = lib.MPI_PDDN_GaussianDiffusion(unet, channels=16, image_size=int(patch_size),
                                                        timesteps=timesteps, loss_type="l1")
        data = torch.load(str(weights), map_location="cpu", weights_only=False)
        if use_ema:
            state = {k[len("ema_model."):]: v for k, v in data["ema"].items() if k.startswith("ema_model.")}
        else:
            state = data["model"]
        self.diffusion.load_state_dict(state)
        self.diffusion.to(self.device).eval()
        self.patch_size = int(patch_size)
        import inspect
        self._numpy_kw = "return_numpy" in inspect.signature(self.diffusion.img_denoise_OLA).parameters
        if not self._numpy_kw and self.device.type != "cuda":
            raise RuntimeError("This version of the PDDN library only runs on a CUDA GPU. Point pddn_lib to the newer "
                               "FASTMMPT/src/fastmmpt/application/denoising/fastPDDN.py to run on CPU.")
        self.info = {"weights": str(weights), "step": data.get("step"), "summary": data.get("summary"),
                     "device": str(self.device), "use_ema": use_ema}

    def denoise(self, I: np.ndarray, tstr: int = 1, hop: float = 0.25, seed: int | None = 0):
        """I: (H, W, 16) raw intensities. Returns (denoised (H, W, 16) float64, seconds incl. GPU sync)."""
        torch = self.torch
        if seed is not None:
            torch.manual_seed(seed)                      # only matters for tstr > 1 (stochastic steps)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):     # the lab code prints per call
            kw = {"return_numpy": True} if self._numpy_kw else {}
            out, _ = self.diffusion.img_denoise_OLA(np.ascontiguousarray(I, dtype=np.float32), tstr=int(tstr),
                                                    tend=0, hop=hop, verboseFlag=False, **kw)
            if torch.is_tensor(out):
                out = out.detach().cpu().numpy()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return out.astype(np.float64), time.perf_counter() - t0


_CACHE: dict = {}


def get_model(weights, lib_path, patch_size=128, device=None, use_ema=False) -> PDDN:
    """Load once per (weights, patch size, device, ema) and reuse."""
    key = (str(weights), int(patch_size), device, bool(use_ema))
    if key not in _CACHE:
        model = PDDN(weights, lib_path, patch_size=patch_size, device=device, use_ema=use_ema)
        # warm-up: the first GPU call includes one-off costs (cuDNN autotuning, memory allocation), which would
        # otherwise be counted in the timing of the first image
        model.denoise(np.random.default_rng(0).uniform(0, 255, (512, 608, 16)), tstr=2)
        _CACHE[key] = model
    return _CACHE[key]
