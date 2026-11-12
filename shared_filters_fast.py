"""Memory- and CPU-efficient Fourier low-pass filtering for the fast viewer.

Same filter shapes as ``shared_filters.lowpass_mask`` but:

- uses the real-input FFT (``rfftn``), which halves memory and time;
- works on the unshifted spectrum, so no ``fftshift``/``ifftshift`` copies;
- never builds a full 3D frequency grid: the mask is computed slab by slab
  from three small 1D frequency axes;
- runs the FFT and the mask multiplication on several threads;
- optionally caches the forward FFT so changing the cutoff only needs one
  inverse FFT.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import scipy.fft as fft

from shared_filters import extract_voxel_size, resolution_bounds  # re-exported

__all__ = [
    "LowPassFilter",
    "available_cpu_count",
    "contrast_limits",
    "extract_voxel_size",
    "resolution_bounds",
]

FILTER_CHOICES = ("hard", "gaussian", "butterworth", "cosine")


def available_cpu_count() -> int:
    """CPUs this process may actually use (respects SLURM and CPU affinity)."""
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm and slurm.isdigit() and int(slurm) > 0:
        return int(slurm)
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(len(os.sched_getaffinity(0)), 1)
        except OSError:
            pass
    return os.cpu_count() or 1


def _mask_from_f2(f2: np.ndarray, resolution_angstrom: float, filter_type: str) -> np.ndarray:
    """Low-pass mask from squared spatial frequency (1/Å²). Mirrors shared_filters.lowpass_mask."""
    cutoff = 1.0 / max(resolution_angstrom, 1e-3)
    cutoff = max(cutoff, 1e-6)
    inv_c2 = np.float32(1.0 / (cutoff * cutoff))

    if filter_type == "hard":
        return (f2 <= np.float32(cutoff * cutoff)).astype(np.float32)

    if filter_type == "butterworth":
        ratio2 = f2 * inv_c2
        np.multiply(ratio2, ratio2, out=ratio2)  # ratio**4 (order 2)
        ratio2 += np.float32(1.0)
        return np.reciprocal(ratio2, out=ratio2)

    if filter_type == "cosine":
        freq = np.sqrt(f2)
        start = 0.8 * cutoff
        stop = cutoff
        mask = np.ones_like(freq)
        mask[freq >= stop] = 0.0
        transition = (freq > start) & (freq < stop)
        t = (freq[transition] - start) / max(stop - start, 1e-6)
        mask[transition] = 0.5 * (1.0 + np.cos(np.pi * t))
        return mask

    # gaussian (also the fallback, as in shared_filters)
    out = f2 * np.float32(-0.5 * inv_c2)
    return np.exp(out, out=out)


class LowPassFilter:
    """Holds one volume and filters it repeatedly at different cutoffs."""

    def __init__(
        self,
        data: np.ndarray,
        voxel_size_angstrom: float,
        workers: Optional[int] = None,
        cache_spectrum: bool = True,
    ) -> None:
        self.data = np.ascontiguousarray(data, dtype=np.float32)
        self.shape = self.data.shape
        self.workers = max(int(workers or available_cpu_count()), 1)
        self.cache_spectrum = cache_spectrum
        self._spectrum: Optional[np.ndarray] = None
        self._axes2: list[np.ndarray] = []
        self.set_voxel_size(voxel_size_angstrom)

    def set_voxel_size(self, voxel_size_angstrom: float) -> None:
        """Rebuild the (tiny) per-axis frequency tables; the cached FFT stays valid."""
        self.voxel_size_angstrom = float(voxel_size_angstrom)
        d = self.voxel_size_angstrom
        axes = [np.fft.fftfreq(n, d=d) for n in self.shape[:-1]]
        axes.append(np.fft.rfftfreq(self.shape[-1], d=d))
        self._axes2 = [(a * a).astype(np.float32) for a in axes]

    def release(self) -> None:
        self._spectrum = None

    def _forward(self) -> np.ndarray:
        if self._spectrum is not None:
            return self._spectrum
        spectrum = fft.rfftn(self.data, workers=self.workers)
        if self.cache_spectrum:
            self._spectrum = spectrum
        return spectrum

    def _f2_slab(self, start: int, stop: int) -> np.ndarray:
        """Squared frequency for spectrum[start:stop] (first axis), broadcast to full slab."""
        ndim = len(self._axes2)
        f2 = None
        for axis, a2 in enumerate(self._axes2):
            part = a2[start:stop] if axis == 0 else a2
            shape = [1] * ndim
            shape[axis] = part.shape[0]
            part = part.reshape(shape)
            f2 = part if f2 is None else f2 + part
        return np.ascontiguousarray(np.broadcast_to(f2, (stop - start,) + tuple(len(a) for a in self._axes2[1:])))

    def apply(self, resolution_angstrom: float, filter_type: str) -> np.ndarray:
        spectrum = self._forward()
        # Work on a copy only if the original spectrum is cached for reuse.
        work = np.empty_like(spectrum) if self._spectrum is not None else spectrum

        n0 = spectrum.shape[0]
        n_slabs = min(n0, self.workers * 4)
        bounds = np.linspace(0, n0, n_slabs + 1, dtype=int)

        def run(i: int) -> None:
            a, b = int(bounds[i]), int(bounds[i + 1])
            if a == b:
                return
            mask = _mask_from_f2(self._f2_slab(a, b), resolution_angstrom, filter_type)
            np.multiply(spectrum[a:b], mask, out=work[a:b])

        if self.workers > 1 and n_slabs > 1:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                list(pool.map(run, range(n_slabs)))
        else:
            for i in range(n_slabs):
                run(i)

        filtered = fft.irfftn(work, s=self.shape, workers=self.workers, overwrite_x=True)
        del work
        return np.asarray(filtered, dtype=np.float32)


def contrast_limits(data: np.ndarray, max_samples: int = 4_000_000) -> tuple[float, float]:
    """1st/99th percentile on a regular subsample instead of the whole volume."""
    data = np.asarray(data)
    if data.size > max_samples and data.ndim > 0:
        step = max(int(np.ceil((data.size / max_samples) ** (1.0 / data.ndim))), 1)
        data = data[(slice(None, None, step),) * data.ndim]
    low, high = np.percentile(data, (1.0, 99.0))
    return float(low), float(high)
