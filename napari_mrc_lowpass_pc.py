"""Launch the fast MRC viewer tuned for a local workstation / laptop.

- FFTs use all CPU cores. (Measured on an M3 Max: all 16 cores were ~5%
  faster than the 12 performance cores alone.)
- The forward FFT of the current tomogram is kept in RAM, so changing the
  low-pass cutoff only needs one inverse FFT.

Usage:
    python napari_mrc_lowpass_pc.py [volume.mrc ...] [--workers N] [--no-spectrum-cache]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def default_workers() -> int:
    return os.cpu_count() or 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast napari MRC low-pass viewer (PC profile).")
    parser.add_argument("paths", nargs="*", type=Path, help="Optional MRC files to open on launch.")
    parser.add_argument("--workers", type=int, default=None, help="FFT threads (default: all cores).")
    parser.add_argument(
        "--no-spectrum-cache",
        action="store_true",
        help="Do not keep the FFT in RAM (saves ~2 GB per 1k x 1k x 512 tomogram, slower cutoff changes).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workers = args.workers or default_workers()

    # Must be set before NumPy/SciPy are imported.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, str(workers))
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

    from napari_mrc_lowpass_fast import RuntimeProfile, run_viewer

    profile = RuntimeProfile(name="PC", workers=workers, cache_spectrum=not args.no_spectrum_cache)
    run_viewer(args.paths, profile)


if __name__ == "__main__":
    main()
