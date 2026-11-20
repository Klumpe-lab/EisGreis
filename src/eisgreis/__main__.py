"""``eisgreis [files…]``: open MRC volumes in EisGreis.

Run it in the folder where the comments CSV, the particle STARs and the PNGs should go. ``--profile`` picks the
machine's settings (``runtime.py``): ``hpc`` in a cluster's remote desktop, ``pc`` on a workstation or laptop; without
it, ``hpc`` where SLURM is present.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args(argv=None) -> argparse.Namespace:
    from eisgreis.runtime import PROFILES

    parser = argparse.ArgumentParser(
        prog="eisgreis", description="Open MRC volumes in napari and interactively apply a low-pass filter."
    )
    parser.add_argument("paths", nargs="*", type=Path, help="MRC files to open on launch; the first is shown.")
    parser.add_argument("--profile", choices=PROFILES, help="machine settings (default: hpc under SLURM, else pc)")
    parser.add_argument("--workers", type=int, default=None, help="FFT threads (default: the profile's CPUs)")
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument(
        "--spectrum-cache",
        dest="spectrum_cache",
        action="store_true",
        default=None,
        help="Keep the FFT in RAM (faster cutoff changes, ~2 GB more per 1k x 1k x 512 tomogram).",
    )
    cache.add_argument(
        "--no-spectrum-cache", dest="spectrum_cache", action="store_false", help="Do not keep the FFT in RAM."
    )
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    from eisgreis import __version__

    if args.version:
        print(f"eisgreis {__version__}")
        return
    from eisgreis.runtime import check_display, default_profile_name, make_profile, report_opengl

    name = args.profile or default_profile_name()
    if name == "hpc":
        check_display()
    # Before NumPy and SciPy are imported: they read the thread limits make_profile sets.
    profile = make_profile(name, args.workers, args.spectrum_cache)
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    from eisgreis.app import run_viewer

    run_viewer(args.paths, profile, on_ready=report_opengl if name == "hpc" else None)


if __name__ == "__main__":
    main()
