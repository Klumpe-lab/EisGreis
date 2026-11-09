"""Launch the fast MRC viewer tuned for an HPC remote-desktop session.

- Uses only the CPUs the job was given (SLURM_CPUS_PER_TASK or CPU affinity),
  not every core on the shared node.
- Keeps the cached FFT only when the job has enough memory for it.
- Checks that a display exists and warns when OpenGL is software-rendered
  (no GPU), which makes 3D rendering slow; 2D slice browsing is still fine.

Usage (inside the remote desktop session):
    python napari_mrc_lowpass_hpc.py [volume.mrc ...] [--workers N] [--no-spectrum-cache | --spectrum-cache]

With a GPU visualisation node and VirtualGL:
    vglrun python napari_mrc_lowpass_hpc.py volume.mrc
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Below this job memory (GB) the FFT cache is off by default.
SPECTRUM_CACHE_MIN_MEMORY_GB = 24.0


def allocated_cpus() -> int:
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm and slurm.isdigit() and int(slurm) > 0:
        return int(slurm)
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(len(os.sched_getaffinity(0)), 1)
        except OSError:
            pass
    return os.cpu_count() or 1


def allocated_memory_gb(cpus: int) -> Optional[float]:
    """Job memory from SLURM, or the cgroup limit; None if unknown."""
    per_node = os.environ.get("SLURM_MEM_PER_NODE")
    if per_node and per_node.isdigit():
        return int(per_node) / 1024.0
    per_cpu = os.environ.get("SLURM_MEM_PER_CPU")
    if per_cpu and per_cpu.isdigit():
        return int(per_cpu) * cpus / 1024.0
    for cgroup_file in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = Path(cgroup_file).read_text().strip()
        except OSError:
            continue
        if raw.isdigit() and int(raw) < 1 << 60:
            return int(raw) / 1024.0**3
    return None


def check_display() -> None:
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise SystemExit(
            "No display found (DISPLAY is not set).\n"
            "Start this from a terminal inside the remote desktop session "
            "(e.g. Open OnDemand / ThinLinc / VNC), not from a plain ssh shell."
        )


def report_opengl(viewer) -> None:
    try:
        from vispy.gloo import gl

        scene_canvas = viewer.window._qt_viewer.canvas._scene_canvas
        scene_canvas._backend._vispy_set_current()
        renderer = str(gl.glGetParameter(gl.GL_RENDERER))
    except Exception as exc:  # never block startup on this check
        print(f"[napari-lorenz] Could not query OpenGL renderer: {exc}", flush=True)
        return
    print(f"[napari-lorenz] OpenGL renderer: {renderer}", flush=True)
    if any(token in renderer.lower() for token in ("llvmpipe", "softpipe", "swrast", "software")):
        print(
            "[napari-lorenz] WARNING: software OpenGL (no GPU). 2D slice browsing works, "
            "3D volume rendering will be slow.\n"
            "               If your cluster has GPU visualisation nodes, start with: "
            "vglrun python napari_mrc_lowpass_hpc.py ...",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast napari MRC low-pass viewer (HPC profile).")
    parser.add_argument("paths", nargs="*", type=Path, help="Optional MRC files to open on launch.")
    parser.add_argument("--workers", type=int, default=None, help="FFT threads (default: CPUs allocated to the job).")
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument("--spectrum-cache", dest="spectrum_cache", action="store_true", default=None,
                       help="Keep the FFT in RAM (faster cutoff changes, ~2 GB more per 1k x 1k x 512 tomogram).")
    cache.add_argument("--no-spectrum-cache", dest="spectrum_cache", action="store_false",
                       help="Do not keep the FFT in RAM.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_display()

    workers = args.workers or allocated_cpus()
    memory_gb = allocated_memory_gb(workers)
    if args.spectrum_cache is None:
        cache_spectrum = memory_gb is None or memory_gb >= SPECTRUM_CACHE_MIN_MEMORY_GB
    else:
        cache_spectrum = args.spectrum_cache
    mem_text = f"{memory_gb:.0f} GB" if memory_gb is not None else "unknown"
    print(f"[napari-lorenz] HPC job: {workers} CPUs, memory limit {mem_text}", flush=True)

    # Must be set before NumPy/SciPy are imported; stops BLAS/OpenMP from
    # starting one thread per core of the whole node.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, str(workers))
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

    from napari_mrc_lowpass_fast import RuntimeProfile, run_viewer

    profile = RuntimeProfile(name="HPC", workers=workers, cache_spectrum=cache_spectrum)
    run_viewer(args.paths, profile, on_viewer_ready=report_opengl)


if __name__ == "__main__":
    main()
