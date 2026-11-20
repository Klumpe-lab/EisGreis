"""The machine the viewer runs on: CPUs, memory, the FFT cache, the GL renderer.

Two profiles, chosen with ``eisgreis --profile``; without it, ``hpc`` where SLURM is present and ``pc`` elsewhere:

- ``pc``, a workstation or laptop: FFTs on every core, and the forward FFT of the tomogram on screen kept in memory,
  so a new cutoff needs one inverse FFT. (On an M3 Max all 16 cores were about 5 % faster than the 12 performance
  cores alone.)
- ``hpc``, a remote desktop on a cluster: only the CPUs the job was given (``SLURM_CPUS_PER_TASK`` or the CPU
  affinity), not every core of the shared node, and the FFT cache only when the job has 24 GB or more.

This module imports nothing heavy: the thread limits must be set before NumPy and SciPy are imported.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Below this job memory (GB) the FFT cache is off by default in the hpc profile.
SPECTRUM_CACHE_MIN_MEMORY_GB = 24.0

# The variables that cap BLAS and OpenMP threads, per profile.
THREAD_VARIABLES = {
    "pc": ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"),
    "hpc": ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"),
}
PROFILES = tuple(THREAD_VARIABLES)


def available_cpu_count() -> int:
    """CPUs this process may actually use (respects SLURM and CPU affinity)."""
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm and slurm.isdigit() and int(slurm) > 0:
        return int(slurm)
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(len(os.sched_getaffinity(0)), 1)
        except OSError:  # no affinity on this platform: count the cores instead
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
        except OSError:  # this cgroup layout is not the machine's: try the other
            continue
        if raw.isdigit() and int(raw) < 1 << 60:
            return int(raw) / 1024.0**3
    return None


def default_profile_name() -> str:
    return "hpc" if os.environ.get("SLURM_JOB_ID") else "pc"


@dataclass
class RuntimeProfile:
    """Machine-dependent performance settings."""

    name: str = "auto"
    workers: int = field(default_factory=available_cpu_count)
    cache_spectrum: bool = True
    contrast_samples: int = 4_000_000

    def describe(self) -> str:
        cache = "on" if self.cache_spectrum else "off"
        return f"profile={self.name} | FFT threads={self.workers} | spectrum cache={cache}"


def make_profile(
    name: Optional[str] = None, workers: Optional[int] = None, spectrum_cache: Optional[bool] = None
) -> RuntimeProfile:
    """The profile `name` (``pc`` or ``hpc``; None picks by SLURM) with the threads capped to its CPUs.

    Call it before NumPy is imported: the BLAS and OpenMP thread limits it sets are read at import."""
    name = name or default_profile_name()
    if name not in PROFILES:
        raise ValueError(f"unknown profile {name!r}: one of {', '.join(PROFILES)}")
    if name == "pc":
        workers = workers or os.cpu_count() or 1
        cache = True if spectrum_cache is None else spectrum_cache
    else:
        workers = workers or available_cpu_count()
        memory_gb = allocated_memory_gb(workers)
        cache = (
            (memory_gb is None or memory_gb >= SPECTRUM_CACHE_MIN_MEMORY_GB)
            if spectrum_cache is None
            else spectrum_cache
        )
        mem_text = f"{memory_gb:.0f} GB" if memory_gb is not None else "unknown"
        print(f"[eisgreis] HPC job: {workers} CPUs, memory limit {mem_text}", flush=True)
    for var in THREAD_VARIABLES[name]:
        os.environ.setdefault(var, str(workers))
    return RuntimeProfile(name=name.upper(), workers=workers, cache_spectrum=cache)


def check_display() -> None:
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise SystemExit(
            "No display found (DISPLAY is not set).\n"
            "Start this from a terminal inside the remote desktop session "
            "(e.g. Open OnDemand / ThinLinc / VNC), not from a plain ssh shell."
        )


def is_software_renderer(renderer: str) -> bool:
    return any(token in renderer.lower() for token in ("llvmpipe", "softpipe", "swrast", "software"))


def report_opengl(viewer) -> str:
    """Print the OpenGL renderer the canvas draws with, and a warning when it is software; returns it."""
    from eisgreis._napari import gl_renderer

    renderer = gl_renderer(viewer)
    print(f"[eisgreis] OpenGL renderer: {renderer}", flush=True)
    if is_software_renderer(renderer):
        print(
            "[eisgreis] WARNING: software OpenGL (no GPU). 2D slice browsing works, "
            "3D volume rendering will be slow.\n"
            "           For 3D, start EisGreis on a node with a full GPU: its launcher draws through VirtualGL there.",
            flush=True,
        )
    return renderer
