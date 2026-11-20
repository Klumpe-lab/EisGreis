"""MRC volumes: memory-mapped data and the header's voxel size."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import mrcfile
import numpy as np

# The files the viewer opens as volumes (IMOD's tilt stacks and aligned stacks included).
MRC_SUFFIXES = (".mrc", ".mrcs", ".map", ".st", ".rec", ".preali", ".ali")


def read_mrc_layer_data(path: Path) -> tuple[np.ndarray, dict, str]:
    """The volume as float32, napari's ``add_image`` keywords for it, and the layer type ("image").

    The file is memory-mapped and converted once. A complex volume gives its amplitude, named so."""
    name = str(path)
    if not name.endswith(MRC_SUFFIXES):
        raise ValueError(f"Unsupported file type: {path}")
    data = mrcfile.mmap(name, mode="r", permissive=True).data
    if data is None:
        raise ValueError(f"No layer data returned for: {path}")
    add_kwargs: dict = {}
    if data.dtype in (np.complex64, np.complex128):
        data = np.abs(data)
        add_kwargs = {"name": "amplitude"}
    return np.asarray(data, dtype=np.float32), add_kwargs, "image"


def extract_voxel_size(voxel_size: np.ndarray) -> float:
    """One pixel size (Å) from the header's x, y, z: their mean over the valid ones; 1.0 when none is valid."""
    valid = voxel_size[np.isfinite(voxel_size) & (voxel_size > 0)]
    if valid.size == 0:
        return 1.0
    return float(valid.mean())


def read_voxel_size_array(path: Path) -> np.ndarray:
    """The header's voxel size (x, y, z in Å), read from the header alone."""
    with mrcfile.open(path, permissive=True, header_only=True) as handle:
        return np.asarray(tuple(handle.voxel_size.tolist()), dtype=np.float32)


def read_voxel_size_from_path(path: Path) -> Optional[float]:
    """The header's pixel size, or None when the file is missing or its header cannot be read."""
    if not Path(path).exists():
        return None
    try:
        voxel_size = read_voxel_size_array(path)
    except (OSError, ValueError):  # not an MRC header mrcfile can read: the caller decides what that means
        return None
    return extract_voxel_size(voxel_size)
