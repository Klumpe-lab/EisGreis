"""MRC helpers for the fast viewer.

Same API as ``shared_mrc`` but the voxel-size lookups read only the header
(``header_only=True``) instead of loading the whole volume a second time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import mrcfile
import numpy as np

from shared_filters import extract_voxel_size
from shared_mrc import read_mrc_layer_data  # unchanged: mmap + one float32 conversion

__all__ = ["read_mrc_layer_data", "read_voxel_size_array", "read_voxel_size_from_path"]


def read_voxel_size_array(path: Path) -> np.ndarray:
    with mrcfile.open(path, permissive=True, header_only=True) as handle:
        return np.asarray(tuple(handle.voxel_size.tolist()), dtype=np.float32)


def read_voxel_size_from_path(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    try:
        voxel_size = read_voxel_size_array(path)
    except Exception:
        return None
    return extract_voxel_size(voxel_size)
