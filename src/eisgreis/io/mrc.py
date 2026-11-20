from __future__ import annotations

from pathlib import Path
from typing import Optional

import mrcfile
import numpy as np
from napari_mrcfile_reader import napari_get_reader

from shared_filters import extract_voxel_size


def read_mrc_layer_data(path: Path) -> tuple[np.ndarray, dict, str]:
    reader = napari_get_reader(str(path))
    if reader is None:
        raise ValueError(f"Unsupported file type: {path}")

    layer_data = reader(str(path))
    if not layer_data:
        raise ValueError(f"No layer data returned for: {path}")

    data, add_kwargs, layer_type = layer_data[0]
    return np.asarray(data, dtype=np.float32), dict(add_kwargs), layer_type


def read_voxel_size_from_path(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    try:
        with mrcfile.open(path, permissive=True) as handle:
            voxel_size = np.asarray(tuple(handle.voxel_size.tolist()), dtype=np.float32)
    except Exception:
        return None

    return extract_voxel_size(voxel_size)


def read_voxel_size_array(path: Path) -> np.ndarray:
    with mrcfile.open(path, permissive=True) as handle:
        return np.asarray(tuple(handle.voxel_size.tolist()), dtype=np.float32)
