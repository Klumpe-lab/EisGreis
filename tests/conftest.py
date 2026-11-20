"""Shared fixtures. Tests that need a napari window run headless (``QT_QPA_PLATFORM=offscreen``) on an X display
for OpenGL: ``xvfb-run`` in CI."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

FIXTURES = Path(__file__).parent / "fixtures"


def write_mrc(path: Path, data: np.ndarray, voxel_size: float = 8.0) -> Path:
    import mrcfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(np.asarray(data, dtype=np.float32))
        mrc.voxel_size = voxel_size
    return path


@pytest.fixture
def mrc(tmp_path):
    """mrc(name, shape=(16, 32, 32), voxel_size=8.0, folder=tmp_path): a noise volume written as an MRC."""

    def make(name: str, shape=(16, 32, 32), voxel_size: float = 8.0, folder: Path | None = None, seed: int = 0):
        data = np.random.default_rng(seed).normal(size=shape).astype(np.float32)
        return write_mrc((folder or tmp_path) / name, data, voxel_size)

    return make


@pytest.fixture
def viewer(tmp_path):
    """An EisGreis window (through the API, as a driver gets it), saving into tmp_path/start; closed afterwards."""
    from eisgreis import api

    profile = api.RuntimeProfile(name="TEST", workers=2, cache_spectrum=True)
    v = api.build(profile, outputs=tmp_path / "start")
    yield v
    v.napari.close()
