"""Fixtures. ``crboost/`` holds session files crboost wrote (its request writer and bundle builder, run on a small
project), with the folders replaced by ``${DATA}`` (the data tree) and ``${SESSION}`` (the session folder).
``crboost_fixture`` substitutes a test's own folders and builds the volumes the files name, so the viewer can open
them."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CRBOOST = Path(__file__).parent / "crboost"


def crboost_fixture(name: str, tmp_path: Path, *, make_volumes: bool = False) -> dict:
    """The fixture `name` with ${DATA} = tmp_path/data and ${SESSION} = tmp_path/session; with `make_volumes`, every
    MRC it names written (noise, on the grid the bundle gives)."""
    data, session = tmp_path / "data", tmp_path / "session"
    text = (CRBOOST / name).read_text().replace("${DATA}", str(data)).replace("${SESSION}", str(session))
    content = json.loads(text)
    if make_volumes:
        build_volumes(tmp_path)
    return content


def build_volumes(tmp_path: Path) -> dict:
    """The data tree the fixtures name: the project marker, the tomogram, its denoised twin, a coarser segmentation;
    the session folder with bundle.json. Returns the bundle."""
    from tests.conftest import write_mrc

    data, session = tmp_path / "data", tmp_path / "session"
    bundle = json.loads(
        (CRBOOST / "bundle.json").read_text().replace("${DATA}", str(data)).replace("${SESSION}", str(session))
    )
    project = Path(bundle["project"])
    project.mkdir(parents=True, exist_ok=True)
    (project / "project_params.json").write_text("{}")
    shape = tuple(bundle["shape_zyx"])
    rng = np.random.default_rng(7)
    for volume in bundle["volumes"]:
        if not Path(volume["path"]).exists():
            write_mrc(Path(volume["path"]), rng.normal(size=shape).astype(np.float32), bundle["voxel_size"])
    if not Path(bundle["tomogram"]).exists():
        write_mrc(Path(bundle["tomogram"]), rng.normal(size=shape).astype(np.float32), bundle["voxel_size"])
    for seg in bundle["segmentations"]:
        seg_shape = tuple(seg["shape_zyx"])
        values = np.zeros(seg_shape, dtype=np.float32)
        values[seg_shape[0] // 3 : 2 * seg_shape[0] // 3, 2:-2, 2:-2] = float(seg["full_scale"])
        write_mrc(Path(seg["path"]), values, seg["voxel_size"])
    session.mkdir(parents=True, exist_ok=True)
    (session / "bundle.json").write_text(json.dumps(bundle))
    return bundle
