import numpy as np
import pytest

from eisgreis.io.mdoc import parse_mdoc, resolve_tilt_image_path, sort_entries_by_tilt_angle
from eisgreis.io.mrc import extract_voxel_size, read_mrc_layer_data, read_voxel_size_array, read_voxel_size_from_path
from eisgreis.io.star import load_points_from_star, sanitize_star_token, save_points_to_star


def test_mrc_voxel_size_and_data(mrc):
    path = mrc("tomo.mrc", shape=(8, 16, 12), voxel_size=6.2)
    assert read_voxel_size_from_path(path) == pytest.approx(6.2, abs=1e-4)
    assert read_voxel_size_array(path).shape == (3,)
    data, add_kwargs, layer_type = read_mrc_layer_data(path)
    assert data.dtype == np.float32 and data.shape == (8, 16, 12)
    assert add_kwargs == {} and layer_type == "image"


def test_mrc_without_voxel_size_and_missing(tmp_path, mrc):
    assert extract_voxel_size(np.array([0.0, 0.0, 0.0])) == 1.0
    assert extract_voxel_size(np.array([2.0, 4.0, np.nan])) == 3.0
    assert read_voxel_size_from_path(tmp_path / "absent.mrc") is None
    with pytest.raises(ValueError, match="Unsupported"):
        read_mrc_layer_data(tmp_path / "picture.png")


def test_star_round_trip_3d(tmp_path):
    coords = np.array([[10.0, 20.5, 30.25], [1.0, 2.0, 3.0]])
    star = tmp_path / "stars" / "tomo__ribosome.star"
    save_points_to_star(coords, star, "ribosome", box_size_angstrom=310.0)
    loaded, box = load_points_from_star(star)
    np.testing.assert_allclose(loaded, coords, atol=1e-6)
    assert box == pytest.approx(310.0)
    assert "_rlnCoordinateZ #1" in star.read_text()


def test_star_round_trip_2d_and_empty(tmp_path):
    coords = np.array([[5.0, 6.0]])
    star = tmp_path / "a.star"
    save_points_to_star(coords, star, "dots")
    loaded, box = load_points_from_star(star)
    np.testing.assert_allclose(loaded, coords)
    assert box is None
    empty = tmp_path / "empty.star"
    save_points_to_star(np.empty((0, 3)), empty, "none")
    loaded, box = load_points_from_star(empty)
    assert loaded.size == 0 and box is None


def test_star_token():
    assert sanitize_star_token(" small ribosome ") == "small_ribosome"
    assert sanitize_star_token("***") == "annotations"


MDOC = """PixelSpacing = 1.7
[ZValue = 0]
TiltAngle = 0.0
SubFramePath = X:\\frames\\TS_01_000_0.0.tif
[ZValue = 1]
TiltAngle = 3.0
SubFramePath = X:\\frames\\TS_01_001_3.0.tif
[ZValue = 2]
TiltAngle = -3.0
SubFramePath = X:\\frames\\TS_01_002_-3.0.tif
"""


def test_mdoc_order_and_matching(tmp_path):
    mdoc = tmp_path / "TS_01.mdoc"
    mdoc.write_text(MDOC)
    tilts = tmp_path / "tilts"
    tilts.mkdir()
    for name in ("TS_01_000_0.0.mrc", "TS_01_001_3.0.mrc", "TS_01_002_-3.0.mrc"):
        (tilts / name).write_bytes(b"")
    entries = sort_entries_by_tilt_angle(parse_mdoc(mdoc))
    assert [e.tilt_angle for e in entries] == [-3.0, 0.0, 3.0]
    assert [e.z_value for e in entries] == [2, 0, 1]
    assert resolve_tilt_image_path(entries[0], tilts).name == "TS_01_002_-3.0.mrc"
