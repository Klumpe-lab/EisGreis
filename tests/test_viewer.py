"""The viewer through eisgreis.api, on a napari window."""

import numpy as np
import pytest


def z_index(viewer) -> int:
    layer = viewer.napari.layers[-1]
    return int(round(viewer.napari.dims.point[0] / layer.scale[0]))


def test_open_centres_the_middle_slice(viewer, mrc):
    a = mrc("a.mrc", shape=(32, 24, 24), voxel_size=8.0)
    b = mrc("b.mrc", shape=(10, 24, 24), voxel_size=3.0)
    viewer.open(a)
    assert viewer.current == a.resolve() and viewer.shape == (32, 24, 24)
    assert viewer.napari.dims.point[0] == pytest.approx(16 * 8.0)
    viewer.open(b, title="b")
    assert viewer.napari.dims.point[0] == pytest.approx(5 * 3.0)
    assert viewer.napari.title == "b"
    assert [p.name for p in viewer.tomograms] == ["a.mrc", "b.mrc"]


def test_scale_bar_reads_nanometres(viewer, mrc):
    viewer.open(mrc("a.mrc", shape=(8, 64, 64), voxel_size=10.0))
    overlay = viewer.napari.canvas.overlays.scale_bar
    assert overlay.visible
    canvas = viewer.napari.window._qt_viewer.canvas
    visuals = canvas._viewer_overlay_to_visual[overlay]
    text = visuals[0].node.text.text
    assert text.endswith(" nm"), text


def test_particles_stay_with_their_tomogram(viewer, mrc, tmp_path):
    a, b = mrc("a.mrc"), mrc("b.mrc")
    viewer.open(a)
    controller = viewer._c
    layer = controller.particles_dock.create_layer("ribosome")
    layer.add([8.0, 10.0, 12.0])  # voxels
    assert controller.viewer.dims.thickness[0] >= 200.0  # thick enough to draw the 200 Å box out of the plane
    viewer.open(b)
    assert len(layer.data) == 0
    viewer.open(a)
    np.testing.assert_allclose(layer.data, [[8.0, 10.0, 12.0]])
    viewer.save()
    star = tmp_path / "start" / "particle_stars" / "a__ribosome.star"
    assert star.is_file() and "64.000000 80.000000 96.000000" in star.read_text()


def test_pixel_size_change_recalibrates(viewer, mrc):
    viewer.open(mrc("a.mrc", voxel_size=8.0))
    controller = viewer._c
    controller.set_pixel_size(4.0)
    assert viewer.pixel_size == 4.0
    assert tuple(controller.filtered_layer.scale) == (4.0, 4.0, 4.0)
    assert controller.measurement_layer.scale[0] == 10.0


def test_swap_volume_keeps_the_grid(viewer, mrc):
    a = mrc("a.mrc", shape=(8, 16, 16))
    same = mrc("denoised.mrc", shape=(8, 16, 16), seed=3)
    other = mrc("other.mrc", shape=(8, 16, 20))
    viewer.open(a)
    viewer._c.swap_volume(same, label="denoised")
    assert "denoised" in viewer._c.source_layer.name and viewer.current == a.resolve()
    with pytest.raises(ValueError, match="the tomogram"):
        viewer._c.swap_volume(other)


def test_reference_layers_follow_their_tomogram(viewer, mrc):
    a, b = mrc("a.mrc", shape=(8, 16, 16)), mrc("b.mrc", shape=(8, 16, 16))
    viewer.open(a)
    viewer.reference.show_picks("picks", [[4, 8, 8]], box_angstrom=80.0, color="#00ff00")
    names = [layer.name for layer in viewer.napari.layers]
    assert "picks" in names
    viewer.open(b)
    assert "picks" not in [layer.name for layer in viewer.napari.layers]
    viewer.open(a)
    assert "picks" in [layer.name for layer in viewer.napari.layers]
    assert viewer.reference.summary()["picks"] == ["picks"]


def test_3d_draws_segmentations_as_surfaces(viewer, tmp_path, mrc):
    from tests.conftest import write_mrc

    a = mrc("a.mrc", shape=(32, 64, 64), voxel_size=10.0)
    values = np.zeros((16, 32, 32), dtype=np.float32)
    values[4:12, 8:24, 8:24] = 1.0  # 2048 voxels: more than view3d's specks
    seg = write_mrc(tmp_path / "seg.mrc", values, 20.0)
    viewer.open(a)
    viewer.reference.show_segmentation(seg, level=0.5, color="#f50057", voxel_size=20.0, name="blob")
    viewer.napari.dims.ndisplay = 3
    surfaces = [layer.name for layer in viewer.napari.layers if layer.name.endswith("(3-D surface)")]
    assert surfaces == ["blob (3-D surface)"]
    viewer.napari.dims.ndisplay = 2
    assert not any(layer.name.endswith("(3-D surface)") for layer in viewer.napari.layers)


def test_docks_are_named(viewer):
    docks = viewer.napari.window.dock_widgets
    assert {"MRC Filter", "Tomogram Comments", "Particle Annotations", "Export PNG"} <= set(docks)
