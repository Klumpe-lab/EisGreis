"""Reference layers from crboost: one tomogram's particle picks and segmentations.

crboost (the cryo-ET pipeline server) writes a bundle per tomogram. The viewer shows it while that tomogram is on
screen, beside your own particle layers, and never saves it: these layers are crboost's results to look at, not
annotations of yours. Load one with ``--bundle bundle.json`` on the command line or
``controller.load_bundle(path_or_dict)``.

The format (JSON):

    {
      "format": "crboost-napari-bundle", "version": 1,
      "tomogram": "/abs/path/tomogram.mrc",   # the volume the coordinates refer to
      "shape_zyx": [512, 1024, 1024], "voxel_size": 6.2,   # that volume's grid (Å)
      "picks": [
        {"name": "ribosome · candidates (tmextractcand) (182)", "species": "ribosome",
         "source": "candidates tmextractcand", "color": "#ff1744", "box_size": 310.0,
         "coordinates_zyx": [[z, y, x], ...]}           # voxels of "tomogram"
      ],
      "segmentations": [
        {"name": "membrane (segment)", "path": "/abs/seg.mrc", "voxel_size": 6.2,
         "shape_zyx": [512, 1024, 1024], "level": 63.5, "full_scale": 127, "color": "#f50057",
         "visible": true}                               # shown where value > level (p = 0.5)
      ],
      "volumes": [                                     # pictures of this tomogram on its grid
        {"name": "reconstruction (tsReconstruct)", "kind": "reconstruction", "path": "/abs/rec.mrc"},
        {"name": "denoised · cryoCARE (denoisePredict)", "kind": "denoised", "path": "/abs/den.mrc"}
      ],
      "notes": ["what crboost left out, and why"]
    }

A "crboost" dock comes with a bundle. *Show* (with two or more ``volumes``) puts another volume in the tomogram's
place (``LowPassController.swap_volume``), low-passed at the current cutoff, while particles, comments and STAR
names stay with the tomogram that was opened; another tomogram starts again on the one it was opened as.
*3-D segmentations* chooses how 3-D draws them (view3d.py): smooth surfaces, one layer per feature, or merged into
one labels volume, the fastest and blocky.

Picks are voxel coordinates of the tomogram, so they follow the pixel size the filter dock sets, like your own
layers. In 2-D a pick draws as the cross-section of a sphere of its box size through the slice on screen (napari's
"rescale_spherical" projection). napari (0.9 on) only draws points inside the slice's thickness, so showing picks
widens the Z thickness to the largest box; images still show their single plane. A segmentation keeps its own grid: ``voxel_size`` relative to the tomogram's says how much coarser it is,
and both grids share the volume's corner.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import mrcfile
import napari
import numpy as np
from napari.utils.colormaps import Colormap
from napari.utils.notifications import show_warning

if TYPE_CHECKING:
    from napari_mrc_lowpass_fast import LowPassController

logger = logging.getLogger(__name__)

FORMAT = "crboost-napari-bundle"
VERSIONS = (1,)
KIND = "crboost"  # layer.metadata["annotation_kind"] of every layer made here
DEFAULT_BOX_ANGSTROM = 200.0


def read_bundle(source: Union[str, Path, dict]) -> dict:
    """The bundle from a path or an already parsed dict, checked for its format and version."""
    bundle = source if isinstance(source, dict) else json.loads(Path(source).expanduser().read_text())
    if bundle.get("format") != FORMAT or bundle.get("version") not in VERSIONS:
        raise ValueError(f"not a crboost bundle this viewer reads (format {bundle.get('format')!r}, "
                         f"version {bundle.get('version')!r})")
    if not bundle.get("tomogram"):
        raise ValueError("the bundle names no tomogram")
    return bundle


def _rgb(color: str) -> tuple[float, float, float]:
    """'#rrggbb' as 0–1 floats; anything else is grey (crboost always writes hex)."""
    c = str(color or "").lstrip("#")
    if len(c) != 6:
        return (0.7, 0.7, 0.7)
    return tuple(int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _mask_colormap(name: str, color: str) -> Colormap:
    """Transparent at the bottom of the contrast range, the feature's colour just above it: with the range starting
    at the bundle's level, a probability volume draws as a solid mask of where it passes p = 0.5."""
    r, g, b = _rgb(color)
    return Colormap(colors=[[r, g, b, 0.0], [r, g, b, 0.75], [r, g, b, 0.75]], controls=[0.0, 0.02, 1.0], name=name)


class BundleLayers:
    """crboost bundles by tomogram path, and the layers of the one for the tomogram on screen."""

    def __init__(self, controller: "LowPassController") -> None:
        self.c = controller
        self.bundles: dict[Path, dict] = {}
        self.layers: list = []
        self.files: list = []
        self.volume_box = None  # the "crboost" dock's widgets, made with the first bundle
        self.mode_box = None
        self.volume_dock = None
        self._volumes: dict[str, dict] = {}
        self._updating = False

    @property
    def viewer(self) -> napari.Viewer:
        return self.c.viewer

    def add(self, source: Union[str, Path, dict]) -> dict:
        """Keep a bundle for its tomogram; draw it now when that tomogram is on screen."""
        bundle = read_bundle(source)
        path = Path(bundle["tomogram"]).expanduser().resolve()
        self.bundles[path] = bundle
        if self.c.current_path == path:
            self.show(path)
        return bundle

    def forget(self, path: Path) -> None:
        self.bundles.pop(Path(path).expanduser().resolve(), None)
        if self.c.current_path == path:
            self.clear()

    def clear(self) -> None:
        for layer in self.layers:
            if layer in self.viewer.layers:
                self.viewer.layers.remove(layer)
        for mrc in self.files:
            mrc.close()
        self.layers, self.files = [], []

    def show(self, path: Path | None) -> None:
        """Replace the drawn bundle with `path`'s (none: nothing drawn)."""
        self.clear()
        bundle = self.bundles.get(Path(path).resolve()) if path is not None else None
        self._volume_controls(bundle)
        if bundle is None:
            return
        active = self.viewer.layers.selection.active
        problems: list[str] = []
        bundle_px = float(bundle.get("voxel_size") or 0) or self.c.header_voxel_size_angstrom
        for seg in bundle.get("segmentations") or []:
            try:
                self._add_segmentation(seg, bundle_px)
            except (OSError, ValueError) as e:
                logger.exception("segmentation %s not shown", seg.get("path"))
                problems.append(f"{seg.get('name')}: {e}")
        for picks in bundle.get("picks") or []:
            try:
                self._add_picks(picks, bundle_px)
            except (TypeError, ValueError) as e:
                logger.exception("picks %s not shown", picks.get("name"))
                problems.append(f"{picks.get('name')}: {e}")
        self._stack_below_annotations()
        self.rescale()
        self._thicken_for_picks(bundle)
        if active is not None and active in self.viewer.layers:
            self.viewer.layers.selection.active = active
        if problems:
            show_warning("crboost layers not shown: " + "; ".join(problems))

    def _add_segmentation(self, seg: dict, bundle_px: float) -> None:
        mrc = mrcfile.mmap(str(seg["path"]), mode="r", permissive=True)
        data = mrc.data
        if data is None or data.ndim != 3:
            mrc.close()
            raise ValueError(f"not a 3-D volume ({None if data is None else data.shape})")
        self.files.append(mrc)
        level = float(seg.get("level", 0.5))
        top = float(seg.get("full_scale") or 1.0)
        ratio = float(seg.get("voxel_size") or bundle_px) / bundle_px
        layer = self.viewer.add_image(
            data,
            name=f"crboost · {seg.get('name', Path(seg['path']).stem)}",
            **self._placement(ratio),
            colormap=_mask_colormap(f"crboost-{seg.get('name')}", seg.get("color", "")),
            contrast_limits=(level, max(top, level + 1e-6)),
            blending="translucent",
            opacity=0.6,
            rendering="iso",
            iso_threshold=level,
            visible=bool(seg.get("visible", True)),
            metadata={
                "annotation_kind": KIND,
                "relative_voxel": ratio,
                "bin_with": "max",  # thin features (membranes) survive binning for 3-D (view3d.py)
                "max_voxels_3d": 16 * 2**20,  # a mask needs less than density: bin 4 for 1k x 1k x 512
                # 3-D draws every visible segmentation as one labels volume (view3d.py): five ray-traced volumes
                # were 6x slower than one.
                "merge_3d": "crboost",
                "level": level,
                "color": seg.get("color", ""),
            },
        )
        self.layers.append(layer)

    def _add_picks(self, picks: dict, bundle_px: float) -> None:
        coords = np.asarray(picks.get("coordinates_zyx") or [], dtype=float).reshape(-1, 3)
        box = float(picks.get("box_size") or DEFAULT_BOX_ANGSTROM)
        r, g, b = _rgb(picks.get("color", ""))
        layer = self.viewer.add_points(
            coords,
            name=f"crboost · {picks.get('name', 'picks')}",
            **self._placement(1.0),
            size=max(box / bundle_px, 2.0),
            symbol="disc",
            face_color="transparent",
            border_color=[r, g, b, 1.0],
            border_width=2,
            border_width_is_relative=False,
            projection_mode="rescale_spherical",
            opacity=0.9,
            metadata={"annotation_kind": KIND, "relative_voxel": 1.0, "source": picks.get("source", "")},
        )
        layer.editable = False
        self.layers.append(layer)

    def _placement(self, ratio: float) -> dict:
        """scale, translate and units on the tomogram's current calibration (the filter dock's pixel size), for a
        grid `ratio` times as coarse as the tomogram's. It shares the volume's corner: its voxel i covers the
        tomogram's [i·r, (i+1)·r). Given at creation, so napari's slider range never sees a layer in voxels."""
        px = float(self.c.voxel_size_angstrom) if self.c.voxel_size_angstrom > 0 else 1.0
        return {
            "scale": (px * ratio,) * 3,
            "translate": ((ratio - 1.0) / 2.0 * px,) * 3,
            "units": ("angstrom",) * 3,
        }

    def rescale(self) -> None:
        """Follow a change of the tomogram's pixel size."""
        for layer in self.layers:
            placement = self._placement(float(layer.metadata.get("relative_voxel", 1.0)))
            layer.scale = placement["scale"]
            layer.translate = placement["translate"]

    def _thicken_for_picks(self, bundle: dict) -> None:
        """Z slice thickness (world Å) at least the largest box, so every pick the slice cuts is drawn."""
        boxes = [float(p.get("box_size") or DEFAULT_BOX_ANGSTROM) for p in bundle.get("picks") or []]
        if not boxes:
            return
        px = float(self.c.voxel_size_angstrom) if self.c.voxel_size_angstrom > 0 else 1.0
        thickness = list(self.viewer.dims.thickness)
        z_axis = len(thickness) - 3
        if z_axis < 0:
            return
        thickness[z_axis] = max(thickness[z_axis], max(boxes) + px)
        self.viewer.dims.thickness = tuple(thickness)

    def _stack_below_annotations(self) -> None:
        """Just above the tomogram, so your own particle and measurement layers stay on top."""
        anchor = self.c.filtered_layer or self.c.source_layer
        if anchor is None or anchor not in self.viewer.layers:
            return
        for layer in reversed(self.layers):
            src = self.viewer.layers.index(layer)
            dest = self.viewer.layers.index(anchor) + 1
            if src != dest:
                self.viewer.layers.move(src, dest)

    _MODES = {"smooth surfaces, one per feature": "surfaces", "merged, fastest (blocky)": "merged"}

    def _volume_controls(self, bundle: dict | None) -> None:
        """The crboost dock: the tomogram's volumes (set to the one on screen; the row hidden with fewer than two)
        and the 3-D drawing of its segmentations. Hidden without a bundle."""
        volumes = (bundle or {}).get("volumes") or []
        if bundle is None:
            self._volumes = {}
            if self.volume_dock is not None:
                self.volume_dock.setVisible(False)
            return
        if self.volume_box is None:
            from magicgui.widgets import ComboBox, Container

            # A callable: napari re-reads a dock's choices (reset_choices) whenever layers change, and a fixed list
            # would fall back to the one the box was made with.
            self.volume_box = ComboBox(label="Show", choices=lambda _widget: list(self._volumes))
            self.volume_box.changed.connect(self._on_volume)
            view3d = getattr(self.c, "view3d", None)
            current_mode = getattr(view3d, "seg_mode", "surfaces")
            self.mode_box = ComboBox(
                label="3-D segmentations",
                choices=lambda _widget: list(self._MODES),
                value=next(k for k, v in self._MODES.items() if v == current_mode),
            )
            self.mode_box.changed.connect(self._on_mode)
            container = Container(widgets=[self.volume_box, self.mode_box], labels=True)
            self.volume_dock = self.viewer.window.add_dock_widget(container, area="right", name="crboost")
        self._volumes = {v["name"]: v for v in volumes} if len(volumes) >= 2 else {}
        self.volume_box.visible = bool(self._volumes)
        self.mode_box.visible = bool(bundle.get("segmentations"))
        self.volume_dock.setVisible(bool(self._volumes) or bool(bundle.get("segmentations")))
        if not self._volumes:
            return
        opened = Path(bundle["tomogram"]).resolve()
        current = next((v["name"] for v in volumes if Path(v["path"]).resolve() == opened), volumes[0]["name"])
        self._updating = True
        try:
            self.volume_box.reset_choices()
            self.volume_box.value = current
        finally:
            self._updating = False
        self.volume_dock.setVisible(True)

    def _on_mode(self, label: str) -> None:
        view3d = getattr(self.c, "view3d", None)
        if view3d is None or label not in self._MODES:
            return
        title = self.viewer.title
        self.viewer.title = "Building 3-D segmentations…"
        from qtpy.QtWidgets import QApplication

        QApplication.processEvents()
        try:
            view3d.set_seg_mode(self._MODES[label])
        finally:
            self.viewer.title = title

    def _on_volume(self, name: str) -> None:
        if self._updating or name not in self._volumes:
            return
        from qtpy.QtWidgets import QApplication

        volume = self._volumes[name]
        title = self.viewer.title
        self.viewer.title = f"Opening {name}…"
        QApplication.processEvents()
        try:
            self.c.swap_volume(Path(volume["path"]), label=name.split(" (")[0])
        except (OSError, ValueError, RuntimeError) as e:
            logger.exception("volume %s not shown", volume["path"])
            show_warning(f"{name} is not shown: {e}")
        finally:
            self.viewer.title = title

    def summary(self) -> dict[str, Any]:
        """What is drawn, for a caller's report."""
        return {
            "picks": [layer.name for layer in self.layers if isinstance(layer, napari.layers.Points)],
            "segmentations": [layer.name for layer in self.layers if isinstance(layer, napari.layers.Image)],
            "volumes": list(self._volumes),
        }
