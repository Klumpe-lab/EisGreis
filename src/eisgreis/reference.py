"""Reference layers: picks, segmentations and other volumes of a tomogram, to look at beside your own particle layers.

They may come from anywhere: a STAR of picks from RELION or Warp, a segmentation from any tool, a pipeline's results.
They are never saved, being someone else's results rather than annotations of yours, and every tomogram keeps its
own: they are drawn while that tomogram is on screen.

- **Picks** are voxel coordinates (Z, Y, X) of the tomogram, so they follow the pixel size the filter dock sets, like
  your own layers. In 2-D a pick draws as the cross-section of a sphere of its box through the slice on screen
  (napari's "rescale_spherical" projection). napari draws points only inside the slice's thickness, so picks widen
  the Z thickness to the largest box; images still show their single plane.
- **Segmentations** keep their own grid: their voxel size relative to the tomogram's says how much coarser it is, and
  both grids share the volume's corner. A segmentation draws where its value passes its level, in its colour. In 3-D
  it is drawn by ``view3d.py``, which knows it as a segmentation by its metadata (``merge_3d``), not by its source.
- **Volumes** are other pictures of the tomogram on its own grid (a denoised one, say). With two or more, the
  Reference dock's *Show* puts one in the tomogram's place (``swap_volume``), low-passed at the current cutoff,
  while particles, comments and STAR names stay with the tomogram that was opened.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import mrcfile
import napari
import numpy as np
from napari.utils.colormaps import Colormap
from napari.utils.notifications import show_warning

if TYPE_CHECKING:
    from eisgreis.app import Controller

logger = logging.getLogger(__name__)

KIND = "reference"  # layer.metadata["annotation_kind"] of every layer made here
MERGE_GROUP = "reference"  # the 3-D merge group of every segmentation (view3d.py)
DEFAULT_BOX_ANGSTROM = 200.0


def _rgb(color: str) -> tuple[float, float, float]:
    """'#rrggbb' as 0–1 floats; anything else is grey."""
    c = str(color or "").lstrip("#")
    if len(c) != 6:
        return (0.7, 0.7, 0.7)
    return tuple(int(c[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def _mask_colormap(name: str, color: str) -> Colormap:
    """Transparent at the bottom of the contrast range, the feature's colour just above it: with the range starting
    at the segmentation's level, a probability volume draws as a solid mask of where it passes that level."""
    r, g, b = _rgb(color)
    return Colormap(colors=[[r, g, b, 0.0], [r, g, b, 0.75], [r, g, b, 0.75]], controls=[0.0, 0.02, 1.0], name=name)


@dataclass
class ReferenceSet:
    """One tomogram's reference layers, as given."""

    picks: list[dict] = field(default_factory=list)
    segmentations: list[dict] = field(default_factory=list)
    volumes: list[dict] = field(default_factory=list)


class ReferenceLayers:
    """Reference layers by tomogram, and the drawn layers of the tomogram on screen."""

    def __init__(self, controller: "Controller") -> None:
        self.c = controller
        self.sets: dict[Path, ReferenceSet] = {}
        self.layers: list = []
        self.files: list = []
        self.not_shown: list[str] = []  # what the last drawing could not draw, and why
        self.dock = None  # the Reference dock (docks/reference.py), made when first needed

    @property
    def viewer(self) -> napari.Viewer:
        return self.c.viewer

    def _target(self, tomogram: Optional[Path]) -> Path:
        path = Path(tomogram).expanduser().resolve() if tomogram is not None else self.c.session.current
        if path is None:
            raise RuntimeError("No tomogram is open, and none was named")
        return path

    def _on_screen(self, path: Path) -> bool:
        return self.c.session.current == path and self.c.source_layer is not None

    def _add(self, path: Path, kind: str, spec: dict) -> None:
        """Keep `spec` for `path`'s tomogram; draw it at once when that tomogram is on screen. A spec that cannot
        be drawn is not kept, and the error is raised."""
        ref_set = self.sets.setdefault(path, ReferenceSet())
        getattr(ref_set, kind).append(spec)
        if not self._on_screen(path):
            return
        active = self.viewer.layers.selection.active
        try:
            if kind == "segmentations":
                self._draw_segmentation(spec)
            elif kind == "picks":
                self._draw_picks(spec)
        except Exception:
            getattr(ref_set, kind).remove(spec)
            raise
        finally:
            self._settle(ref_set, active)

    # ── what a driver calls (through eisgreis.api) ─────────────────────────────────────────────────────────────

    def show_picks(
        self,
        name: str,
        coords_zyx,
        box_angstrom: Optional[float] = None,
        color: str = "",
        *,
        source: str = "",
        pixel_size: Optional[float] = None,
        tomogram: Optional[Path] = None,
    ) -> None:
        """Picks as voxel coordinates (Z, Y, X) of the tomogram whose voxels are `pixel_size` Å (default: its
        header's), drawn as discs of `box_angstrom` in `color` ('#rrggbb')."""
        coords = np.asarray(coords_zyx if coords_zyx is not None else [], dtype=float).reshape(-1, 3)
        spec = {
            "name": name,
            "coords": coords,
            "box": float(box_angstrom or DEFAULT_BOX_ANGSTROM),
            "color": color,
            "source": source,
            "pixel_size": pixel_size,
        }
        self._add(self._target(tomogram), "picks", spec)

    def show_segmentation(
        self,
        path: Path,
        level: float = 0.5,
        color: str = "",
        voxel_size: Optional[float] = None,
        *,
        name: Optional[str] = None,
        full_scale: float = 1.0,
        visible: bool = True,
        pixel_size: Optional[float] = None,
        tomogram: Optional[Path] = None,
    ) -> None:
        """A segmentation volume at `path`, drawn where it passes `level`, in `color`. `voxel_size` is its own
        voxel size (Å; default the tomogram's), `pixel_size` the tomogram grid's it is placed on (default: the
        tomogram header's); `full_scale` the value at the top of its range."""
        spec = {
            "name": name or Path(path).stem,
            "path": str(path),
            "level": float(level),
            "full_scale": float(full_scale or 1.0),
            "color": color,
            "voxel_size": voxel_size,
            "visible": bool(visible),
            "pixel_size": pixel_size,
        }
        self._add(self._target(tomogram), "segmentations", spec)

    def add_volume(self, name: str, path: Path, kind: str = "", *, tomogram: Optional[Path] = None) -> None:
        """Another picture of the tomogram on its grid, offered by the Reference dock's *Show*."""
        target = self._target(tomogram)
        self.sets.setdefault(target, ReferenceSet()).volumes.append({"name": name, "path": str(path), "kind": kind})
        if self._on_screen(target):
            self._update_dock(target)

    def clear(self, tomogram: Optional[Path] = None) -> None:
        """Drop a tomogram's reference layers (the one on screen by default)."""
        target = self._target(tomogram)
        self.sets.pop(target, None)
        if self._on_screen(target):
            self._remove_drawn()
            self._update_dock(target)

    def forget(self, path: Path) -> None:
        path = Path(path).expanduser().resolve()
        self.sets.pop(path, None)
        if self.c.session.current == path:
            self._remove_drawn()

    def summary(self, tomogram: Optional[Path] = None) -> dict[str, Any]:
        """What is drawn for the tomogram on screen (or what is kept for `tomogram`), for a caller's report."""
        target = self._target(tomogram)
        ref_set = self.sets.get(target, ReferenceSet())
        return {
            "picks": [f"{p['name']}" for p in ref_set.picks],
            "segmentations": [f"{s['name']}" for s in ref_set.segmentations],
            "volumes": [v["name"] for v in self._volume_choices(target)],
            "not_shown": list(self.not_shown) if self._on_screen(target) else [],
        }

    # ── drawing ────────────────────────────────────────────────────────────────────────────────────────────────

    def show(self, path: Optional[Path]) -> None:
        """Replace the drawn layers with `path`'s (none: nothing drawn). Problems are listed in `not_shown` and
        shown as a warning; the rest is drawn."""
        self._remove_drawn()
        path = Path(path).resolve() if path is not None else None
        ref_set = self.sets.get(path) if path is not None else None
        self._update_dock(path)
        if ref_set is None:
            return
        active = self.viewer.layers.selection.active
        problems: list[str] = []
        for seg in ref_set.segmentations:
            try:
                self._draw_segmentation(seg)
            except (OSError, ValueError) as e:
                logger.exception("segmentation %s not shown", seg.get("path"))
                problems.append(f"{seg.get('name')}: {e}")
        for picks in ref_set.picks:
            try:
                self._draw_picks(picks)
            except (TypeError, ValueError) as e:
                logger.exception("picks %s not shown", picks.get("name"))
                problems.append(f"{picks.get('name')}: {e}")
        self.not_shown = problems
        self._settle(ref_set, active)
        if problems:
            show_warning("Reference layers not shown: " + "; ".join(problems))

    def _remove_drawn(self) -> None:
        for layer in self.layers:
            if layer in self.viewer.layers:
                self.viewer.layers.remove(layer)
        for mrc in self.files:
            mrc.close()
        self.layers, self.files, self.not_shown = [], [], []

    def _grid_px(self, spec: dict) -> float:
        return float(spec.get("pixel_size") or 0) or float(self.c.header_voxel_size_angstrom)

    def _draw_segmentation(self, seg: dict) -> None:
        mrc = mrcfile.mmap(str(seg["path"]), mode="r", permissive=True)
        data = mrc.data
        if data is None or data.ndim != 3:
            mrc.close()
            raise ValueError(f"not a 3-D volume ({None if data is None else data.shape})")
        self.files.append(mrc)
        grid_px = self._grid_px(seg)
        level = float(seg["level"])
        top = float(seg.get("full_scale") or 1.0)
        ratio = float(seg.get("voxel_size") or grid_px) / grid_px
        layer = self.viewer.add_image(
            data,
            name=seg["name"],
            **self._placement(ratio),
            colormap=_mask_colormap(f"reference-{seg['name']}", seg.get("color", "")),
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
                "merge_3d": MERGE_GROUP,
                "level": level,
                "color": seg.get("color", ""),
            },
        )
        self.layers.append(layer)

    def _draw_picks(self, picks: dict) -> None:
        coords = np.asarray(picks["coords"], dtype=float).reshape(-1, 3)
        r, g, b = _rgb(picks.get("color", ""))
        layer = self.viewer.add_points(
            coords,
            name=picks["name"],
            **self._placement(1.0),
            size=max(float(picks["box"]) / self._grid_px(picks), 2.0),
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

    def _settle(self, ref_set: ReferenceSet, active) -> None:
        """After drawing: just above the tomogram, on its calibration, the slice thick enough for every pick; the
        layer that was selected stays selected."""
        self._stack_below_annotations()
        self.rescale()
        self._thicken_for_picks(ref_set)
        if active is not None and active in self.viewer.layers:
            self.viewer.layers.selection.active = active

    def _placement(self, ratio: float) -> dict:
        """scale, translate and units on the tomogram's current calibration (the filter dock's pixel size), for a
        grid `ratio` times as coarse as the tomogram's. It shares the volume's corner: its voxel i covers the
        tomogram's [i·r, (i+1)·r). Given at creation, so napari's slider range never sees a layer in voxels."""
        px = float(self.c.voxel_size_angstrom) if self.c.voxel_size_angstrom > 0 else 1.0
        return {"scale": (px * ratio,) * 3, "translate": ((ratio - 1.0) / 2.0 * px,) * 3, "units": ("angstrom",) * 3}

    def rescale(self) -> None:
        """Follow a change of the tomogram's pixel size."""
        for layer in self.layers:
            placement = self._placement(float(layer.metadata.get("relative_voxel", 1.0)))
            layer.scale = placement["scale"]
            layer.translate = placement["translate"]

    def _thicken_for_picks(self, ref_set: ReferenceSet) -> None:
        """Z slice thickness (world Å) at least the largest box, so every pick the slice cuts is drawn."""
        boxes = [float(p["box"]) for p in ref_set.picks]
        if boxes:
            self.c.ensure_z_thickness(max(boxes))

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

    # ── the Reference dock ─────────────────────────────────────────────────────────────────────────────────────

    def _volume_choices(self, path: Optional[Path]) -> list[dict]:
        """The volumes *Show* offers for `path`: its own, then the others; none with fewer than two."""
        ref_set = self.sets.get(path) if path is not None else None
        if ref_set is None or not ref_set.volumes:
            return []
        volumes = list(ref_set.volumes)
        if not any(Path(v["path"]).resolve() == path for v in volumes):
            volumes.insert(0, {"name": f"opened ({path.name})", "path": str(path), "kind": "opened"})
        return volumes if len(volumes) >= 2 else []

    def _update_dock(self, path: Optional[Path]) -> None:
        ref_set = self.sets.get(path) if path is not None else None
        volumes = self._volume_choices(path)
        has_segmentations = bool(ref_set and ref_set.segmentations)
        if self.dock is None:
            if not volumes and not has_segmentations:
                return
            from eisgreis.docks.reference import ReferenceDock

            self.dock = ReferenceDock(self.c)
        self.dock.update(volumes, path, has_segmentations)
