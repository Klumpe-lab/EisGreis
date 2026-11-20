"""3-D that works on any renderer: binned copies of the big volumes while the canvas is in 3-D.

napari draws a volume in 3-D as one OpenGL texture. A 1024 x 1024 x 512 float32 tomogram is 2 GB of it: software
rendering (Mesa's llvmpipe, a remote desktop without a GPU) cannot allocate that, and a GPU takes long to upload
it. So when the 2D/3D button (bottom left, or Ctrl+Y) turns the canvas to 3-D, every visible image layer over
``max_voxels`` is drawn as a 2 x 2 x 2 (or coarser) binned copy and the full one is hidden; back in 2-D the copies
go and the full layers come back. A low-pass change, another tomogram or another layer while in 3-D redraws the
copies; a copy already made is reused while its layer's data stays the same, so turning 3-D off and on again is
instant. The copies are binned in parallel threads. Density is binned by mean, a layer whose
``metadata["bin_with"]`` is ``"max"`` by maximum (thin segmentations, a membrane, survive that); a layer may lower
its own budget with ``metadata["max_voxels_3d"]``.

Density draws as an attenuated maximum projection in the inverted colour map, a grey slab the segmentations and
picks stand out against (a plain maximum projection is white, a minimum one saturates through 512 slices).
``metadata["rendering_3d"]`` and ``metadata["colormap_3d"]`` on a layer choose otherwise; a layer's own rendering
stays for the rest (a segmentation's isosurface).

Layers that name the same ``metadata["merge_3d"]`` group (segmentations: probability volumes, shown where the
value passes ``metadata["level"]``, in ``metadata["color"]``) draw in 3-D one of two ways (``seg_mode``):

- ``"surfaces"`` (the default): a smooth mesh per layer, each its own layer with its own eye. Marching cubes on the
  binned volume (lightly smoothed, specks under 500 voxels dropped) at the finest step that keeps the mesh within
  ``triangle_budget``, then Taubin smoothing. Every layer of the group gets one, hidden ones hidden, so any can be
  turned on in 3-D. Meshes are cheap to draw where volumes are not: napari traces rays through every volume.
- ``"merged"``: ONE labels volume, a label per visible layer; layers covering more go in first, so a membrane stays
  on top inside the cytoplasm. The fastest, but blocky (whole bin-4 voxels) and all or nothing.

Five segmentation volumes took 380 ms a frame in software on 8 CPUs, one labels volume 62 ms.

The budget: 64 M voxels (256 MB as float32), what llvmpipe takes; ``NAPARI_3D_MAX_VOXELS`` sets another.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import napari
import numpy as np
from napari.utils.colormaps import Colormap, DirectLabelColormap
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication

if TYPE_CHECKING:
    from eisgreis.app import Controller

logger = logging.getLogger(__name__)

DEFAULT_MAX_VOXELS = 64 * 2**20
KIND = "view3d"


def bin_factor(shape: tuple[int, ...], max_voxels: int) -> int:
    """The smallest power of two that brings `shape` within `max_voxels`."""
    f = 1
    while int(np.prod([max(n // f, 1) for n in shape])) > max_voxels and min(shape) // (2 * f) >= 2:
        f *= 2
    return f


def _rgba(color: str | None) -> list[float]:
    """'#rrggbb' as RGBA floats; anything else light grey."""
    c = str(color or "").lstrip("#")
    if len(c) != 6:
        return [0.8, 0.8, 0.8, 1.0]
    return [int(c[i : i + 2], 16) / 255.0 for i in (0, 2, 4)] + [1.0]


def taubin(verts: np.ndarray, faces: np.ndarray, iters: int = 8, lam: float = 0.5, mu: float = -0.53) -> np.ndarray:
    """Taubin smoothing (a shrink step, then an inflate step, per iteration): marching cubes' staircase goes,
    the volume stays."""
    from scipy import sparse

    n = len(verts)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    rows, cols = np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]]
    a = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    a.data[:] = 1.0
    deg = np.asarray(a.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    v = verts.astype(np.float64)
    for _ in range(iters):
        for factor in (lam, mu):
            v = v + factor * (a @ v / deg[:, None] - v)
    return v.astype(np.float32)


def segmentation_mesh(volume: np.ndarray, level: float, budget: int, min_voxels: int = 500):
    """(vertices in `volume`'s voxels, faces, step) of the surface where `volume` passes `level`, or None when it
    never does. Specks under `min_voxels` go; the marching-cubes step coarsens (up to 4) until the mesh fits
    `budget` triangles."""
    from scipy import ndimage
    from skimage.measure import marching_cubes

    v = ndimage.gaussian_filter(volume, 0.7)
    mask = v > level
    if not mask.any():
        return None
    labels, _n = ndimage.label(mask)
    sizes = np.bincount(labels.ravel())
    specks = np.flatnonzero(sizes < min_voxels)
    specks = specks[specks > 0]
    if len(specks):
        v = np.where(np.isin(labels, specks), np.minimum(v, level - 1e-3), v)
    if v.max() <= level:
        return None
    for step in (1, 2, 3, 4):
        verts, faces, _normals, _values = marching_cubes(v, level=level, step_size=step)
        if len(faces) <= budget:
            break
    return taubin(verts, faces), faces, step


def bin_volume(data, f: int, how: str = "mean") -> np.ndarray:
    """`data` binned f x f x f (mean, or max), slab by slab so a memory-mapped volume is read once and never held
    whole as float."""
    nz, ny, nx = (n // f for n in data.shape)
    out = np.empty((nz, ny, nx), dtype=np.float32)
    for z in range(nz):
        slab = np.asarray(data[z * f : (z + 1) * f, : ny * f, : nx * f], dtype=np.float32)
        blocks = slab.reshape(f, ny, f, nx, f)
        out[z] = blocks.max(axis=(0, 2, 4)) if how == "max" else blocks.mean(axis=(0, 2, 4))
    return out


class Volume3D:
    """Binned stand-ins for the big image layers while the canvas is in 3-D."""

    def __init__(self, controller: "Controller", max_voxels: int | None = None) -> None:
        self.c = controller
        self.max_voxels = int(max_voxels or os.environ.get("NAPARI_3D_MAX_VOXELS") or DEFAULT_MAX_VOXELS)
        self.proxies: dict = {}  # full layer -> its binned stand-in
        self.cache: dict = {}  # id(full layer) -> (id(its data), binned array, factor)
        self.label_cache: dict = {}  # merge group's members' (id, id(data)) -> labels volume
        self.mesh_cache: dict = {}  # (id(layer), id(its data), budget) -> mesh or None
        self.seg_mode = os.environ.get("NAPARI_3D_SEGMENTATIONS", "surfaces")
        self.triangle_budget = int(os.environ.get("NAPARI_3D_TRIANGLES") or 40_000)
        self.was_visible: dict = {}  # full layer -> its visibility before 3-D hid it
        self._busy = False
        self._pending = QTimer()
        self._pending.setSingleShot(True)
        self._pending.setInterval(200)
        self._pending.timeout.connect(self.refresh)
        self.viewer.dims.events.ndisplay.connect(self._on_ndisplay)
        self.viewer.layers.events.inserted.connect(self._on_layers_changed)
        self.viewer.layers.events.removed.connect(self._on_layers_changed)

    @property
    def viewer(self) -> napari.Viewer:
        return self.c.viewer

    @property
    def active(self) -> bool:
        return self.viewer.dims.ndisplay == 3

    def set_seg_mode(self, mode: str) -> None:
        """ "surfaces" or "merged"; redrawn at once when in 3-D."""
        if mode not in ("surfaces", "merged"):
            raise ValueError(f"unknown segmentation mode {mode!r}")
        self.seg_mode = mode
        if self.active:
            self.refresh()

    def _on_ndisplay(self, _event=None) -> None:
        if self.active:
            self.refresh()
        else:
            self.restore()

    def _on_layers_changed(self, _event=None) -> None:
        if self.active and not self._busy:
            self._pending.start()

    def watch(self, layer) -> None:
        """Redraw `layer`'s stand-in when its data changes (the low-pass, the next tomogram)."""
        if layer is not None and not layer.metadata.get("_view3d_watched"):
            layer.events.data.connect(self._on_layers_changed)
            layer.metadata["_view3d_watched"] = True

    def _budget(self, layer) -> int:
        return min(self.max_voxels, int(layer.metadata.get("max_voxels_3d") or self.max_voxels))

    def _too_big(self, layer) -> bool:
        return (
            isinstance(layer, napari.layers.Image)
            and layer.metadata.get("annotation_kind") != KIND
            and layer.ndim == 3
            and not layer.multiscale
            and int(np.prod(layer.data.shape)) > self._budget(layer)
        )

    @staticmethod
    def _is_density(layer) -> bool:
        return layer.metadata.get("bin_with", "mean") == "mean" and layer.rendering != "iso"

    def _rendering(self, full) -> str:
        return full.metadata.get("rendering_3d") or ("attenuated_mip" if self._is_density(full) else full.rendering)

    def _colormap(self, full):
        if "colormap_3d" in full.metadata:
            return full.metadata["colormap_3d"]
        if self._is_density(full) and full.colormap.name == "gray":
            return "gray_r"
        return None

    def _binned(self, full) -> tuple[np.ndarray, int]:
        """`full`'s binned copy, from the cache while its data is the same array."""
        hit = self.cache.get(id(full))
        if hit is not None and hit[0] == id(full.data):
            return hit[1], hit[2]
        f = bin_factor(full.data.shape, self._budget(full))
        binned = bin_volume(full.data, f, full.metadata.get("bin_with", "mean"))
        self.cache[id(full)] = (id(full.data), binned, f)
        return binned, f

    def restore(self) -> None:
        """Back to the full layers (2-D)."""
        self._busy = True
        try:
            for full, proxy in self.proxies.items():
                if proxy is not None and proxy in self.viewer.layers:  # a merged proxy: removed once
                    self.viewer.layers.remove(proxy)
                if full in self.viewer.layers:
                    full.visible = self.was_visible.get(full, True)
            self.proxies = {}
            self.was_visible = {}
        finally:
            self._busy = False

    def _merged(self, members: list, binned: list) -> np.ndarray | None:
        """One labels volume of a merge group (label i+1 = members[i]), or None when the binned members do not share
        a grid (then each draws on its own)."""
        if len({b.shape for b in binned}) != 1 or len({tuple(m.scale) for m in members}) != 1:
            return None
        key = tuple((id(m), id(m.data)) for m in members)
        if key in self.label_cache:
            return self.label_cache[key]
        masks = [b > float(m.metadata.get("level", 0.5)) for m, b in zip(members, binned, strict=True)]
        labels = np.zeros(binned[0].shape, dtype=np.uint8)
        for i in sorted(range(len(members)), key=lambda i: -int(masks[i].sum())):
            labels[masks[i]] = i + 1
        self.label_cache = {key: labels}
        return labels

    def _add_labels(self, key: str, members: list, labels: np.ndarray, f: int):
        first = members[0]
        sz, sy, sx = (float(s) for s in first.scale[-3:])
        tz, ty, tx = (float(t) for t in first.translate[-3:])
        colors = {0: "transparent", None: "transparent"}
        for i, m in enumerate(members, start=1):
            colors[i] = m.metadata.get("color") or "white"
        proxy = self.viewer.add_labels(
            labels,
            name=f"{key} segmentations (3-D, bin {f})",
            scale=(sz * f, sy * f, sx * f),
            translate=(tz + (f - 1) / 2 * sz, ty + (f - 1) / 2 * sy, tx + (f - 1) / 2 * sx),
            colormap=DirectLabelColormap(color_dict=colors),
            opacity=0.8,
            units=first.units,
            metadata={"annotation_kind": KIND, "labels": {i: m.name for i, m in enumerate(members, start=1)}},
        )
        top = max(self.viewer.layers.index(m) for m in members)
        self.viewer.layers.move(self.viewer.layers.index(proxy), top + 1)
        return proxy

    def _mesh(self, full):
        """`full`'s surface mesh (on its binned grid), cached while its data and the budget stay the same."""
        key = (id(full), id(full.data), self.triangle_budget)
        if key not in self.mesh_cache:
            binned, _f = self._binned(full)
            self.mesh_cache[key] = segmentation_mesh(
                binned, float(full.metadata.get("level", 0.5)), self.triangle_budget
            )
        return self.mesh_cache[key]

    def _add_surface(self, full, mesh, f: int):
        verts, faces, step = mesh
        sz, sy, sx = (float(s) for s in full.scale[-3:])
        tz, ty, tx = (float(t) for t in full.translate[-3:])
        color = _rgba(full.metadata.get("color"))
        proxy = self.viewer.add_surface(
            (verts, faces, np.ones(len(verts), dtype=np.float32)),
            name=f"{full.name} (3-D surface)",
            scale=(sz * f, sy * f, sx * f),
            translate=(tz + (f - 1) / 2 * sz, ty + (f - 1) / 2 * sy, tx + (f - 1) / 2 * sx),
            colormap=Colormap([color, color], name=f"view3d-{full.name}"),
            shading="smooth",
            opacity=0.9,
            visible=self.was_visible.get(full, True),
            units=full.units,
            metadata={"annotation_kind": KIND, "triangles": int(len(faces)), "step": step},
        )
        self.viewer.layers.move(self.viewer.layers.index(proxy), self.viewer.layers.index(full) + 1)
        return proxy

    def refresh(self) -> None:
        """Stand-ins for every visible image layer over the budget and for every merge group, made afresh."""
        if not self.active:
            return
        self.restore()
        groups: dict[str, list] = {}
        singles = []
        surfaces = self.seg_mode == "surfaces"
        for layer in self.viewer.layers:
            if not isinstance(layer, napari.layers.Image) or layer.ndim != 3:
                continue
            if layer.metadata.get("annotation_kind") == KIND:
                continue
            key = layer.metadata.get("merge_3d")
            # Surfaces: every layer of a group, so a hidden one can be turned on in 3-D; merged: the visible ones.
            if key and (layer.visible or surfaces):
                groups.setdefault(key, []).append(layer)
            elif layer.visible and not key and self._too_big(layer):
                singles.append(layer)
        targets = singles + [m for members in groups.values() for m in members]
        live = {id(layer) for layer in self.viewer.layers}
        self.cache = {k: v for k, v in self.cache.items() if k in live}
        self.mesh_cache = {k: v for k, v in self.mesh_cache.items() if k[0] in live}
        if not targets:
            return
        active = self.viewer.layers.selection.active
        self._busy = True
        # Hidden before anything draws: a full volume drawn in 3-D asks for one texture the size of it.
        for full in targets:
            self.was_visible[full] = full.visible
            full.visible = False
            self.proxies[full] = None
        title = self.viewer.title
        self.viewer.title = "Binning for 3-D…"
        QApplication.processEvents()
        try:
            with ThreadPoolExecutor(max_workers=max(1, min(len(targets), os.cpu_count() or 1))) as pool:
                results = dict(zip(map(id, targets), pool.map(self._binned, targets), strict=True))
                members_all = [m for members in groups.values() for m in members]
                meshes = (
                    dict(zip(map(id, members_all), pool.map(self._mesh, members_all), strict=True)) if surfaces else {}
                )
            for key, members in groups.items():
                if surfaces:
                    for m in members:
                        mesh = meshes.get(id(m))
                        if mesh is not None:
                            self.proxies[m] = self._add_surface(m, mesh, results[id(m)][1])
                    continue
                labels = self._merged(members, [results[id(m)][0] for m in members])
                if labels is None:
                    singles.extend(members)
                    continue
                proxy = self._add_labels(key, members, labels, results[id(members[0])][1])
                for m in members:
                    self.proxies[m] = proxy
            for full in singles:
                binned, f = results[id(full)]
                sz, sy, sx = (float(s) for s in full.scale[-3:])
                tz, ty, tx = (float(t) for t in full.translate[-3:])
                proxy = self.viewer.add_image(
                    binned,
                    name=f"{full.name} (3-D, bin {f})",
                    scale=(sz * f, sy * f, sx * f),
                    # Bin i covers full voxels [i·f, (i+1)·f): its centre sits (f-1)/2 voxels in.
                    translate=(tz + (f - 1) / 2 * sz, ty + (f - 1) / 2 * sy, tx + (f - 1) / 2 * sx),
                    colormap=full.colormap,
                    contrast_limits=full.contrast_limits,
                    blending=full.blending,
                    opacity=full.opacity,
                    rendering=self._rendering(full),
                    units=full.units,
                    metadata={"annotation_kind": KIND},
                )
                colormap = self._colormap(full)
                if colormap is not None:
                    proxy.colormap = colormap
                if full.rendering == "iso":
                    proxy.iso_threshold = full.iso_threshold
                self.viewer.layers.move(self.viewer.layers.index(proxy), self.viewer.layers.index(full) + 1)
                self.proxies[full] = proxy
                logger.info("3-D: %s %s binned %d -> %s", full.name, full.data.shape, f, binned.shape)
        finally:
            self._busy = False
            self.viewer.title = title
            if active is not None and active in self.viewer.layers:
                self.viewer.layers.selection.active = active
