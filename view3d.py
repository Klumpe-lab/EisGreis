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
picks stand out against (a plain maximum projection is white, a minimum one saturates through 512 slices). ``metadata["rendering_3d"]`` and
``metadata["colormap_3d"]`` on a layer choose otherwise; a layer's own rendering stays for the rest (a
segmentation's isosurface).

The budget: 64 M voxels (256 MB as float32), what llvmpipe takes; ``NAPARI_3D_MAX_VOXELS`` sets another.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import napari
import numpy as np
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication

if TYPE_CHECKING:
    from napari_mrc_lowpass_fast import LowPassController

logger = logging.getLogger(__name__)

DEFAULT_MAX_VOXELS = 64 * 2**20
KIND = "view3d"


def bin_factor(shape: tuple[int, ...], max_voxels: int) -> int:
    """The smallest power of two that brings `shape` within `max_voxels`."""
    f = 1
    while int(np.prod([max(n // f, 1) for n in shape])) > max_voxels and min(shape) // (2 * f) >= 2:
        f *= 2
    return f


def bin_volume(data, f: int, how: str = "mean") -> np.ndarray:
    """`data` binned f x f x f (mean, or max), slab by slab so a memory-mapped volume is read once and never held
    whole as float."""
    nz, ny, nx = (n // f for n in data.shape)
    out = np.empty((nz, ny, nx), dtype=np.float32)
    for z in range(nz):
        slab = np.asarray(data[z * f:(z + 1) * f, :ny * f, :nx * f], dtype=np.float32)
        blocks = slab.reshape(f, ny, f, nx, f)
        out[z] = blocks.max(axis=(0, 2, 4)) if how == "max" else blocks.mean(axis=(0, 2, 4))
    return out


class Volume3D:
    """Binned stand-ins for the big image layers while the canvas is in 3-D."""

    def __init__(self, controller: "LowPassController", max_voxels: int | None = None) -> None:
        self.c = controller
        self.max_voxels = int(max_voxels or os.environ.get("NAPARI_3D_MAX_VOXELS") or DEFAULT_MAX_VOXELS)
        self.proxies: dict = {}  # full layer -> its binned stand-in
        self.cache: dict = {}  # id(full layer) -> (id(its data), binned array, factor)
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
                if proxy is not None and proxy in self.viewer.layers:
                    self.viewer.layers.remove(proxy)
                if full in self.viewer.layers:
                    full.visible = True
            self.proxies = {}
        finally:
            self._busy = False

    def refresh(self) -> None:
        """Stand-ins for every visible image layer over the budget, made afresh."""
        if not self.active:
            return
        self.restore()
        targets = [layer for layer in self.viewer.layers if layer.visible and self._too_big(layer)]
        live = {id(layer) for layer in self.viewer.layers}
        self.cache = {k: v for k, v in self.cache.items() if k in live}
        if not targets:
            return
        active = self.viewer.layers.selection.active
        self._busy = True
        # Hidden before anything draws: a full volume drawn in 3-D asks for one texture the size of it.
        for full in targets:
            full.visible = False
            self.proxies[full] = None
        title = self.viewer.title
        self.viewer.title = "Binning for 3-D…"
        QApplication.processEvents()
        try:
            with ThreadPoolExecutor(max_workers=max(1, min(len(targets), os.cpu_count() or 1))) as pool:
                results = list(pool.map(self._binned, targets))
            for full, (binned, f) in zip(targets, results):
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
