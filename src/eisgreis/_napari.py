"""Every use of napari's private modules, in one place: what breaks between napari versions breaks here.

- the Qt main window, for its close and to maximize it;
- the GL canvas, for the OpenGL renderer it draws with;
- ``napari_builtins._measure_shapes``, the length text of the measurement layer;
- the scale bar's vispy overlay, so the bar reads in nanometres.

Checked against napari 0.10. The DPI fallback of napari's text visual that older napari needed is napari's own now
(``Text.dpi_ratio``).
"""

from __future__ import annotations

import logging
from typing import Callable

import napari

logger = logging.getLogger(__name__)

# The scale bar's length and label per viewer, keyed by id() of its overlay model (napari's models are not
# hashable): desired length (nm) -> (length to draw (nm), label).
_SCALE_BAR_HOOKS: dict[int, Callable[[float], tuple[float, str]]] = {}


def qt_window(viewer: napari.Viewer):
    """The viewer's QMainWindow, or None when napari moved it."""
    return getattr(viewer.window, "_qt_window", None)


def on_window_closed(viewer: napari.Viewer, callback: Callable[[], None]) -> None:
    """Call `callback` once the viewer's window is gone (closed by the user or by code)."""
    window = qt_window(viewer)
    if window is None:
        logger.warning("napari's main window is not reachable: nothing is saved when the window closes")
        return
    window.destroyed.connect(lambda *_args: callback())


def gl_renderer(viewer: napari.Viewer) -> str:
    """The OpenGL renderer the canvas draws with ("NVIDIA B200/PCIe/SSE2", "llvmpipe (LLVM 20.1.2, 256 bits)"), or
    "unknown" when napari's canvas cannot be reached."""
    from vispy.gloo import gl

    try:
        viewer.window._qt_viewer.canvas._scene_canvas.native.makeCurrent()
    except AttributeError:
        logger.warning("could not reach napari's GL canvas; the renderer is unknown")
        return "unknown"
    return str(gl.glGetParameter(gl.GL_RENDERER))


def toggle_shape_measures(layer) -> bool:
    """Turn on napari's length measurement on a shapes layer; False when napari no longer offers it."""
    try:
        from napari_builtins._measure_shapes import toggle_shape_measures as toggle
    except ImportError:
        logger.warning("napari_builtins no longer measures shapes: the measurement layer shows no lengths")
        return False
    toggle(layer)
    return True


class _Label:
    """What napari formats as the scale bar's text (``f"{quantity:g~#P}"``): our text, whatever the format."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __format__(self, _spec: str) -> str:
        return self.text


def nm_scale_bar(viewer: napari.Viewer, length_and_label: Callable[[float], tuple[float, str]]) -> None:
    """Draw `viewer`'s scale bar at the lengths `length_and_label` picks, with its label.

    napari sizes the bar from the layers' unit and prints it compacted: with layers in Å a bar reads "1.2 kÅ".
    `length_and_label(desired_nm)` gets the length napari would like the bar to be (nm) and returns the length to
    draw (nm) and its text. Only this viewer's bar changes; other viewers keep napari's own."""
    from napari._vispy.overlays.scale_bar import VispyScaleBarOverlay
    from pint.errors import DimensionalityError

    if not getattr(VispyScaleBarOverlay, "_eisgreis_nm", False):
        original = VispyScaleBarOverlay._calculate_best_length

        def _calculate_best_length(self, desired_length: float):
            hook = _SCALE_BAR_HOOKS.get(id(self.overlay))
            if hook is None:
                return original(self, desired_length)
            try:
                nm_per_unit = (1 * self._unit).to("nanometer").magnitude
            except DimensionalityError:  # the layers are in no length unit: napari's own bar
                return original(self, desired_length)
            length_nm, label = hook(desired_length * nm_per_unit)
            return length_nm / nm_per_unit, _Label(label)

        VispyScaleBarOverlay._calculate_best_length = _calculate_best_length
        VispyScaleBarOverlay._eisgreis_nm = True
    _SCALE_BAR_HOOKS[id(viewer.canvas.overlays.scale_bar)] = length_and_label


def refresh_scale_bar(viewer: napari.Viewer) -> None:
    """Redraw the scale bar's length and text now, after the layers' calibration changed."""
    try:
        canvas = viewer.window._qt_viewer.canvas
        visuals = canvas._viewer_overlay_to_visual.get(viewer.canvas.overlays.scale_bar, [])
    except AttributeError:
        logger.warning("napari's scale bar visual is not reachable; it updates at the next zoom")
        return
    for visual in visuals:
        visual._on_unit_change()
