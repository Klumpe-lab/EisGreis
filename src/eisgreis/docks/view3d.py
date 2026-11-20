"""The 3-D segmentations switch: how 3-D draws segmentations (``view3d.py``), smooth surfaces or one merged labels
volume. It sits in the Reference dock, shown while the tomogram on screen has segmentations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from magicgui.widgets import ComboBox

if TYPE_CHECKING:
    from eisgreis.view3d import Volume3D

MODES = {"smooth surfaces, one per feature": "surfaces", "merged, fastest (blocky)": "merged"}


def seg_mode_box(view3d: "Volume3D") -> ComboBox:
    """The switch, set to `view3d`'s mode; the window title says what it is doing while it redraws."""
    current = getattr(view3d, "seg_mode", "surfaces")
    # A callable: napari re-reads a dock's choices (reset_choices) whenever layers change, and a fixed list would
    # fall back to the one the box was made with.
    box = ComboBox(
        label="3-D segmentations",
        choices=lambda _widget: list(MODES),
        value=next(k for k, v in MODES.items() if v == current),
    )

    def _on_mode(label: str) -> None:
        if label not in MODES:
            return
        from qtpy.QtWidgets import QApplication

        viewer = view3d.viewer
        title = viewer.title
        viewer.title = "Building 3-D segmentations…"
        QApplication.processEvents()
        try:
            view3d.set_seg_mode(MODES[label])
        finally:
            viewer.title = title

    box.changed.connect(_on_mode)
    return box
