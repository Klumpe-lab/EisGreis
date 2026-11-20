"""The Export PNG dock: the slice on screen as a PNG with a 100 nm scale bar (see ``scalebar.py``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from magicgui.widgets import Container, PushButton

from eisgreis.scalebar import save_slice_png

if TYPE_CHECKING:
    from eisgreis.app import Controller


class ExportDock:
    def __init__(self, controller: "Controller") -> None:
        self.c = controller
        self.save_png = PushButton(label="Save current view PNG")
        self.save_png.changed.connect(self._on_save_png)
        self.container = Container(widgets=[self.save_png], labels=True)
        self.container.native.setWindowTitle("Export PNG")

    def _png_source_layer(self):
        """The image the user is looking at: filtered on top, else the original."""
        for layer in (self.c.filtered_layer, self.c.source_layer):
            if layer is not None and layer.visible:
                return layer
        return self.c.filtered_layer or self.c.source_layer

    def _on_save_png(self, _value: object) -> None:
        current = self.c.session.current
        layer = self._png_source_layer()
        if current is None or layer is None:
            return
        try:
            _path, detail = save_slice_png(
                self.c.viewer, layer, self.c.session.outputs.png_dir, current.stem, float(self.c.voxel_size_angstrom)
            )
        except Exception as exc:  # reported in the status line, and the viewer carries on
            self.c.set_status("PNG not saved", str(exc))
            return
        self.c.set_status("Saved PNG", detail)
