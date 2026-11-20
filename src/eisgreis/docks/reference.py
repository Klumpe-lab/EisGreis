"""The Reference dock: *Show* switches the volume on screen between the tomogram's pictures (``reference.py``), and
*3-D segmentations* chooses how 3-D draws its segmentations. Each row shows only when it has something to offer; the
dock hides when neither does."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from magicgui.widgets import ComboBox, Container
from napari.utils.notifications import show_warning

from eisgreis.docks.view3d import seg_mode_box

if TYPE_CHECKING:
    from eisgreis.app import Controller

logger = logging.getLogger(__name__)


class ReferenceDock:
    def __init__(self, controller: "Controller") -> None:
        self.c = controller
        self._volumes: dict[str, dict] = {}
        self._updating = False
        # A callable: napari re-reads a dock's choices (reset_choices) whenever layers change, and a fixed list
        # would fall back to the one the box was made with.
        self.volume_box = ComboBox(label="Show", choices=lambda _widget: list(self._volumes))
        self.volume_box.changed.connect(self._on_volume)
        self.mode_box = seg_mode_box(controller.view3d)
        self.container = Container(widgets=[self.volume_box, self.mode_box], labels=True)
        self.dock = controller.viewer.window.add_dock_widget(self.container, area="right", name="Reference")

    def update(self, volumes: list[dict], path: Optional[Path], has_segmentations: bool) -> None:
        """The volumes of the tomogram at `path` (set to the one on screen; the row hidden with none), and the 3-D
        row while it has segmentations."""
        self._volumes = {v["name"]: v for v in volumes}
        self.volume_box.visible = bool(self._volumes)
        self.mode_box.visible = has_segmentations
        self.dock.setVisible(bool(self._volumes) or has_segmentations)
        if not self._volumes or path is None:
            return
        current = next((v["name"] for v in volumes if Path(v["path"]).resolve() == path), volumes[0]["name"])
        self._updating = True
        try:
            self.volume_box.reset_choices()
            self.volume_box.value = current
        finally:
            self._updating = False

    def _on_volume(self, name: str) -> None:
        if self._updating or name not in self._volumes:
            return
        from qtpy.QtWidgets import QApplication

        volume = self._volumes[name]
        viewer = self.c.viewer
        title = viewer.title
        viewer.title = f"Opening {name}…"
        QApplication.processEvents()
        try:
            self.c.swap_volume(Path(volume["path"]), label=name.split(" (")[0])
        except (OSError, ValueError, RuntimeError) as e:
            logger.exception("volume %s not shown", volume["path"])
            show_warning(f"{name} is not shown: {e}")
        finally:
            viewer.title = title
