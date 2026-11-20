"""The MRC Filter dock: add a file, pick the tomogram on screen, its pixel size, the low-pass cutoff and filter."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from magicgui.widgets import CheckBox, ComboBox, Container, FileEdit, Label, LineEdit

from eisgreis.docks import NO_FILES, float_validator
from eisgreis.filters import FILTER_CHOICES
from eisgreis.io.mrc import MRC_SUFFIXES

if TYPE_CHECKING:
    from eisgreis.app import Controller


class FilterDock:
    MRC_PATTERNS = " ".join(f"*{suffix}" for suffix in MRC_SUFFIXES)

    def __init__(self, controller: "Controller") -> None:
        self.c = controller
        self._updating = False
        self.add_file = FileEdit(label="Add MRC", mode="r", filter=self.MRC_PATTERNS, value=None)
        # Choices as a callable: napari re-reads every dock widget's choices (reset_choices) whenever a layer is
        # added or removed, and a fixed list would fall back to the one the box was made with.
        self.selected_file = ComboBox(label="File", choices=self._file_choice_names, value=NO_FILES)
        self.pixel_size = LineEdit(label="Pixel size (Å/px)", value="1.0")
        self.cutoff = LineEdit(label="Low-pass (Å)", value="20")
        self.filter_type_box = ComboBox(label="Filter", choices=FILTER_CHOICES, value="gaussian")
        self.show_original_box = CheckBox(label="Show original", value=False)
        self.status = Label(value="Registered 0 tomograms; loaded 0")

        self.pixel_size.native.setValidator(float_validator())
        self.cutoff.native.setValidator(float_validator())

        self.add_file.changed.connect(self._on_add_file_changed)
        self.selected_file.changed.connect(self._on_selected_file_changed)
        self.pixel_size.native.editingFinished.connect(self._on_pixel_size_edit_finished)
        self.cutoff.native.editingFinished.connect(self._on_cutoff_edit_finished)
        self.filter_type_box.changed.connect(self._on_filter_type_changed)
        self.show_original_box.changed.connect(self._on_show_original_changed)

        self.container = Container(
            widgets=[
                self.add_file,
                self.selected_file,
                self.pixel_size,
                self.cutoff,
                self.filter_type_box,
                self.show_original_box,
                self.status,
            ],
            labels=True,
        )
        self.container.native.setWindowTitle("MRC Filter")

    # ── what the viewer reads ──────────────────────────────────────────────────────────────────────────────────

    def cutoff_value(self) -> Optional[float]:
        try:
            return float(self.cutoff.value)
        except ValueError:  # an empty or half-typed field: no cutoff yet
            return None

    def filter_type(self) -> str:
        return self.filter_type_box.value

    # ── what the viewer sets ───────────────────────────────────────────────────────────────────────────────────

    def show_pixel_size(self, value: float) -> None:
        self.pixel_size.value = str(value)

    def show_cutoff(self, value: float) -> None:
        self.cutoff.value = str(value)

    def show_original(self, value: bool) -> None:
        self.show_original_box.value = value

    def update_status(self) -> None:
        loaded = 1 if self.c.session.current is not None else 0
        self.status.value = f"Registered {len(self.c.session.paths)} tomograms; loaded {loaded}"

    def _file_choice_names(self, _widget=None) -> list[str]:
        controller = getattr(self, "c", None)
        labels = list(controller.session.file_labels()) if controller is not None else []
        return labels or [NO_FILES]

    def refresh_files(self) -> None:
        """The file list again, from the session, keeping the selection (or the tomogram on screen)."""
        was_updating = self._updating
        self._updating = True
        try:
            current_value = self.selected_file.current_choice
            self.selected_file.reset_choices()
            labels = self._file_choice_names()
            if labels == [NO_FILES]:
                self.selected_file.value = NO_FILES
                self._sync_native(labels, NO_FILES)
                self.update_status()
                return
            if current_value in labels:
                selected = current_value
            elif self.c.session.current is not None and self.c.session.label_for(self.c.session.current):
                selected = self.c.session.label_for(self.c.session.current)
            else:
                selected = labels[0]
            self.selected_file.value = selected
            self._sync_native(labels, selected)
            self.update_status()
        finally:
            self._updating = was_updating

    def select(self, path: Path) -> None:
        """Show `path` as the file on screen, and empty the Add field."""
        self._updating = True
        try:
            label = self.c.session.label_for(path)
            if label is not None:
                self.selected_file.reset_choices()
                self._sync_native(self._file_choice_names(), label)
                self.selected_file.value = label
            self.add_file.value = ""
        finally:
            self._updating = False

    def _sync_native(self, labels: list[str], selected: str) -> None:
        """The Qt combo box's items in the list's order, the selection set, without a change signal."""
        native = self.selected_file.native
        if not hasattr(native, "clear") or not hasattr(native, "addItems"):
            return
        native.blockSignals(True)
        try:
            native.clear()
            for label in labels:
                native.addItem(label, label)
            native.setCurrentIndex(labels.index(selected) if selected in labels else 0)
        finally:
            native.blockSignals(False)

    # ── handlers ───────────────────────────────────────────────────────────────────────────────────────────────

    def _on_add_file_changed(self, value: object) -> None:
        if self._updating or value in (None, ""):
            return
        self.c.add_path(Path(value), activate=True)

    def _on_selected_file_changed(self, value: object) -> None:
        if self._updating or value == NO_FILES:
            return
        labels = self.c.session.file_labels()
        path = labels.get(str(value))
        if path is None and hasattr(self.selected_file.native, "currentText"):
            path = labels.get(self.selected_file.native.currentText())
        if path is not None:
            self.c.open(path)

    def _on_filter_type_changed(self, _value: object) -> None:
        self._apply_cutoff()

    def _on_cutoff_edit_finished(self) -> None:
        self._apply_cutoff()

    def _apply_cutoff(self) -> None:
        if self._updating or self.c.lowpass is None:
            return
        cutoff = self.cutoff_value()
        if cutoff is None:
            return
        self.c.apply_lowpass(cutoff, self.filter_type())

    def _on_pixel_size_edit_finished(self) -> None:
        if self._updating or self.c.original_data is None:
            return
        try:
            pixel_size = float(self.pixel_size.value)
        except ValueError:  # an empty or half-typed field
            return
        if pixel_size > 0:
            self.c.set_pixel_size(pixel_size)

    def _on_show_original_changed(self, value: bool) -> None:
        if self.c.source_layer is not None:
            self.c.source_layer.visible = value
