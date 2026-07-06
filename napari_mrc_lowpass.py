import argparse
import csv
import os
from pathlib import Path
from typing import Optional

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

try:
    import napari
except ModuleNotFoundError as exc:
    if exc.name == "napari":
        script_path = Path(__file__).resolve()
        raise SystemExit(
            "napari is not installed in the current Python environment.\n"
            "Use the napari-lorenz environment instead:\n"
            f"  conda run -n napari-lorenz python {script_path}\n"
            "or:\n"
            "  conda activate napari-lorenz\n"
            f"  python {script_path}"
        ) from exc
    raise

import numpy as np
import napari.layers
from magicgui.widgets import CheckBox, ComboBox, Container, FileEdit, Label, LineEdit, PushButton
from qtpy.QtCore import QRegularExpression, QTimer
from qtpy.QtGui import QRegularExpressionValidator
from qtpy.QtWidgets import QSizePolicy
from napari_builtins._measure_shapes import toggle_shape_measures
from shared_filters import apply_lowpass, extract_voxel_size, frequency_grid, resolution_bounds
from shared_mrc import read_mrc_layer_data, read_voxel_size_array, read_voxel_size_from_path
from shared_star import load_points_from_star, sanitize_star_token, save_points_to_star


def _patch_napari_text_dpi_fallback() -> None:
    try:
        from napari._vispy.visuals.text import Text as NapariVispyText
    except Exception:
        return
    if getattr(NapariVispyText, "_napari_lorenz_dpi_fallback", False):
        return

    def dpi_ratio(self) -> float:
        return (getattr(self.transforms, "dpi", 96) or 96) / 96

    NapariVispyText.dpi_ratio = property(dpi_ratio)
    NapariVispyText._napari_lorenz_dpi_fallback = True


_patch_napari_text_dpi_fallback()


class StartupProgress:
    def __init__(self, matched_count: int) -> None:
        self.matched_count = matched_count
        self.bar_width = 32
        self._last_len = 0

    def update(self, fraction: float, message: str) -> None:
        fraction = max(0.0, min(1.0, fraction))
        filled = int(round(self.bar_width * fraction))
        bar = "#" * filled + "-" * (self.bar_width - filled)
        text = f"\r[{bar}] {fraction * 100:5.1f}%  {message}"
        padding = " " * max(0, self._last_len - len(text))
        print(text + padding, end="", flush=True)
        self._last_len = len(text)

    def done(self, message: str = "Startup complete") -> None:
        self.update(1.0, message)
        print(flush=True)


class LowPassController:
    FILTER_CHOICES = ("hard", "gaussian", "butterworth", "cosine")
    MRC_PATTERNS = "*.mrc *.mrcs *.map *.st *.rec *.preali *.ali"
    SUPPORTED_SUFFIXES = (".mrc", ".mrcs", ".map", ".st", ".rec", ".preali", ".ali")
    SCALE_BAR_PREFERRED_VALUES = (1, 2, 5, 10, 15, 20, 25, 50, 75, 100, 125, 150, 200, 500, 750)

    def __init__(self, viewer: napari.Viewer, startup_progress: Optional[StartupProgress] = None) -> None:
        self.viewer = viewer
        self.startup_progress = startup_progress
        self.source_layer: Optional[napari.layers.Image] = None
        self.filtered_layer: Optional[napari.layers.Image] = None
        self.measurement_layer: Optional[napari.layers.Shapes] = None
        self.original_data: Optional[np.ndarray] = None
        self.frequency_grid: Optional[np.ndarray] = None
        self.voxel_size_angstrom: float = 1.0
        self.header_voxel_size_angstrom: float = 1.0
        self.min_resolution_angstrom: float = 2.0
        self.max_resolution_angstrom: float = 50.0
        self.current_path: Optional[Path] = None
        self.loaded_paths: list[Path] = []
        self.file_labels: dict[str, Path] = {}
        self.comments: dict[Path, str] = {}
        self.path_pixel_sizes_angstrom: dict[Path, float] = {}
        self.particle_layers: dict[str, napari.layers.Points] = {}
        self.particle_coordinates: dict[str, dict[Path, np.ndarray]] = {}
        self.particle_box_sizes_angstrom: dict[str, float] = {}
        self.default_comments_csv = Path.cwd() / "tomogram_comments.csv"
        self.default_star_dir = Path.cwd() / "particle_stars"
        self.default_png_dir = Path.cwd() / "tomogram_pngs"
        self._updating_controls = False

        self.controls = self._build_filter_controls()
        self.comment_controls = self._build_comment_controls()
        self.annotation_controls = self._build_annotation_controls()
        self.export_controls = self._build_export_controls()
        self.viewer.window.add_dock_widget(self.controls, area="right")
        self.viewer.window.add_dock_widget(self.comment_controls, area="right")
        self.viewer.window.add_dock_widget(self.annotation_controls, area="right")
        self.viewer.window.add_dock_widget(self.export_controls, area="right")
        self.viewer.layers.events.inserted.connect(self._on_layer_inserted)
        self.viewer.layers.events.removed.connect(self._on_layer_removed)
        self.viewer.window._qt_window.destroyed.connect(self._on_window_destroyed)

    def _build_filter_controls(self):
        add_file = FileEdit(label="Add MRC", mode="r", filter=self.MRC_PATTERNS, value=None)
        selected_file = ComboBox(label="File", choices=["No files loaded"], value="No files loaded")
        pixel_size = LineEdit(label="Pixel size (Å/px)", value="1.0")
        cutoff = LineEdit(label="Low-pass (Å)", value="20")
        filter_type = ComboBox(label="Filter", choices=self.FILTER_CHOICES, value="gaussian")
        show_original = CheckBox(label="Show original", value=False)
        status = Label(value="Registered 0 tomograms; loaded 0")

        float_validator = QRegularExpressionValidator(QRegularExpression(r"[0-9]*\.?[0-9]*"))
        pixel_size.native.setValidator(float_validator)
        cutoff.native.setValidator(float_validator)

        add_file.changed.connect(self._on_add_file_changed)
        selected_file.changed.connect(self._on_selected_file_changed)
        pixel_size.native.editingFinished.connect(self._on_pixel_size_edit_finished)
        cutoff.native.editingFinished.connect(self._on_cutoff_edit_finished)
        filter_type.changed.connect(self._on_filter_controls_changed)
        show_original.changed.connect(self._on_show_original_changed)

        container = Container(
            widgets=[
                add_file,
                selected_file,
                pixel_size,
                cutoff,
                filter_type,
                show_original,
                status,
            ],
            labels=True,
        )
        container.native.setWindowTitle("MRC Filter")
        container.add_file = add_file
        container.selected_file = selected_file
        container.pixel_size = pixel_size
        container.cutoff = cutoff
        container.filter_type = filter_type
        container.show_original = show_original
        container.status = status
        return container

    def _build_comment_controls(self):
        comment = LineEdit(label="Comment", value="")
        load_comments = FileEdit(label="Load comments CSV", mode="r", filter="*.csv", value=None)
        save_comments = FileEdit(label="Save comments CSV", mode="w", filter="*.csv", value=str(self.default_comments_csv))
        save_now = PushButton(label="Save comments now")

        comment.changed.connect(self._on_comment_changed)
        load_comments.changed.connect(self._on_load_comments_changed)
        save_now.changed.connect(self._on_save_now)

        container = Container(
            widgets=[comment, load_comments, save_comments, save_now],
            labels=True,
        )
        container.native.setWindowTitle("Tomogram Comments")
        container.comment = comment
        container.load_comments = load_comments
        container.save_comments = save_comments
        container.save_now = save_now
        return container

    def _build_annotation_controls(self):
        particle_name = LineEdit(label="Particle name", value="ribosome")
        particle_layers = ComboBox(label="Particle layer", choices=["No particle layers"], value="No particle layers")
        box_size = LineEdit(label="Box size (Å)", value="200")
        star_dir = LineEdit(label="STAR output dir", value=str(self.default_star_dir))
        load_star_dir = FileEdit(label="Load STAR dir", mode="d", value=None)
        create_layer = PushButton(label="Create particle layer")
        load_layers = PushButton(label="Load particle STARs")
        save_layer = PushButton(label="Save selected STAR")
        save_all = PushButton(label="Save all STARs")
        status = Label(value="STAR output ready")

        float_validator = QRegularExpressionValidator(QRegularExpression(r"[0-9]*\.?[0-9]*"))
        box_size.native.setValidator(float_validator)

        create_layer.changed.connect(self._on_create_particle_layer)
        load_layers.changed.connect(self._on_load_particle_stars)
        save_layer.changed.connect(self._on_save_selected_star)
        save_all.changed.connect(self._on_save_all_stars)
        particle_layers.changed.connect(self._on_particle_layer_selected)
        box_size.native.editingFinished.connect(self._on_box_size_edit_finished)

        container = Container(
            widgets=[
                particle_name,
                particle_layers,
                box_size,
                star_dir,
                load_star_dir,
                create_layer,
                load_layers,
                save_layer,
                save_all,
                status,
            ],
            labels=True,
        )
        container.native.setWindowTitle("Particle Annotations")
        container.particle_name = particle_name
        container.particle_layers = particle_layers
        container.box_size = box_size
        container.star_dir = star_dir
        container.load_star_dir = load_star_dir
        container.create_layer = create_layer
        container.load_layers = load_layers
        container.save_layer = save_layer
        container.save_all = save_all
        container.status = status
        self._configure_status_label(status)
        return container

    def _build_export_controls(self):
        save_png = PushButton(label="Save current view PNG")

        save_png.changed.connect(self._on_save_current_view_png)

        container = Container(widgets=[save_png], labels=True)
        container.native.setWindowTitle("Export PNG")
        container.save_png = save_png
        return container

    def _configure_status_label(self, label: Label) -> None:
        native = label.native
        native.setWordWrap(False)
        native.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        native.setMaximumHeight(native.fontMetrics().height() + 8)
        native.setToolTip(str(label.value))

    def _on_add_file_changed(self, value: object) -> None:
        if self._updating_controls or value in (None, ""):
            return
        self.add_path(Path(value), activate=True)

    def _on_selected_file_changed(self, value: object) -> None:
        if self._updating_controls or value == "No files loaded":
            return
        label = str(value)
        path = self.file_labels.get(label)
        if path is None and hasattr(self.controls.selected_file.native, "currentText"):
            label = self.controls.selected_file.native.currentText()
            path = self.file_labels.get(label)
        if path is not None:
            self.load_mrc(path)

    def _on_filter_controls_changed(self, _value: object) -> None:
        if self._updating_controls or self.original_data is None or self.frequency_grid is None:
            return
        try:
            cutoff_value = float(self.controls.cutoff.value)
        except ValueError:
            return
        self.apply_lowpass(cutoff_value, self.controls.filter_type.value)

    def _on_cutoff_edit_finished(self) -> None:
        if self._updating_controls or self.original_data is None or self.frequency_grid is None:
            return
        try:
            cutoff_value = float(self.controls.cutoff.value)
        except ValueError:
            return
        self.apply_lowpass(cutoff_value, self.controls.filter_type.value)

    def _on_pixel_size_edit_finished(self) -> None:
        if self._updating_controls or self.original_data is None:
            return
        try:
            pixel_size = float(self.controls.pixel_size.value)
        except ValueError:
            return
        if pixel_size <= 0:
            return
        self.voxel_size_angstrom = pixel_size
        if self.current_path is not None:
            self.path_pixel_sizes_angstrom[self.current_path] = pixel_size
        self.frequency_grid = frequency_grid(self.original_data.shape, self.voxel_size_angstrom)
        self.min_resolution_angstrom, self.max_resolution_angstrom = resolution_bounds(
            self.original_data.shape,
            self.voxel_size_angstrom,
        )
        self._update_layer_spatial_calibration()
        try:
            cutoff_value = float(self.controls.cutoff.value)
        except ValueError:
            cutoff_value = min(max(20.0, self.min_resolution_angstrom), self.max_resolution_angstrom)
            self.controls.cutoff.value = str(cutoff_value)
        self.apply_lowpass(cutoff_value, self.controls.filter_type.value)
        self._update_all_particle_layer_sizes()
        self._show_scale_bar_when_ready()

    def _on_box_size_edit_finished(self) -> None:
        if self._updating_controls:
            return
        layer = self._get_selected_particle_layer()
        try:
            box_size = float(self.annotation_controls.box_size.value)
        except ValueError:
            return
        if box_size <= 0:
            return
        if layer is None:
            particle_name = self.annotation_controls.particle_name.value.strip() or "particle"
            self.particle_box_sizes_angstrom[particle_name] = box_size
            return
        particle_name = layer.metadata.get("particle_name", layer.name)
        self.particle_box_sizes_angstrom[particle_name] = box_size
        layer.metadata["box_size_angstrom"] = box_size
        self._apply_particle_layer_style(layer)

    def _on_show_original_changed(self, value: bool) -> None:
        if self.source_layer is not None:
            self.source_layer.visible = value

    def _on_comment_changed(self, value: str) -> None:
        if self._updating_controls or self.current_path is None:
            return
        self.comments[self.current_path] = value.strip()

    def _on_load_comments_changed(self, value: object) -> None:
        if self._updating_controls or value in (None, ""):
            return
        self.load_comments_csv(Path(value))
        self._updating_controls = True
        try:
            self.comment_controls.load_comments.value = ""
        finally:
            self._updating_controls = False

    def _on_save_now(self, _value: object) -> None:
        self.save_comments_csv(self._comments_csv_path())

    def _on_window_destroyed(self, *_args) -> None:
        self.save_comments_csv(self._comments_csv_path())
        self.save_all_particle_layers()

    def _on_layer_inserted(self, event) -> None:
        layer = event.value
        if layer in (self.source_layer, self.filtered_layer, self.measurement_layer):
            return

        if isinstance(layer, napari.layers.Points) and layer.metadata.get("annotation_kind") == "particle":
            self._bind_particle_layer_events(layer)
            self._refresh_particle_layer_choices()
            return

        source_path = getattr(getattr(layer, "source", None), "path", None)
        if not source_path:
            return

        path = Path(source_path).expanduser().resolve()
        if path.suffix.lower() not in self.SUPPORTED_SUFFIXES:
            return

        if path not in self.loaded_paths:
            self.loaded_paths.append(path)
            self.loaded_paths.sort()
            self._refresh_selected_file_choices()
            self._update_status()

    def _on_layer_removed(self, _event) -> None:
        stale_names = [
            name for name, layer in self.particle_layers.items() if layer not in self.viewer.layers
        ]
        for name in stale_names:
            self.particle_layers.pop(name, None)
            self.particle_coordinates.pop(name, None)
        if self.measurement_layer is not None and self.measurement_layer not in self.viewer.layers:
            self.measurement_layer = None
        self._refresh_particle_layer_choices()

    def add_path(self, path: Path, activate: bool = False) -> None:
        resolved = self.register_path(path)
        if self.current_path is None:
            self.load_mrc(resolved)
        elif activate and resolved != self.current_path:
            self.load_mrc(resolved)

    def register_path(self, path: Path) -> Path:
        resolved = path.expanduser().resolve()
        if resolved not in self.loaded_paths:
            self.loaded_paths.append(resolved)
            self.loaded_paths.sort()
            self._refresh_selected_file_choices()
            self._update_status()
        return resolved

    def _refresh_selected_file_choices(self) -> None:
        was_updating = self._updating_controls
        self._updating_controls = True
        try:
            current_value = self.controls.selected_file.current_choice
            if not self.loaded_paths:
                self.file_labels = {}
                self.controls.selected_file._default_choices = ["No files loaded"]
                self.controls.selected_file.choices = ["No files loaded"]
                self.controls.selected_file.value = "No files loaded"
                self._sync_native_file_combo(["No files loaded"], "No files loaded")
                self._update_status()
                return

            labels: list[str] = []
            self.file_labels = {}
            counts: dict[str, int] = {}
            for path in self.loaded_paths:
                base = path.name
                counts[base] = counts.get(base, 0) + 1
                label = base if counts[base] == 1 else f"{base} [{counts[base]}]"
                self.file_labels[label] = path
                labels.append(label)

            self.controls.selected_file._default_choices = labels
            self.controls.selected_file.choices = labels
            if current_value in labels:
                self.controls.selected_file.value = current_value
                selected_label = current_value
            elif self.current_path is not None:
                selected_label = self._label_for_path(self.current_path)
                if selected_label is not None:
                    self.controls.selected_file.value = selected_label
                else:
                    selected_label = labels[0]
                    self.controls.selected_file.value = selected_label
            else:
                selected_label = labels[0]
                self.controls.selected_file.value = selected_label
            self._sync_native_file_combo(labels, selected_label)
            self._update_status()
        finally:
            self._updating_controls = was_updating

    def _label_for_path(self, path: Path) -> Optional[str]:
        for label, candidate in self.file_labels.items():
            if candidate == path:
                return label
        return None

    def _sync_native_file_combo(self, labels: list[str], selected_label: str) -> None:
        native = self.controls.selected_file.native
        if not hasattr(native, "clear") or not hasattr(native, "addItems"):
            return
        native.blockSignals(True)
        try:
            native.clear()
            for label in labels:
                native.addItem(label, label)
            index = labels.index(selected_label) if selected_label in labels else 0
            native.setCurrentIndex(index)
        finally:
            native.blockSignals(False)

    def _set_selected_file(self, path: Path) -> None:
        self._updating_controls = True
        try:
            label = self._label_for_path(path)
            if label is not None:
                self._sync_native_file_combo(list(self.file_labels.keys()), label)
                try:
                    self.controls.selected_file.value = label
                except ValueError:
                    self.controls.selected_file.choices = list(self.file_labels.keys())
                    self.controls.selected_file.value = label
            self.controls.add_file.value = ""
            self.comment_controls.comment.value = self.comments.get(path, "")
        finally:
            self._updating_controls = False
        self._sync_particle_layer_visibility()
        self._refresh_particle_layer_choices()

    def _update_status(self) -> None:
        loaded_count = 1 if self.current_path is not None else 0
        self.controls.status.value = f"Registered {len(self.loaded_paths)} tomograms; loaded {loaded_count}"

    def _set_annotation_status(self, message: str, detail: Optional[str] = None) -> None:
        self.annotation_controls.status.value = message
        self.annotation_controls.status.native.setToolTip(detail or message)

    def _star_output_dir(self) -> Path:
        raw_path = self.annotation_controls.star_dir.value.strip()
        if not raw_path:
            return self.default_star_dir
        return Path(raw_path).expanduser().resolve()

    def _comments_csv_path(self) -> Path:
        raw_path = self.comment_controls.save_comments.value
        if raw_path in (None, ""):
            return self.default_comments_csv
        return Path(raw_path).expanduser().resolve()

    def _png_output_dir(self) -> Path:
        return self.default_png_dir

    def load_comments_csv(self, csv_path: Path) -> None:
        loaded_comments: dict[Path, str] = {}
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames and {"path", "comment"}.issubset(set(reader.fieldnames)):
                for row in reader:
                    path_value = (row.get("path") or "").strip()
                    if not path_value:
                        continue
                    loaded_comments[Path(path_value).expanduser().resolve()] = (row.get("comment") or "").strip()
            else:
                handle.seek(0)
                plain_reader = csv.reader(handle)
                for row in plain_reader:
                    if len(row) < 2:
                        continue
                    path_value = row[0].strip()
                    if path_value.lower() == "path":
                        continue
                    loaded_comments[Path(path_value).expanduser().resolve()] = row[1].strip()

        self.comments.update(loaded_comments)
        if self.current_path is not None:
            self._updating_controls = True
            try:
                self.comment_controls.comment.value = self.comments.get(self.current_path, "")
            finally:
                self._updating_controls = False

    def save_comments_csv(self, csv_path: Path) -> None:
        if self.current_path is not None:
            self.comments[self.current_path] = self.comment_controls.comment.value.strip()

        rows = []
        all_paths = sorted(set(self.loaded_paths) | set(self.comments.keys()))
        for path in all_paths:
            rows.append({"path": str(path), "comment": self.comments.get(path, "")})

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "comment"])
            writer.writeheader()
            writer.writerows(rows)

    def _particle_layers_for_current_path(self) -> list[napari.layers.Points]:
        return [self.particle_layers[name] for name in sorted(self.particle_layers)]

    def _particle_display_size_pixels(self, particle_name: str) -> float:
        box_size_angstrom = self.particle_box_sizes_angstrom.get(particle_name, 200.0)
        pixel_size = self.voxel_size_angstrom if self.voxel_size_angstrom > 0 else 1.0
        return max(box_size_angstrom / pixel_size, 2.0)

    def _store_particle_coordinates_for_current_path(self) -> None:
        if self.current_path is None:
            return
        for particle_name, layer in self.particle_layers.items():
            per_path = self.particle_coordinates.setdefault(particle_name, {})
            per_path[self.current_path] = np.asarray(layer.data, dtype=float).copy()

    def _restore_particle_coordinates_for_current_path(self) -> None:
        if self.current_path is None:
            return
        point_dims = int(self.source_layer.ndim) if self.source_layer is not None and int(self.source_layer.ndim) >= 2 else 2
        empty = np.empty((0, point_dims), dtype=float)
        for particle_name, layer in self.particle_layers.items():
            coords = self.particle_coordinates.get(particle_name, {}).get(self.current_path)
            layer.data = coords.copy() if coords is not None else empty.copy()
            layer.visible = True
            layer.metadata["tomogram_path"] = str(self.current_path)
            self._apply_particle_layer_style(layer)

    def _refresh_particle_layer_choices(self) -> None:
        layers = self._particle_layers_for_current_path()
        self._updating_controls = True
        try:
            if not layers:
                self.annotation_controls.particle_layers.choices = ["No particle layers"]
                self.annotation_controls.particle_layers.value = "No particle layers"
                if self.current_path is None:
                    self._set_annotation_status("No particle layers", f"STAR dir: {self._star_output_dir()}")
                else:
                    self._set_annotation_status(
                        f"No particle layers for {self.current_path.name}",
                        f"STAR dir: {self._star_output_dir()}",
                    )
                return
            names = [layer.metadata.get("particle_name", layer.name) for layer in layers]
            current = self.annotation_controls.particle_layers.value
            self.annotation_controls.particle_layers.choices = names
            self.annotation_controls.particle_layers.value = current if current in names else names[0]
            selected_name = self.annotation_controls.particle_layers.value
            if selected_name in self.particle_box_sizes_angstrom:
                self.annotation_controls.box_size.value = str(self.particle_box_sizes_angstrom[selected_name])
            self._set_annotation_status(
                f"{len(names)} particle layer(s) for {self.current_path.name}",
                f"STAR dir: {self._star_output_dir()}",
            )
        finally:
            self._updating_controls = False

    def _sync_particle_layer_visibility(self) -> None:
        visible = self.current_path is not None
        for layer in self.particle_layers.values():
            layer.visible = visible

    def _bind_particle_layer_events(self, layer: napari.layers.Points) -> None:
        if layer.metadata.get("_particle_events_bound"):
            return
        layer.events.name.connect(lambda _event: self._refresh_particle_layer_choices())
        layer.metadata["_particle_events_bound"] = True

    def _apply_particle_layer_style(self, layer: napari.layers.Points) -> None:
        particle_name = layer.metadata.get("particle_name", layer.name)
        layer.size = self._particle_display_size_pixels(particle_name)
        layer.face_color_mode = "direct"
        layer.border_color_mode = "direct"
        layer.face_color = "transparent"
        layer.current_face_color = "transparent"
        layer.border_color = "red"
        layer.current_border_color = "red"
        layer.border_width_is_relative = False
        layer.border_width = 3
        layer.current_border_width = 3
        layer.symbol = "disc"
        layer.opacity = 0.9
        layer.out_of_slice_display = True

    def _update_all_particle_layer_sizes(self) -> None:
        for layer in self.particle_layers.values():
            self._apply_particle_layer_style(layer)

    def _spatial_scale_for_current_data(self) -> tuple[float, ...]:
        if self.original_data is None:
            return (float(self.voxel_size_angstrom),)
        ndim = max(int(self.original_data.ndim), 1)
        pixel_size = float(self.voxel_size_angstrom) if self.voxel_size_angstrom > 0 else 1.0
        return tuple(pixel_size for _ in range(ndim))

    def _unit_tuple_for_ndim(self, ndim: int) -> tuple[str, ...]:
        return tuple("angstrom" for _ in range(max(ndim, 1)))

    def _update_layer_spatial_calibration(self) -> None:
        scale = self._spatial_scale_for_current_data()
        units = self._unit_tuple_for_ndim(len(scale))

        for layer in (self.source_layer, self.filtered_layer):
            if layer is None:
                continue
            layer.scale = scale
            layer.units = units

        for layer in self.particle_layers.values():
            layer.scale = scale
            layer.units = units

        if self.measurement_layer is not None:
            measurement_ndim = int(self.measurement_layer.ndim)
            self.measurement_layer.scale = tuple(1.0 for _ in range(measurement_ndim))
            self.measurement_layer.units = self._unit_tuple_for_ndim(measurement_ndim)

    def _preferred_scale_bar_length_angstrom(self, desired_length: float) -> float:
        if not np.isfinite(desired_length) or desired_length <= 0:
            return 1.0
        candidates: list[float] = []
        for power in range(-6, 7):
            factor = 1000.0**power
            candidates.extend(value * factor for value in self.SCALE_BAR_PREFERRED_VALUES)
        candidates = sorted(value for value in candidates if value > 0)
        smaller = [value for value in candidates if value <= desired_length]
        return smaller[-1] if smaller else candidates[0]

    def _format_angstrom_length(self, value: float) -> str:
        if 0.01 <= abs(value) < 1_000_000:
            text = f"{value:.3f}".rstrip("0").rstrip(".")
        else:
            text = f"{value:.3g}"
        return f"{text} Å"

    def _scale_bar_visual(self):
        qt_viewer = getattr(self.viewer.window, "_qt_viewer", None)
        canvas = getattr(qt_viewer, "canvas", None)
        overlay_map = getattr(canvas, "_overlay_to_visual", {})
        visuals = overlay_map.get(self.viewer.scale_bar, [])
        return visuals[0] if visuals else None

    def _prepare_scale_bar_visual(self, visual) -> bool:
        if visual is None:
            return False
        text = getattr(getattr(visual, "node", None), "text", None)
        transforms = getattr(text, "transforms", None)
        if transforms is not None and not hasattr(transforms, "dpi"):
            try:
                transforms.dpi = 96
            except Exception:
                return False
        return True

    def _show_scale_bar_when_ready(self, attempts: int = 0) -> None:
        visual = self._scale_bar_visual()
        if not self._prepare_scale_bar_visual(visual):
            if attempts < 20:
                QTimer.singleShot(50, lambda: self._show_scale_bar_when_ready(attempts + 1))
            return

        try:
            self._patch_scale_bar_zoom_callback(visual)
            self.viewer.scale_bar.unit = "angstrom"
            self.viewer.scale_bar.length = None
            self.viewer.scale_bar.visible = True
            self._force_scale_bar_angstrom_labels()
        except AttributeError:
            if attempts < 20:
                self.viewer.scale_bar.visible = False
                QTimer.singleShot(50, lambda: self._show_scale_bar_when_ready(attempts + 1))
        except Exception:
            pass

    def _patch_scale_bar_zoom_callback(self, visual) -> None:
        if getattr(visual, "_napari_lorenz_zoom_callback_patched", False):
            return

        original_callback = visual._on_size_or_zoom_change

        def safe_scale_bar_zoom_callback(*_args, force: bool = False, **_kwargs) -> None:
            if not self._prepare_scale_bar_visual(visual):
                self.viewer.scale_bar.visible = False
                QTimer.singleShot(50, self._show_scale_bar_when_ready)
                return
            try:
                original_callback(force=force)
            except AttributeError:
                self.viewer.scale_bar.visible = False
                QTimer.singleShot(50, self._show_scale_bar_when_ready)

        try:
            self.viewer.camera.events.zoom.disconnect(original_callback)
        except Exception:
            pass

        visual._napari_lorenz_original_zoom_callback = original_callback
        visual._napari_lorenz_safe_zoom_callback = safe_scale_bar_zoom_callback
        visual._napari_lorenz_zoom_callback_patched = True
        self.viewer.camera.events.zoom.connect(safe_scale_bar_zoom_callback)

    def _force_scale_bar_angstrom_labels(self) -> None:
        visual = self._scale_bar_visual()
        if not self._prepare_scale_bar_visual(visual):
            return

        def update_angstrom_scale_bar(*_args, force: bool = False, **_kwargs) -> None:
            scale = 1 / self.viewer.camera.zoom
            if not force and getattr(visual, "_napari_lorenz_last_scale", None) == scale:
                return
            visual._napari_lorenz_last_scale = scale
            target_world_angstrom = scale * visual._target_length
            length_angstrom = self._preferred_scale_bar_length_angstrom(target_world_angstrom)
            visual._current_length = length_angstrom / scale
            visual.node.text.text = self._format_angstrom_length(length_angstrom)
            self._prepare_scale_bar_visual(visual)
            visual._on_rendering_change()
            visual._on_position_change()

        if not getattr(visual, "_napari_lorenz_angstrom_labels", False):
            visual._napari_lorenz_angstrom_labels = True
            visual._napari_lorenz_update_angstrom_scale_bar = update_angstrom_scale_bar
            self.viewer.camera.events.zoom.connect(update_angstrom_scale_bar)

        visual._napari_lorenz_update_angstrom_scale_bar(force=True)

    def _ensure_measurement_layer(self) -> None:
        if self.source_layer is None:
            return
        ndim = int(self.source_layer.ndim)
        if self.measurement_layer is not None and self.measurement_layer in self.viewer.layers:
            if int(self.measurement_layer.ndim) == ndim:
                self._update_layer_spatial_calibration()
                return
            self.measurement_layer.visible = False

        layer = self.viewer.add_shapes(
            data=[],
            ndim=ndim,
            name="Measurements (Å)",
            shape_type="line",
            edge_color="yellow",
            face_color="transparent",
            edge_width=3,
            opacity=0.95,
            scale=tuple(1.0 for _ in range(ndim)),
            units=self._unit_tuple_for_ndim(ndim),
            metadata={"annotation_kind": "measurement_angstrom"},
        )
        self.measurement_layer = layer
        toggle_shape_measures(layer)
        layer.text = "L = {_perimeter:.3g} Å"
        layer.mode = "add_line"
        self._update_layer_spatial_calibration()

    def _create_particle_layer(self, particle_name: str, update_status: bool = True) -> napari.layers.Points:
        if self.current_path is None or self.source_layer is None:
            raise RuntimeError("No active tomogram loaded")
        existing_layer = self.particle_layers.get(particle_name)
        if existing_layer is not None:
            self.annotation_controls.particle_name.value = particle_name
            self.annotation_controls.particle_layers.value = particle_name
            if particle_name in self.particle_box_sizes_angstrom:
                self.annotation_controls.box_size.value = str(self.particle_box_sizes_angstrom[particle_name])
            self.viewer.layers.selection.active = existing_layer
            existing_layer.mode = "add"
            return existing_layer

        point_dims = int(self.source_layer.ndim) if int(self.source_layer.ndim) >= 2 else 2
        try:
            box_size_angstrom = float(self.annotation_controls.box_size.value)
        except ValueError:
            box_size_angstrom = 200.0
        if box_size_angstrom <= 0:
            box_size_angstrom = 200.0
        layer = self.viewer.add_points(
            data=np.empty((0, point_dims), dtype=float),
            name=particle_name,
            metadata={
                "annotation_kind": "particle",
                "particle_name": particle_name,
                "tomogram_path": str(self.current_path),
                "box_size_angstrom": box_size_angstrom,
            },
        )
        self.particle_layers[particle_name] = layer
        self.particle_coordinates.setdefault(particle_name, {})
        self.particle_box_sizes_angstrom[particle_name] = box_size_angstrom
        self._apply_particle_layer_style(layer)
        layer.scale = self._spatial_scale_for_current_data()
        layer.units = self._unit_tuple_for_ndim(int(layer.ndim))
        layer.mode = "add"
        layer.visible = True
        self._bind_particle_layer_events(layer)
        self._refresh_selected_file_choices()
        if self.current_path is not None:
            self._set_selected_file(self.current_path)
        self._sync_particle_layer_visibility()
        self._refresh_particle_layer_choices()
        self.annotation_controls.particle_layers.value = particle_name
        self.annotation_controls.particle_name.value = particle_name
        self.annotation_controls.box_size.value = str(box_size_angstrom)
        self.viewer.layers.selection.active = layer
        if update_status:
            self._set_annotation_status(
                f"Created particle layer '{particle_name}'",
                f"STAR dir: {self._star_output_dir()}",
            )
        return layer

    def _on_create_particle_layer(self, _value: object) -> None:
        if self.current_path is None or self.source_layer is None:
            return
        particle_name = self.annotation_controls.particle_name.value.strip() or "particle"
        self._create_particle_layer(particle_name, update_status=True)

    def _on_load_particle_stars(self, _value: object) -> None:
        raw_dir = self.annotation_controls.load_star_dir.value
        if raw_dir in (None, ""):
            raw_dir = str(self._star_output_dir())
        self.load_particle_stars_from_dir(Path(raw_dir))

    def load_particle_stars_from_dir(self, star_dir: Path) -> None:
        star_dir = star_dir.expanduser().resolve()
        if not star_dir.exists():
            self._set_annotation_status("STAR folder not found", str(star_dir))
            return

        path_by_stem = {path.stem: path for path in self.loaded_paths}
        imported = 0
        for star_path in sorted(star_dir.glob("*.star")):
            stem = star_path.stem
            if "__" not in stem:
                continue
            tomogram_stem, particle_token = stem.split("__", 1)
            tomogram_path = path_by_stem.get(tomogram_stem)
            if tomogram_path is None:
                continue
            coords_angstrom, box_size_angstrom = load_points_from_star(star_path)
            pixel_size = self.path_pixel_sizes_angstrom.get(tomogram_path)
            if pixel_size is None:
                pixel_size = read_voxel_size_from_path(tomogram_path) or 1.0
                self.path_pixel_sizes_angstrom[tomogram_path] = pixel_size
            particle_name = particle_token.replace("_", " ").strip() or particle_token
            coords_pixels = coords_angstrom / pixel_size if coords_angstrom.size else coords_angstrom
            self.particle_coordinates.setdefault(particle_name, {})[tomogram_path] = coords_pixels
            if box_size_angstrom is not None:
                self.particle_box_sizes_angstrom[particle_name] = box_size_angstrom
            self._create_particle_layer(particle_name, update_status=False)
            imported += 1

        self._restore_particle_coordinates_for_current_path()
        self._refresh_particle_layer_choices()
        self._set_annotation_status(
            f"Loaded {imported} STAR file(s)",
            f"STAR dir: {star_dir}",
        )

    def _get_selected_particle_layer(self) -> Optional[napari.layers.Points]:
        selected_name = self.annotation_controls.particle_layers.value
        if selected_name in ("", "No particle layers"):
            return None
        return self.particle_layers.get(selected_name)

    def _on_particle_layer_selected(self, value: str) -> None:
        if self._updating_controls or value in ("", "No particle layers"):
            return
        layer = self._get_selected_particle_layer()
        if layer is not None:
            particle_name = layer.metadata.get("particle_name", layer.name)
            if particle_name in self.particle_box_sizes_angstrom:
                self._updating_controls = True
                try:
                    self.annotation_controls.box_size.value = str(self.particle_box_sizes_angstrom[particle_name])
                    self.annotation_controls.particle_name.value = particle_name
                finally:
                    self._updating_controls = False
            self.viewer.layers.selection.active = layer

    def _current_slice_suffix(self) -> str:
        if self.source_layer is None:
            return "slice"
        point = list(self.viewer.dims.point)
        if len(point) >= 3:
            z = int(round(point[0]))
            y = int(round(point[1]))
            x = int(round(point[2]))
            return f"x{x}_y{y}_z{z}"
        if len(point) == 2:
            y = int(round(point[0]))
            x = int(round(point[1]))
            return f"x{x}_y{y}"
        return "_".join(f"d{idx}_{int(round(value))}" for idx, value in enumerate(point))

    def _on_save_current_view_png(self, _value: object) -> None:
        if self.current_path is None:
            return
        image = self.viewer.screenshot(canvas_only=True)
        output_dir = self._png_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"{self.current_path.stem}__{self._current_slice_suffix()}.png"
        output_path = output_dir / file_name
        try:
            import imageio.v3 as iio

            iio.imwrite(output_path, image)
        except Exception:
            from imageio import imwrite

            imwrite(output_path, image)
        self._set_annotation_status("Saved PNG", str(output_path))

    def _star_path_for_layer(self, layer: napari.layers.Points) -> Path:
        if self.current_path is None:
            raise RuntimeError("No active tomogram loaded")
        tomogram_path = self.current_path
        layer_token = sanitize_star_token(layer.metadata.get("particle_name", layer.name))
        file_name = f"{tomogram_path.stem}__{layer_token}.star"
        return self._star_output_dir() / file_name

    def save_particle_layer(self, layer: napari.layers.Points) -> None:
        if self.current_path is None:
            return
        particle_name = layer.metadata.get("particle_name", layer.name)
        coords = self.particle_coordinates.get(particle_name, {}).get(self.current_path)
        if coords is None:
            coords = np.asarray(layer.data, dtype=float)
        pixel_size = self.path_pixel_sizes_angstrom.get(self.current_path, self.voxel_size_angstrom)
        box_size_angstrom = self.particle_box_sizes_angstrom.get(
            particle_name,
            layer.metadata.get("box_size_angstrom"),
        )
        save_points_to_star(
            coords * pixel_size,
            self._star_path_for_layer(layer),
            particle_name,
            box_size_angstrom=box_size_angstrom,
        )

    def _on_save_selected_star(self, _value: object) -> None:
        layer = self._get_selected_particle_layer()
        if layer is not None:
            self._store_particle_coordinates_for_current_path()
            self.save_particle_layer(layer)
            self._set_annotation_status(
                f"Saved '{layer.metadata.get('particle_name', layer.name)}'",
                str(self._star_path_for_layer(layer)),
            )

    def save_all_particle_layers(self) -> None:
        self._store_particle_coordinates_for_current_path()
        saved = 0
        for particle_name, per_path in self.particle_coordinates.items():
            for path, coords in per_path.items():
                file_name = f"{path.stem}__{sanitize_star_token(particle_name)}.star"
                pixel_size = self.path_pixel_sizes_angstrom.get(path, self.header_voxel_size_angstrom)
                save_points_to_star(
                    coords * pixel_size,
                    self._star_output_dir() / file_name,
                    particle_name,
                    box_size_angstrom=self.particle_box_sizes_angstrom.get(particle_name),
                )
                saved += 1
        self._set_annotation_status(
            f"Saved {saved} particle STAR file(s)",
            f"STAR dir: {self._star_output_dir()}",
        )

    def _on_save_all_stars(self, _value: object) -> None:
        self.save_all_particle_layers()

    def load_mrc(self, path: Path) -> None:
        path = path.expanduser().resolve()
        preferred_particle_name: Optional[str] = None
        keep_measurement_active = self.viewer.layers.selection.active is self.measurement_layer
        selected_particle_layer = self._get_selected_particle_layer()
        if selected_particle_layer is not None:
            preferred_particle_name = selected_particle_layer.metadata.get("particle_name")
        if self.current_path is not None and self.current_path != path:
            self._store_particle_coordinates_for_current_path()
        progress = self.startup_progress if self.current_path is None else None
        if progress is not None:
            progress.update(0.30, f"Opening first tomogram: {path.name}")
        data, add_kwargs, _layer_type = read_mrc_layer_data(path)
        if progress is not None:
            progress.update(0.50, f"Reading header: {path.name}")
        voxel_size = read_voxel_size_array(path)

        self.current_path = path
        if path not in self.loaded_paths:
            self.loaded_paths.append(path)
            self.loaded_paths.sort()
        self._refresh_selected_file_choices()
        self._set_selected_file(path)
        self.original_data = data
        self.header_voxel_size_angstrom = extract_voxel_size(voxel_size)
        self.voxel_size_angstrom = self.header_voxel_size_angstrom
        self.path_pixel_sizes_angstrom[path] = self.voxel_size_angstrom
        self.controls.pixel_size.value = str(self.header_voxel_size_angstrom)
        self.frequency_grid = frequency_grid(data.shape, self.voxel_size_angstrom)
        self.min_resolution_angstrom, self.max_resolution_angstrom = resolution_bounds(data.shape, self.voxel_size_angstrom)
        if progress is not None:
            progress.update(0.68, f"Preparing frequency grid: {path.name}")
        default_cutoff = min(
            max(30.0, self.min_resolution_angstrom),
            self.max_resolution_angstrom,
        )
        self.controls.cutoff.value = str(default_cutoff)
        self.controls.show_original.value = False

        if self.source_layer is None:
            self.source_layer = self.viewer.add_image(
                data,
                name=add_kwargs.get("name", f"{path.name} (original)"),
                colormap="gray",
                visible=False,
            )
        else:
            self.source_layer.data = data
            self.source_layer.name = add_kwargs.get("name", f"{path.name} (original)")
            self.source_layer.colormap = "gray"
            self.source_layer.visible = False
        self._update_layer_spatial_calibration()

        if progress is not None:
            progress.update(0.82, f"Applying low-pass filter: {path.name}")
        filtered = apply_lowpass(data, self.frequency_grid, default_cutoff, self.controls.filter_type.value)
        if self.filtered_layer is None:
            self.filtered_layer = self.viewer.add_image(
                filtered,
                name=f"{path.name} (low-pass)",
                colormap="gray",
            )
        else:
            self.filtered_layer.data = filtered
            self.filtered_layer.name = f"{path.name} (low-pass)"
            self.filtered_layer.colormap = "gray"
        self._update_layer_spatial_calibration()

        self._auto_contrast(self.filtered_layer.data, self.filtered_layer)
        self._update_status()
        self._ensure_measurement_layer()
        self._sync_particle_layer_visibility()
        self._restore_particle_coordinates_for_current_path()
        self._refresh_particle_layer_choices()
        if preferred_particle_name:
            matching_layer = self.particle_layers.get(preferred_particle_name)
            if matching_layer is None:
                matching_layer = self._create_particle_layer(preferred_particle_name, update_status=False)
            self.annotation_controls.particle_name.value = preferred_particle_name
            self.annotation_controls.particle_layers.value = preferred_particle_name
            self.viewer.layers.selection.active = matching_layer
            matching_layer.mode = "add"
        elif keep_measurement_active and self.measurement_layer is not None:
            self.viewer.layers.selection.active = self.measurement_layer
            self.measurement_layer.mode = "add_line"
        elif self.filtered_layer is not None:
            self.viewer.layers.selection.active = self.filtered_layer
        if progress is not None:
            progress.done(f"Loaded 1/{self.matched_count_for_status()} tomograms; current: {path.name}")
            self.startup_progress = None

        self.viewer.scale_bar.visible = False
        self.viewer.reset_view()
        self._show_scale_bar_when_ready()

    def matched_count_for_status(self) -> int:
        return len(self.loaded_paths)

    def apply_lowpass(self, resolution_angstrom: float, filter_type: str) -> None:
        if self.original_data is None or self.filtered_layer is None:
            return
        filtered = apply_lowpass(self.original_data, self.frequency_grid, resolution_angstrom, filter_type)
        self.filtered_layer.data = filtered
        self.filtered_layer.name = f"{self.current_path.name} ({filter_type} low-pass)"
        self._auto_contrast(filtered, self.filtered_layer)

    def _auto_contrast(self, data: np.ndarray, layer: napari.layers.Image) -> None:
        low, high = np.percentile(data, (1.0, 99.0))
        if not np.isfinite(low) or not np.isfinite(high):
            return
        if low == high:
            high = low + 1e-6
        layer.contrast_limits = (float(low), float(high))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open MRC volumes in napari and interactively apply a low-pass filter."
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Optional .mrc files to open on launch.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    startup_progress = StartupProgress(len(args.paths)) if args.paths else None
    if startup_progress is not None:
        startup_progress.update(0.0, f"Matched {len(args.paths)} tomograms")
    viewer = napari.Viewer(title="Napari MRC Low-pass Viewer")
    controller = LowPassController(viewer, startup_progress=startup_progress)
    for index, path in enumerate(args.paths, start=1):
        controller.register_path(path)
        if startup_progress is not None:
            fraction = 0.05 + 0.20 * (index / max(len(args.paths), 1))
            startup_progress.update(fraction, f"Registered {index}/{len(args.paths)} tomograms")
    if args.paths:
        controller.load_mrc(Path(args.paths[0]))
    napari.run()


if __name__ == "__main__":
    main()
