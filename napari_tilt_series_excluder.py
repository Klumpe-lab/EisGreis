import argparse
import os
from dataclasses import dataclass
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
from magicgui.widgets import CheckBox, ComboBox, Container, FileEdit, FloatSlider, Label, PushButton
from napari.utils.notifications import show_info

from shared_filters import apply_lowpass, frequency_grid, resolution_bounds
from shared_mdoc import MdocTiltEntry, parse_mdoc, resolve_tilt_image_path, sort_entries_by_tilt_angle
from shared_mrc import read_mrc_layer_data, read_voxel_size_from_path

try:
    import imageio.v3 as iio
except Exception:  # pragma: no cover - runtime fallback
    iio = None


@dataclass(frozen=True)
class TiltRecord:
    entry: MdocTiltEntry
    path: Path

    @property
    def label(self) -> str:
        return f"{self.entry.tilt_angle:+.1f} deg | {self.path.name}"


class TiltSeriesLowPassController:
    FILTER_CHOICES = ("hard", "gaussian", "butterworth", "cosine")

    def __init__(self, viewer: napari.Viewer) -> None:
        self.viewer = viewer
        self._updating_controls = False
        self.current_original_data: Optional[np.ndarray] = None
        self.current_frequency_grid: Optional[np.ndarray] = None
        self.image_layer: Optional[napari.layers.Image] = None
        self.current_path: Optional[Path] = None

        self.controls = self._build_controls()
        self.viewer.window.add_dock_widget(self.controls, area="right")

    def _build_controls(self) -> Container:
        lowpass_enabled = CheckBox(label="Low-pass enabled", value=False)
        cutoff = FloatSlider(label="Low-pass (A)", min=2.0, max=50.0, step=0.5, value=10.0)
        filter_type = ComboBox(label="Filter", choices=self.FILTER_CHOICES, value="gaussian")
        status = Label(value="Load a tilt series to enable low-pass filtering.")

        lowpass_enabled.changed.connect(self._on_lowpass_controls_changed)
        cutoff.changed.connect(self._on_lowpass_controls_changed)
        filter_type.changed.connect(self._on_lowpass_controls_changed)

        container = Container(
            widgets=[
                lowpass_enabled,
                cutoff,
                filter_type,
                status,
            ],
            labels=True,
        )
        container.native.setWindowTitle("Tilt Low-pass")
        container.lowpass_enabled = lowpass_enabled
        container.cutoff = cutoff
        container.filter_type = filter_type
        container.status = status
        return container

    def set_current_tilt(self, path: Path, data: np.ndarray, image_layer: napari.layers.Image) -> None:
        voxel_size = self._read_voxel_size(path)
        self.current_original_data = data
        self.current_path = path
        self.image_layer = image_layer
        self.current_frequency_grid = frequency_grid(data.shape, voxel_size)
        self._configure_lowpass_controls(data.shape, voxel_size)
        self.controls.status.value = f"Filtering {path.name}"
        self._render_current_image()

    def _read_voxel_size(self, path: Path) -> float:
        if path.suffix.lower() in TiltSeriesExcluderController.MRC_SUFFIXES:
            voxel_size = read_voxel_size_from_path(path)
            if voxel_size is not None and np.isfinite(voxel_size) and voxel_size > 0:
                return float(voxel_size)
        return 1.0

    def _configure_lowpass_controls(self, shape: tuple[int, ...], voxel_size: float) -> None:
        min_resolution, max_resolution = resolution_bounds(shape, voxel_size)
        self._updating_controls = True
        try:
            self.controls.cutoff.min = min_resolution
            self.controls.cutoff.max = max_resolution
            self.controls.cutoff.value = min(
                max(self.controls.cutoff.value, min_resolution),
                max_resolution,
            )
        finally:
            self._updating_controls = False

    def _on_lowpass_controls_changed(self, _value: object) -> None:
        if self._updating_controls:
            return
        self._render_current_image()

    def _filtered_or_original_data(self) -> np.ndarray:
        if self.current_original_data is None:
            raise ValueError("No tilt image is loaded.")
        if not self.controls.lowpass_enabled.value or self.current_frequency_grid is None:
            return self.current_original_data
        return apply_lowpass(
            self.current_original_data,
            self.current_frequency_grid,
            self.controls.cutoff.value,
            self.controls.filter_type.value,
        )

    def _render_current_image(self) -> None:
        if self.image_layer is None or self.current_original_data is None:
            return
        display_data = self._filtered_or_original_data()
        self.image_layer.data = display_data
        self._auto_contrast(display_data)

    def _auto_contrast(self, data: np.ndarray) -> None:
        if self.image_layer is None:
            return
        low, high = np.percentile(data, (1.0, 99.0))
        if not np.isfinite(low) or not np.isfinite(high):
            return
        if low == high:
            high = low + 1e-6
        self.image_layer.contrast_limits = (float(low), float(high))


class TiltSeriesExcluderController:
    IMAGE_PATTERNS = "*.mrc *.mrcs *.map *.st *.rec *.preali *.ali *.tif *.tiff *.png *.jpg *.jpeg"
    MRC_SUFFIXES = {".mrc", ".mrcs", ".map", ".st", ".rec", ".preali", ".ali"}

    def __init__(self, viewer: napari.Viewer) -> None:
        self.viewer = viewer
        self.image_layer: Optional[napari.layers.Image] = None
        self.records: list[TiltRecord] = []
        self.record_labels: dict[str, int] = {}
        self.selection: list[bool] = []
        self.current_index = -1
        self._updating_controls = False
        self.lowpass_controller = TiltSeriesLowPassController(viewer)

        self.controls = self._build_controls()
        self.viewer.window.add_dock_widget(self.controls, area="right")

    def _build_controls(self) -> Container:
        mdoc_file = FileEdit(label="Mdoc file", mode="r", filter="*.mdoc", value=None)
        tilt_directory = FileEdit(label="Tilt folder", mode="d", value=None)
        load_button = PushButton(text="Load tilt series")
        selected_tilt = ComboBox(label="Tilt image", choices=["No tilt series loaded"], value="No tilt series loaded")
        keep_tilt = CheckBox(label="Keep current tilt", value=True)
        previous_button = PushButton(text="Previous")
        next_button = PushButton(text="Next")
        keep_all_button = PushButton(text="Keep all")
        exclude_all_button = PushButton(text="Exclude all")
        status = Label(value="Load an .mdoc file and a tilt folder.")

        load_button.changed.connect(self._on_load_clicked)
        selected_tilt.changed.connect(self._on_selected_tilt_changed)
        keep_tilt.changed.connect(self._on_keep_tilt_changed)
        previous_button.changed.connect(self._show_previous_tilt)
        next_button.changed.connect(self._show_next_tilt)
        keep_all_button.changed.connect(self._keep_all)
        exclude_all_button.changed.connect(self._exclude_all)

        container = Container(
            widgets=[
                mdoc_file,
                tilt_directory,
                load_button,
                selected_tilt,
                keep_tilt,
                previous_button,
                next_button,
                keep_all_button,
                exclude_all_button,
                status,
            ],
            labels=True,
        )
        container.native.setWindowTitle("Tilt Series Excluder")
        container.mdoc_file = mdoc_file
        container.tilt_directory = tilt_directory
        container.load_button = load_button
        container.selected_tilt = selected_tilt
        container.keep_tilt = keep_tilt
        container.previous_button = previous_button
        container.next_button = next_button
        container.keep_all_button = keep_all_button
        container.exclude_all_button = exclude_all_button
        container.status = status
        return container

    def _on_load_clicked(self, _value: object) -> None:
        mdoc_value = self.controls.mdoc_file.value
        tilt_dir_value = self.controls.tilt_directory.value
        if mdoc_value in (None, "") or tilt_dir_value in (None, ""):
            show_info("Select both an .mdoc file and a tilt folder.")
            return

        try:
            self.load_tilt_series(Path(mdoc_value), Path(tilt_dir_value))
        except Exception as exc:
            show_info(str(exc))

    def load_tilt_series(self, mdoc_path: Path, tilt_directory: Path) -> None:
        entries = sort_entries_by_tilt_angle(parse_mdoc(mdoc_path))
        records = [TiltRecord(entry=entry, path=resolve_tilt_image_path(entry, tilt_directory)) for entry in entries]
        if not records:
            raise ValueError(f"No tilt images could be resolved from {mdoc_path}")

        self.records = records
        self.selection = [True] * len(records)
        self.record_labels = {record.label: idx for idx, record in enumerate(records)}
        self.controls.selected_tilt.choices = list(self.record_labels.keys())
        self.controls.status.value = (
            f"Loaded {len(records)} tilts from {mdoc_path.name}. Ordered by TiltAngle."
        )
        self.show_tilt(0)

    def show_tilt(self, index: int) -> None:
        if not self.records:
            return

        index = index % len(self.records)
        record = self.records[index]
        data = self._read_tilt_image(record.path)

        if self.image_layer is None:
            self.image_layer = self.viewer.add_image(data, name=record.path.name, colormap="gray")
        else:
            self.image_layer.data = data
            self.image_layer.name = record.path.name

        self.lowpass_controller.set_current_tilt(record.path, data, self.image_layer)
        self.current_index = index
        self._sync_controls_to_current_record()
        self.viewer.reset_view()
        kept = sum(self.selection)
        self.viewer.status = (
            f"{index + 1}/{len(self.records)} | angle {record.entry.tilt_angle:+.1f} deg | "
            f"kept {kept}/{len(self.records)}"
        )

    def _read_tilt_image(self, path: Path) -> np.ndarray:
        suffix = path.suffix.lower()
        if suffix in self.MRC_SUFFIXES:
            data, _kwargs, _layer_type = read_mrc_layer_data(path)
        elif iio is not None:
            data = np.asarray(iio.imread(path), dtype=np.float32)
        else:
            data, _kwargs, _layer_type = read_mrc_layer_data(path)

        while data.ndim > 2:
            data = data.mean(axis=0)
        return np.asarray(data, dtype=np.float32)

    def _sync_controls_to_current_record(self) -> None:
        if self.current_index < 0:
            return

        record = self.records[self.current_index]
        self._updating_controls = True
        try:
            self.controls.selected_tilt.value = record.label
            self.controls.keep_tilt.value = self.selection[self.current_index]
            self.controls.status.value = (
                f"{record.label} | ZValue {record.entry.z_value} | "
                f"{'kept' if self.selection[self.current_index] else 'excluded'}"
            )
        finally:
            self._updating_controls = False

    def _on_selected_tilt_changed(self, value: str) -> None:
        if self._updating_controls or value == "No tilt series loaded":
            return
        index = self.record_labels.get(value)
        if index is not None:
            self.show_tilt(index)

    def _on_keep_tilt_changed(self, value: bool) -> None:
        if self._updating_controls or self.current_index < 0:
            return
        self.selection[self.current_index] = bool(value)
        self._sync_controls_to_current_record()

    def _show_previous_tilt(self, _value: object) -> None:
        if self.current_index >= 0:
            self.show_tilt(self.current_index - 1)

    def _show_next_tilt(self, _value: object) -> None:
        if self.current_index >= 0:
            self.show_tilt(self.current_index + 1)

    def _keep_all(self, _value: object) -> None:
        if not self.records:
            return
        self.selection = [True] * len(self.records)
        self._sync_controls_to_current_record()

    def _exclude_all(self, _value: object) -> None:
        if not self.records:
            return
        self.selection = [False] * len(self.records)
        self._sync_controls_to_current_record()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Browse tilt images resolved from an .mdoc file and a tilt-image folder."
    )
    parser.add_argument("--mdoc", type=Path, help="Path to a SerialEM .mdoc file.")
    parser.add_argument("--tilt-dir", type=Path, help="Directory containing tilt images.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    viewer = napari.Viewer(title="Napari Tilt Series Excluder")
    controller = TiltSeriesExcluderController(viewer)

    if args.mdoc is not None:
        controller.controls.mdoc_file.value = args.mdoc
    if args.tilt_dir is not None:
        controller.controls.tilt_directory.value = args.tilt_dir
    if args.mdoc is not None and args.tilt_dir is not None:
        controller.load_tilt_series(args.mdoc, args.tilt_dir)

    napari.run()


if __name__ == "__main__":
    main()
