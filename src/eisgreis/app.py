"""The viewer: napari with EisGreis's docks around one tomogram on screen.

The controller here opens a tomogram (memory-mapped, its FFT taken once, low-passed at 30 Å or the nearest the box
allows), keeps its calibration (the header's pixel size or the filter dock's, in Å, on every layer), the measurement
layer (lengths in nm), the scale bar (nm) and the docks in step. What it holds lives in ``session.py``; the docks
are in ``docks/``. Drivers use ``eisgreis.api``, never this module.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import napari
import napari.layers
import numpy as np

from eisgreis import _napari
from eisgreis.docks.comments import CommentsDock
from eisgreis.docks.export import ExportDock
from eisgreis.docks.filter import FilterDock
from eisgreis.docks.particles import KIND as PARTICLE_KIND
from eisgreis.docks.particles import ParticlesDock
from eisgreis.filters import LowPassFilter, contrast_limits, resolution_bounds
from eisgreis.io.mrc import MRC_SUFFIXES, extract_voxel_size, read_mrc_layer_data, read_voxel_size_array
from eisgreis.reference import ReferenceLayers
from eisgreis.runtime import RuntimeProfile
from eisgreis.scalebar import length_and_label
from eisgreis.session import Outputs, Session
from eisgreis.view3d import Volume3D

logger = logging.getLogger(__name__)

MEASUREMENT_UNIT_ANGSTROM = 10.0  # the measurement layer's data unit: 1 nm, so napari's lengths read in nm


class StartupProgress:
    """A progress bar on the terminal while the first tomogram opens."""

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


class Controller:
    """The tomogram on screen, its layers and the docks (see the module docstring)."""

    def __init__(
        self,
        viewer: napari.Viewer,
        startup_progress: Optional[StartupProgress] = None,
        profile: Optional[RuntimeProfile] = None,
        outputs: Optional[Outputs] = None,
    ) -> None:
        self.viewer = viewer
        self.profile = profile or RuntimeProfile()
        self.startup_progress = startup_progress
        self.session = Session(outputs)
        self.source_layer: Optional[napari.layers.Image] = None
        self.filtered_layer: Optional[napari.layers.Image] = None
        self.measurement_layer: Optional[napari.layers.Shapes] = None
        self.original_data: Optional[np.ndarray] = None
        self.lowpass: Optional[LowPassFilter] = None
        self.voxel_size_angstrom: float = 1.0
        self.header_voxel_size_angstrom: float = 1.0
        self.min_resolution_angstrom: float = 2.0
        self.max_resolution_angstrom: float = 50.0
        self.closed = False

        self.filter_dock = FilterDock(self)
        self.comments_dock = CommentsDock(self)
        self.particles_dock = ParticlesDock(self)
        self.export_dock = ExportDock(self)
        for dock in (self.filter_dock, self.comments_dock, self.particles_dock, self.export_dock):
            # napari names a dock by `name` only; without one it is "Dock widget N".
            viewer.window.add_dock_widget(dock.container, area="right", name=dock.container.native.windowTitle())
        viewer.layers.events.inserted.connect(self._on_layer_inserted)
        viewer.layers.events.removed.connect(self._on_layer_removed)
        _napari.on_window_closed(viewer, self._on_window_closed)
        _napari.nm_scale_bar(viewer, length_and_label)
        # Reference layers (picks, segmentations, other volumes of the tomogram) and 3-D on any renderer.
        self.reference = ReferenceLayers(self)
        self.view3d = Volume3D(self)

    # ── tomograms ──────────────────────────────────────────────────────────────────────────────────────────────

    def register(self, path: Path) -> Path:
        """Add a tomogram to the file list without opening it."""
        resolved = self.session.register(path)
        self.filter_dock.refresh_files()
        return resolved

    def add_path(self, path: Path, activate: bool = False) -> None:
        resolved = self.register(path)
        if self.session.current is None or (activate and resolved != self.session.current):
            self.open(resolved)

    def forget(self, path: Path) -> None:
        """Drop a tomogram from the file list, with its comment, particles and reference layers."""
        path = Path(path).expanduser().resolve()
        self.reference.forget(path)
        was_current = self.session.current == path
        self.session.forget(path)
        if was_current:
            self.session.current = None
        self.filter_dock.refresh_files()

    def show_current_file(self) -> None:
        """The docks on the tomogram on screen: the file list's selection, its comment, its particle layers."""
        current = self.session.current
        if current is not None:
            self.filter_dock.select(current)
        self.comments_dock.show_comment()
        self.particles_dock.sync_visibility()
        self.particles_dock.refresh_choices()

    def open(self, path: Path, title: Optional[str] = None) -> None:
        """Open the tomogram at `path`: read it, low-pass it, put the docks on it, and show its middle slice with the
        whole volume in view. `title` names the window."""
        path = Path(path).expanduser().resolve()
        session = self.session
        preferred_particle_name = self.particles_dock.selected_particle_name()
        keep_measurement_active = self.viewer.layers.selection.active is self.measurement_layer
        if session.current is not None and session.current != path:
            self.particles_dock.store_current()
        progress = self.startup_progress if session.current is None else None
        if progress is not None:
            progress.update(0.30, f"Opening first tomogram: {path.name}")
        if self.lowpass is not None:
            # Free the previous tomogram's cached FFT before reading the next one.
            self.lowpass.release()
        data, add_kwargs, _layer_type = read_mrc_layer_data(path)
        if progress is not None:
            progress.update(0.50, f"Reading header: {path.name}")
        voxel_size = read_voxel_size_array(path)

        session.current = path
        session.register(path)
        self.filter_dock.refresh_files()
        self.show_current_file()
        self.original_data = data
        self.header_voxel_size_angstrom = extract_voxel_size(voxel_size)
        self.voxel_size_angstrom = self.header_voxel_size_angstrom
        session.pixel_sizes[path] = self.voxel_size_angstrom
        self.filter_dock.show_pixel_size(self.header_voxel_size_angstrom)
        self.lowpass = LowPassFilter(
            data, self.voxel_size_angstrom, workers=self.profile.workers, cache_spectrum=self.profile.cache_spectrum
        )
        self.min_resolution_angstrom, self.max_resolution_angstrom = resolution_bounds(
            data.shape, self.voxel_size_angstrom
        )
        if progress is not None:
            progress.update(0.68, f"Preparing frequency grid: {path.name}")
        default_cutoff = min(max(30.0, self.min_resolution_angstrom), self.max_resolution_angstrom)
        self.filter_dock.show_cutoff(default_cutoff)
        self.filter_dock.show_original(False)

        scale, units = self.spatial_scale(data.ndim), ("angstrom",) * max(int(data.ndim), 1)
        source_name = add_kwargs.get("name", f"{path.name} (original)")
        if self.source_layer is None:
            self.source_layer = self.viewer.add_image(
                data, name=source_name, colormap="gray", visible=False, scale=scale, units=units
            )
        else:
            self.source_layer.data = data
            self.source_layer.name = source_name
            self.source_layer.colormap = "gray"
            self.source_layer.visible = False
        self.update_calibration()

        if progress is not None:
            progress.update(0.82, f"Applying low-pass filter: {path.name}")
        filtered = self.lowpass.apply(default_cutoff, self.filter_dock.filter_type())
        if self.filtered_layer is None:
            self.filtered_layer = self.viewer.add_image(
                filtered, name=f"{path.name} (low-pass)", colormap="gray", scale=scale, units=units
            )
        else:
            self.filtered_layer.data = filtered
            self.filtered_layer.name = f"{path.name} (low-pass)"
            self.filtered_layer.colormap = "gray"
        self.update_calibration()
        self.view3d.watch(self.source_layer)
        self.view3d.watch(self.filtered_layer)

        self._auto_contrast(self.filtered_layer.data, self.filtered_layer)
        self.filter_dock.update_status()
        self._ensure_measurement_layer()
        self.particles_dock.sync_visibility()
        self.particles_dock.restore_current()
        self.particles_dock.refresh_choices()
        if preferred_particle_name:
            matching_layer = self.particles_dock.layers.get(preferred_particle_name)
            if matching_layer is None:
                matching_layer = self.particles_dock.create_layer(preferred_particle_name, update_status=False)
            self.particles_dock.particle_name.value = preferred_particle_name
            self.particles_dock.particle_layers.value = preferred_particle_name
            self.viewer.layers.selection.active = matching_layer
            matching_layer.mode = "add"
        elif keep_measurement_active and self.measurement_layer is not None:
            self.viewer.layers.selection.active = self.measurement_layer
            self.measurement_layer.mode = "add_line"
        elif self.filtered_layer is not None:
            self.viewer.layers.selection.active = self.filtered_layer
        if progress is not None:
            progress.done(f"Loaded 1/{len(session.paths)} tomograms; current: {path.name}")
            self.startup_progress = None

        self.reference.show(path)
        self._centre()
        self.viewer.canvas.overlays.scale_bar.visible = True
        _napari.refresh_scale_bar(self.viewer)
        if title:
            self.viewer.title = title

    def _centre(self) -> None:
        """The middle slice, the whole volume in view. napari keeps the slider's world position when a layer's scale
        changes, so without this a new tomogram opens wherever the last one's position falls on it."""
        layer = self.filtered_layer or self.source_layer
        if layer is not None and layer.ndim == 3:
            z_mid = layer.data.shape[0] // 2
            self.viewer.dims.set_point(0, float(layer.data_to_world((z_mid, 0, 0))[0]))
        self.viewer.reset_view()

    def swap_volume(self, path: Path, label: str = "") -> None:
        """Show another volume of the tomogram on screen in its place (a denoised version): the same grid, so
        particles, comments, measurements and STAR names stay with the tomogram that was opened. The low-pass is
        applied afresh at the current cutoff. Raises when the grid differs."""
        if self.session.current is None or self.source_layer is None or self.filtered_layer is None:
            raise RuntimeError("No active tomogram loaded")
        data, _add_kwargs, _layer_type = read_mrc_layer_data(Path(path))
        if self.original_data is not None and data.shape != self.original_data.shape:
            raise ValueError(f"{Path(path).name} is {data.shape}, the tomogram {self.original_data.shape}")
        if self.lowpass is not None:
            self.lowpass.release()
        self.original_data = data
        self.lowpass = LowPassFilter(
            data, self.voxel_size_angstrom, workers=self.profile.workers, cache_spectrum=self.profile.cache_spectrum
        )
        self.source_layer.data = data
        suffix = f" · {label}" if label else ""
        self.source_layer.name = f"{self.session.current.name}{suffix} (original)"
        cutoff = self.filter_dock.cutoff_value()
        if cutoff is None:
            cutoff = self.min_resolution_angstrom
        self.apply_lowpass(cutoff, self.filter_dock.filter_type())
        self.filtered_layer.name = f"{self.filtered_layer.name}{suffix}"

    # ── filtering and calibration ──────────────────────────────────────────────────────────────────────────────

    def apply_lowpass(self, resolution_angstrom: float, filter_type: str) -> None:
        if self.lowpass is None or self.filtered_layer is None:
            return
        filtered = self.lowpass.apply(resolution_angstrom, filter_type)
        self.filtered_layer.data = filtered
        self.filtered_layer.name = f"{self.session.current.name} ({filter_type} low-pass)"
        self._auto_contrast(filtered, self.filtered_layer)

    def set_pixel_size(self, pixel_size: float) -> None:
        """The tomogram on screen is `pixel_size` Å per voxel: recalibrate every layer and filter again."""
        self.voxel_size_angstrom = pixel_size
        if self.session.current is not None:
            self.session.pixel_sizes[self.session.current] = pixel_size
        if self.lowpass is not None:
            self.lowpass.set_voxel_size(self.voxel_size_angstrom)
        self.min_resolution_angstrom, self.max_resolution_angstrom = resolution_bounds(
            self.original_data.shape, self.voxel_size_angstrom
        )
        self.update_calibration()
        cutoff = self.filter_dock.cutoff_value()
        if cutoff is None:
            cutoff = min(max(20.0, self.min_resolution_angstrom), self.max_resolution_angstrom)
            self.filter_dock.show_cutoff(cutoff)
        self.apply_lowpass(cutoff, self.filter_dock.filter_type())
        self.particles_dock.restyle_all()
        _napari.refresh_scale_bar(self.viewer)

    def spatial_scale(self, ndim: Optional[int] = None) -> tuple[float, ...]:
        """The scale of a layer on the tomogram's grid: the pixel size in Å on every axis."""
        if ndim is None:
            ndim = int(self.original_data.ndim) if self.original_data is not None else 1
        pixel_size = float(self.voxel_size_angstrom) if self.voxel_size_angstrom > 0 else 1.0
        return tuple(pixel_size for _ in range(max(int(ndim), 1)))

    def update_calibration(self) -> None:
        """Every layer on the tomogram's current pixel size (Å)."""
        scale = self.spatial_scale()
        units = tuple("angstrom" for _ in scale)
        for layer in (self.source_layer, self.filtered_layer):
            if layer is not None:
                layer.scale = scale
                layer.units = units
        for layer in self.particles_dock.layers.values():
            layer.scale = self.spatial_scale(layer.ndim)
            layer.units = tuple("angstrom" for _ in range(layer.ndim))
        if getattr(self, "reference", None) is not None:
            self.reference.rescale()
        if self.measurement_layer is not None:
            ndim = int(self.measurement_layer.ndim)
            # The measurement layer's data unit is 1 nm (the world's is Å), so its lengths read in nm.
            self.measurement_layer.scale = tuple(MEASUREMENT_UNIT_ANGSTROM for _ in range(ndim))
            self.measurement_layer.units = tuple("angstrom" for _ in range(ndim))

    def ensure_z_thickness(self, min_angstrom: float) -> None:
        """Make the Z slice at least `min_angstrom` (plus a voxel) thick, so a point that size whose sphere the slice
        cuts is drawn: napari draws points only inside the slice's thickness. Images still show their single
        plane, and shapes theirs."""
        thickness = list(self.viewer.dims.thickness)
        z_axis = len(thickness) - 3
        if z_axis < 0:
            return
        px = float(self.voxel_size_angstrom) if self.voxel_size_angstrom > 0 else 1.0
        wanted = float(min_angstrom) + px
        if thickness[z_axis] < wanted:
            thickness[z_axis] = wanted
            self.viewer.dims.thickness = tuple(thickness)

    def _auto_contrast(self, data: np.ndarray, layer: napari.layers.Image) -> None:
        low, high = contrast_limits(data, self.profile.contrast_samples)
        if not np.isfinite(low) or not np.isfinite(high):
            return
        if low == high:
            high = low + 1e-6
        layer.contrast_limits = (float(low), float(high))

    def _ensure_measurement_layer(self) -> None:
        if self.source_layer is None:
            return
        ndim = int(self.source_layer.ndim)
        if self.measurement_layer is not None and self.measurement_layer in self.viewer.layers:
            if int(self.measurement_layer.ndim) == ndim:
                self.update_calibration()
                return
            self.measurement_layer.visible = False

        layer = self.viewer.add_shapes(
            data=[],
            ndim=ndim,
            name="Measurements (nm)",
            shape_type="line",
            edge_color="yellow",
            face_color="transparent",
            edge_width=0.3,  # in nm data units: a 3 Å line
            opacity=0.95,
            scale=tuple(MEASUREMENT_UNIT_ANGSTROM for _ in range(ndim)),
            units=tuple("angstrom" for _ in range(ndim)),
            metadata={"annotation_kind": "measurement_nm"},
        )
        self.measurement_layer = layer
        if _napari.toggle_shape_measures(layer):
            layer.text = "L = {_perimeter:.3g} nm"
        layer.mode = "add_line"
        self.update_calibration()

    # ── saving ─────────────────────────────────────────────────────────────────────────────────────────────────

    def set_status(self, message: str, detail: Optional[str] = None) -> None:
        """A line in the Particle Annotations dock's status (saves and exports report there)."""
        if not self.closed:
            self.particles_dock.set_status(message, detail)

    def save(self) -> None:
        """The comments CSV and every particle STAR, into the current folders."""
        self.session.save_comments_csv()
        self.particles_dock.save_all()

    def set_outputs(self, folder: Path) -> None:
        """Save into `folder` from now on: its comments CSV, ``particle_stars/`` and ``tomogram_pngs/``. What the
        viewer holds is saved into the old folders first."""
        if self.session.current is not None:
            self.save()
        self.session.set_outputs(Outputs.under(folder))
        self.comments_dock.show_outputs()
        self.particles_dock.show_outputs()

    def reload_annotations(self) -> None:
        """Read the comments and the particle STARs back from the current folders, where they exist."""
        outputs = self.session.default_outputs
        if outputs.comments_csv.is_file():
            self.comments_dock.load_csv(outputs.comments_csv)
        if outputs.star_dir.is_dir():
            self.particles_dock.load_stars(outputs.star_dir)

    def _on_window_closed(self) -> None:
        self.closed = True
        self.save()

    # ── napari's layer list ────────────────────────────────────────────────────────────────────────────────────

    def _on_layer_inserted(self, event) -> None:
        layer = event.value
        if layer in (self.source_layer, self.filtered_layer, self.measurement_layer):
            return
        if isinstance(layer, napari.layers.Points) and layer.metadata.get("annotation_kind") == PARTICLE_KIND:
            self.particles_dock.bind_layer(layer)
            self.particles_dock.refresh_choices()
            return
        # A volume opened with napari's File > Open joins the file list.
        source_path = getattr(getattr(layer, "source", None), "path", None)
        if not source_path:
            return
        path = Path(source_path).expanduser().resolve()
        if path.suffix.lower() not in MRC_SUFFIXES:
            return
        if path not in self.session.paths:
            self.register(path)

    def _on_layer_removed(self, _event) -> None:
        self.particles_dock.drop_stale()
        if self.measurement_layer is not None and self.measurement_layer not in self.viewer.layers:
            self.measurement_layer = None
        self.particles_dock.refresh_choices()


def build_viewer(
    profile: Optional[RuntimeProfile] = None,
    *,
    title: Optional[str] = None,
    outputs: Optional[Outputs] = None,
    startup_progress: Optional[StartupProgress] = None,
    on_ready: Optional[Callable[[napari.Viewer], None]] = None,
) -> Controller:
    """A napari window with EisGreis's docks."""
    profile = profile or RuntimeProfile()
    viewer = napari.Viewer(title=title or f"EisGreis ({profile.name})")
    if on_ready is not None:
        on_ready(viewer)
    return Controller(viewer, startup_progress=startup_progress, profile=profile, outputs=outputs)


def run_viewer(
    paths: list[Path],
    profile: Optional[RuntimeProfile] = None,
    on_ready: Optional[Callable[[napari.Viewer], None]] = None,
) -> None:
    """EisGreis with `paths` in its file list, the first of them open, until the window closes."""
    profile = profile or RuntimeProfile()
    print(f"[eisgreis] {profile.describe()}", flush=True)
    startup_progress = StartupProgress(len(paths)) if paths else None
    if startup_progress is not None:
        startup_progress.update(0.0, f"Matched {len(paths)} tomograms")
    controller = build_viewer(profile, startup_progress=startup_progress, on_ready=on_ready)
    for index, path in enumerate(paths, start=1):
        controller.register(path)
        if startup_progress is not None:
            fraction = 0.05 + 0.20 * (index / max(len(paths), 1))
            startup_progress.update(fraction, f"Registered {index}/{len(paths)} tomograms")
    if paths:
        controller.open(Path(paths[0]))
    napari.run()
