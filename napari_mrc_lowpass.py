import argparse
from pathlib import Path
from typing import Optional

import mrcfile
import napari
import numpy as np
from magicgui import magicgui
from scipy.fft import fftn, fftshift, ifftn, ifftshift


class LowPassController:
    FILTER_CHOICES = ("hard", "gaussian", "butterworth", "cosine")

    def __init__(self, viewer: napari.Viewer) -> None:
        self.viewer = viewer
        self.source_layer: Optional[napari.layers.Image] = None
        self.filtered_layer: Optional[napari.layers.Image] = None
        self.original_data: Optional[np.ndarray] = None
        self.radius_grid: Optional[np.ndarray] = None
        self.frequency_grid: Optional[np.ndarray] = None
        self.voxel_size_angstrom: float = 1.0
        self.min_resolution_angstrom: float = 2.0
        self.max_resolution_angstrom: float = 50.0
        self.current_path: Optional[Path] = None

        self.controls = self._build_controls()
        self.viewer.window.add_dock_widget(self.controls, area="right")

    def _build_controls(self):
        @magicgui(
            auto_call=True,
            call_button=False,
            path={"label": "MRC file", "mode": "r", "filter": "*.mrc"},
            cutoff={"label": "Low-pass (A)", "widget_type": "FloatSlider", "min": 2.0, "max": 50.0, "step": 0.5},
            filter_type={"label": "Filter", "choices": self.FILTER_CHOICES},
            show_original={"label": "Show original"},
        )
        def controls(
            path: Optional[Path] = None,
            cutoff: float = 10.0,
            filter_type: str = "gaussian",
            show_original: bool = False,
        ) -> None:
            if path and path != self.current_path:
                self.load_mrc(path)
            if self.source_layer is not None:
                self.source_layer.visible = show_original
            if self.original_data is not None and self.frequency_grid is not None:
                self.apply_lowpass(cutoff, filter_type)

        controls.native.setWindowTitle("MRC Low-pass")
        return controls

    def load_mrc(self, path: Path) -> None:
        with mrcfile.open(path, permissive=True) as handle:
            data = np.asarray(handle.data, dtype=np.float32)
            voxel_size = np.asarray(tuple(handle.voxel_size.tolist()), dtype=np.float32)

        self.current_path = path
        self.original_data = data
        self.voxel_size_angstrom = self._extract_voxel_size(voxel_size)
        self.frequency_grid = self._build_frequency_grid(data.shape, self.voxel_size_angstrom)
        self.min_resolution_angstrom, self.max_resolution_angstrom = self._resolution_bounds(
            data.shape, self.voxel_size_angstrom
        )
        self.controls.cutoff.min = self.min_resolution_angstrom
        self.controls.cutoff.max = self.max_resolution_angstrom
        self.controls.cutoff.value = min(
            max(10.0, self.min_resolution_angstrom),
            self.max_resolution_angstrom,
        )

        if self.source_layer is None:
            self.source_layer = self.viewer.add_image(
                data,
                name=f"{path.name} (original)",
                colormap="gray",
                visible=False,
            )
        else:
            self.source_layer.data = data
            self.source_layer.name = f"{path.name} (original)"
            self.source_layer.colormap = "gray"
            self.source_layer.visible = False

        filtered = self._lowpass(data, self.controls.cutoff.value, self.controls.filter_type.value)
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

        self._auto_contrast(self.filtered_layer.data, self.filtered_layer)

        self.viewer.reset_view()

    def _extract_voxel_size(self, voxel_size: np.ndarray) -> float:
        valid = voxel_size[np.isfinite(voxel_size) & (voxel_size > 0)]
        if valid.size == 0:
            return 1.0
        return float(valid.mean())

    def _build_frequency_grid(self, shape: tuple[int, ...], voxel_size_angstrom: float) -> np.ndarray:
        axes = np.meshgrid(
            *[fftshift(np.fft.fftfreq(size, d=voxel_size_angstrom)) for size in shape],
            indexing="ij",
        )
        return np.sqrt(sum(axis**2 for axis in axes)).astype(np.float32, copy=False)

    def _resolution_bounds(self, shape: tuple[int, ...], voxel_size_angstrom: float) -> tuple[float, float]:
        nyquist_resolution = 2.0 * voxel_size_angstrom
        max_dimension = max(shape) if shape else 1
        lowest_frequency = 1.0 / (max_dimension * voxel_size_angstrom)
        lowest_resolution = 1.0 / lowest_frequency
        return float(max(nyquist_resolution, 0.1)), float(max(lowest_resolution, nyquist_resolution + 0.5))

    def _filter_mask(self, cutoff_frequency: float, filter_type: str) -> np.ndarray:
        ratio = self.frequency_grid / max(cutoff_frequency, 1e-6)

        if filter_type == "hard":
            return (self.frequency_grid <= cutoff_frequency).astype(np.float32, copy=False)

        if filter_type == "gaussian":
            return np.exp(-(ratio**2) / 2.0).astype(np.float32, copy=False)

        if filter_type == "butterworth":
            order = 2
            return (1.0 / (1.0 + ratio ** (2 * order))).astype(np.float32, copy=False)

        if filter_type == "cosine":
            start = 0.8 * cutoff_frequency
            stop = cutoff_frequency
            mask = np.ones_like(self.frequency_grid, dtype=np.float32)
            outer = self.frequency_grid >= stop
            transition = (self.frequency_grid > start) & (self.frequency_grid < stop)
            mask[outer] = 0.0
            transition_ratio = (self.frequency_grid[transition] - start) / max(stop - start, 1e-6)
            mask[transition] = 0.5 * (1.0 + np.cos(np.pi * transition_ratio))
            return mask

        return np.exp(-(ratio**2) / 2.0).astype(np.float32, copy=False)

    def _lowpass(self, data: np.ndarray, resolution_angstrom: float, filter_type: str) -> np.ndarray:
        cutoff_frequency = 1.0 / max(resolution_angstrom, 1e-3)
        spectrum = fftshift(fftn(data))
        mask = self._filter_mask(cutoff_frequency, filter_type)
        filtered = ifftn(ifftshift(spectrum * mask))
        return np.asarray(np.real(filtered), dtype=np.float32)

    def apply_lowpass(self, resolution_angstrom: float, filter_type: str) -> None:
        if self.original_data is None or self.filtered_layer is None:
            return
        filtered = self._lowpass(self.original_data, resolution_angstrom, filter_type)
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
    parser.add_argument("path", nargs="?", type=Path, help="Optional .mrc file to open on launch.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    viewer = napari.Viewer(title="Napari MRC Low-pass Viewer")
    controller = LowPassController(viewer)
    if args.path is not None:
        controller.controls.path.value = args.path
    napari.run()


if __name__ == "__main__":
    main()
