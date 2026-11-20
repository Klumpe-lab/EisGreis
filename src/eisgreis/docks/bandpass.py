"""A crYOLO-style Gaussian low-pass and high-pass on an open image layer, as a new layer; optionally live on the 2-D
slice on screen. ``eisgreis-bandpass`` opens it in a viewer of its own."""

import os
from pathlib import Path
from typing import Optional

import napari
import napari.layers
import numpy as np
from napari.utils.notifications import show_info
from qtpy.QtCore import QRegularExpression, Slot
from qtpy.QtGui import QRegularExpressionValidator
from qtpy.QtWidgets import QCheckBox, QComboBox, QFormLayout, QLineEdit, QPushButton, QWidget

from eisgreis.filters import gaussian_bandpass_filter
from eisgreis.io.mrc import read_voxel_size_from_path


class CryoloBandpassWidget(QWidget):
    def __init__(self, napari_viewer: "napari.Viewer"):
        super().__init__()
        self.napari_viewer = napari_viewer
        self._layer = QComboBox(self)
        self._lp_filter_resolution = QLineEdit("20", self)
        self._lp_filter_resolution.setToolTip(
            "Low-pass filter value in angstrom. Set to 0 to disable low-pass filtering."
        )
        self._hp_filter_resolution = QLineEdit("0", self)
        self._hp_filter_resolution.setToolTip(
            "High-pass filter value in angstrom. Set to 0 to disable high-pass filtering."
        )
        self._pixel_size = QLineEdit("-1", self)
        self._pixel_size.setToolTip("Pixel size of the image in A/px.")
        self._filter_2d = QCheckBox(self)
        self._filter_2d.setToolTip("If the layer is 3D, filter slice-by-slice in z rather than the full volume.")
        self._show_mask = QCheckBox(self)
        self._show_mask.setToolTip("Visualize the Gaussian Fourier mask used for filtering.")
        self._run_btn = QPushButton("Run", self)

        float_validator = QRegularExpressionValidator(QRegularExpression(r"[0-9]*\.?[0-9]*"))
        self._lp_filter_resolution.setValidator(float_validator)
        self._hp_filter_resolution.setValidator(float_validator)
        self._pixel_size.setValidator(float_validator)

        layout = QFormLayout()
        layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.setLayout(layout)
        self.layout().addRow("Image layer:", self._layer)
        self.layout().addRow("LP resolution [A]:", self._lp_filter_resolution)
        self.layout().addRow("HP resolution [A]:", self._hp_filter_resolution)
        self.layout().addRow("Pixel size [A/px]:", self._pixel_size)
        self.layout().addRow("Live filter 2D slices", self._filter_2d)
        self.layout().addRow("Show mask", self._show_mask)
        self.layout().addRow(self._run_btn)

        self._layer.currentTextChanged.connect(self._update_pixel_size)
        self._run_btn.clicked.connect(self._run)
        self.napari_viewer.layers.events.inserted.connect(self._update_combo)
        self.napari_viewer.layers.events.removed.connect(self._update_combo)

        self._update_combo()

    @Slot()
    def _run(self):
        for attr in ["pixel_size", "lp_filter_resolution", "hp_filter_resolution"]:
            try:
                if getattr(self, attr) < 0:
                    show_info(f"Error: {attr} needs to be larger than 0")
                    return None
            except ValueError:
                show_info(f"Error: {attr} needs to be provided")
                return None

        if not self._layer.currentText():
            show_info("No layer selected")
            return None

        kwargs = {
            "lp_filter_resolution_ang": self.lp_filter_resolution,
            "hp_filter_resolution_ang": self.hp_filter_resolution,
            "pixel_size": self.pixel_size,
        }
        filtered_image, mask = self.handle_filter(self.layer, self.filter_2d, **kwargs)
        if filtered_image is None:
            return None

        if self.show_mask:
            self.napari_viewer.add_image(
                mask,
                name=(
                    f"GAUSS MASK LP {int(self.lp_filter_resolution)} "
                    f"HP {int(self.hp_filter_resolution)} AP {self.pixel_size} - {self.layer.name}"
                ),
                colormap="gray",
            )

        image = napari.layers.Image(
            filtered_image,
            name=(
                f"GAUSS LP {int(self.lp_filter_resolution)} HP {int(self.hp_filter_resolution)} "
                f"AP {self.pixel_size} LIVE {self.filter_2d} - {self.layer.name}"
            ),
            metadata=self.layer.metadata.copy(),
            colormap="gray",
        )
        image.contrast_limits_range = [float(filtered_image.min()), float(filtered_image.max())]
        image.contrast_limits = [float(np.percentile(filtered_image, 1)), float(np.percentile(filtered_image, 99))]

        self.napari_viewer.layers.insert(self.napari_viewer.layers.index(self.layer) + 1, image)

        if self.filter_2d:
            connect_dims = self.napari_viewer.dims.events.current_step.connect(
                lambda *x, new_layer=image, old_layer=self.layer, filter_kwargs=kwargs: self._filter_layer(
                    new_layer=new_layer, old_layer=old_layer, **filter_kwargs
                )
            )
            connect_visible = image.events.visible.connect(
                lambda *x, new_layer=image, old_layer=self.layer, filter_kwargs=kwargs: self._filter_layer(
                    new_layer=new_layer, old_layer=old_layer, **filter_kwargs
                )
            )
            image.connect_visible = connect_visible
            image.connect_dims = connect_dims

    def _filter_layer(self, new_layer, old_layer, **filter_kwargs):
        if new_layer not in self.napari_viewer.layers or old_layer not in self.napari_viewer.layers:
            new_layer.events.disconnect(new_layer.connect_visible)
            self.napari_viewer.dims.events.current_step.disconnect(new_layer.connect_dims)
            return

        if new_layer.visible:
            filtered_image, _ = self.handle_filter(old_layer, True, **filter_kwargs)
            new_layer.data[...] = filtered_image
            new_layer.contrast_limits_range = [float(filtered_image.min()), float(filtered_image.max())]
            new_layer.contrast_limits = [
                float(np.percentile(filtered_image, 1)),
                float(np.percentile(filtered_image, 99)),
            ]
            with new_layer.events.visible.blocker():
                new_layer.visible = True

    def handle_filter(self, layer, filter_2d, **kwargs):
        if filter_2d and len(layer.data.shape) >= 3:
            slice_axis = self.napari_viewer.dims.order[0]
            slice_index = self.napari_viewer.dims.current_step[slice_axis]
            slc = [slice(None)] * len(layer.data.shape)
            slc[slice_axis] = slice_index
            data = layer.data[tuple(slc)]
        else:
            data = layer.data

        result = gaussian_bandpass_filter(data, log=show_info, **kwargs)
        if result is None:
            return None, None

        filtered_image, mask = result
        std = np.std(filtered_image)
        if std > 0:
            filtered_image = (filtered_image - np.mean(filtered_image)) / std

        return filtered_image, mask

    @Slot(str)
    def _update_pixel_size(self, layer_name):
        try:
            layer = self.napari_viewer.layers[layer_name]
        except KeyError:
            return

        pixel_spacing = layer.metadata.get("pixel_spacing")
        if pixel_spacing is None:
            pixel_spacing = self._pixel_size_from_source(layer)

        self.pixel_size = pixel_spacing if pixel_spacing is not None else -1

        is_3d = isinstance(layer, napari.layers.Image) and len(layer.data.shape) >= 3
        self.filter_2d = False
        self._filter_2d.setEnabled(is_3d)

    def _pixel_size_from_source(self, layer) -> Optional[float]:
        source_path = getattr(getattr(layer, "source", None), "path", None)
        if not source_path:
            return None
        path = Path(source_path)
        return read_voxel_size_from_path(path)

    @Slot(object)
    def _update_combo(self, *_):
        layer_names = sorted(
            entry.name for entry in self.napari_viewer.layers if isinstance(entry, napari.layers.Image)
        )
        current_text = self._layer.currentText()

        self._layer.currentTextChanged.disconnect(self._update_pixel_size)
        self._layer.clear()
        self._layer.addItems(layer_names)
        self._layer.setCurrentText(current_text)
        self._layer.currentTextChanged.connect(self._update_pixel_size)
        self._layer.currentTextChanged.emit(self._layer.currentText())

    @property
    def layer(self):
        return self.napari_viewer.layers[self._layer.currentText()]

    @property
    def show_mask(self):
        return self._show_mask.isChecked()

    @property
    def lp_filter_resolution(self):
        return float(self._lp_filter_resolution.text())

    @property
    def hp_filter_resolution(self):
        return float(self._hp_filter_resolution.text())

    @property
    def pixel_size(self):
        return float(self._pixel_size.text())

    @pixel_size.setter
    def pixel_size(self, value):
        self._pixel_size.setText(str(value))

    @property
    def filter_2d(self):
        return self._filter_2d.isChecked()

    @filter_2d.setter
    def filter_2d(self, value):
        self._filter_2d.setChecked(value)


def main() -> None:
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    viewer = napari.Viewer(title="Cryolo-style Gaussian Filter")
    widget = CryoloBandpassWidget(viewer)
    viewer.window.add_dock_widget(widget, area="right")
    napari.run()


if __name__ == "__main__":
    main()
