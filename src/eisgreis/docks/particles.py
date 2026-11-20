"""The Particle Annotations dock and the particle layers: named napari points layers, each holding its points for every
tomogram, saved as one STAR per tomogram and name (see ``session.py``).

A particle layer shows the points of the tomogram on screen; switching tomograms keeps its points in the session and
puts the next tomogram's in. A point draws as the cross-section of a sphere of its box through the slice on screen
(napari's "rescale_spherical" projection), and the Z slice is made at least as thick as the box, since napari draws
points only inside the slice's thickness.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import napari
import numpy as np
from magicgui.widgets import ComboBox, Container, FileEdit, Label, LineEdit, PushButton
from qtpy.QtWidgets import QSizePolicy

from eisgreis.docks import NO_PARTICLE_LAYERS, float_validator

if TYPE_CHECKING:
    from eisgreis.app import Controller

DEFAULT_BOX_ANGSTROM = 200.0
KIND = "particle"  # layer.metadata["annotation_kind"] of a particle layer


class ParticlesDock:
    def __init__(self, controller: "Controller") -> None:
        self.c = controller
        self._updating = False
        self.layers: dict[str, napari.layers.Points] = {}

        self.particle_name = LineEdit(label="Particle name", value="ribosome")
        self.particle_layers = ComboBox(
            label="Particle layer", choices=self._layer_choice_names, value=NO_PARTICLE_LAYERS
        )
        self.box_size = LineEdit(label="Box size (Å)", value=str(int(DEFAULT_BOX_ANGSTROM)))
        self.star_dir = LineEdit(label="STAR output dir", value=str(controller.session.outputs.star_dir))
        self.load_star_dir = FileEdit(label="Load STAR dir", mode="d", value=None)
        self.create_layer_button = PushButton(label="Create particle layer")
        self.load_layers = PushButton(label="Load particle STARs")
        self.save_layer = PushButton(label="Save selected STAR")
        self.save_all_button = PushButton(label="Save all STARs")
        self.status = Label(value="STAR output ready")

        self.box_size.native.setValidator(float_validator())

        self.create_layer_button.changed.connect(self._on_create_layer)
        self.load_layers.changed.connect(self._on_load_stars)
        self.save_layer.changed.connect(self._on_save_selected)
        self.save_all_button.changed.connect(self._on_save_all)
        self.particle_layers.changed.connect(self._on_layer_selected)
        self.box_size.native.editingFinished.connect(self._on_box_size_edit_finished)
        self.star_dir.changed.connect(self._on_star_dir_changed)

        self.container = Container(
            widgets=[
                self.particle_name,
                self.particle_layers,
                self.box_size,
                self.star_dir,
                self.load_star_dir,
                self.create_layer_button,
                self.load_layers,
                self.save_layer,
                self.save_all_button,
                self.status,
            ],
            labels=True,
        )
        self.container.native.setWindowTitle("Particle Annotations")
        native = self.status.native
        native.setWordWrap(False)
        native.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        native.setMaximumHeight(native.fontMetrics().height() + 8)
        native.setToolTip(str(self.status.value))

    # ── status and fields ──────────────────────────────────────────────────────────────────────────────────────

    def set_status(self, message: str, detail: Optional[str] = None) -> None:
        self.status.value = message
        self.status.native.setToolTip(detail or message)

    def _star_dir_detail(self) -> str:
        return f"STAR dir: {self.c.session.outputs.star_dir}"

    def show_outputs(self) -> None:
        """The STAR folder the session saves to, in the field."""
        self._updating = True
        try:
            self.star_dir.value = str(self.c.session.outputs.star_dir)
        finally:
            self._updating = False

    def _on_star_dir_changed(self, value: str) -> None:
        if not self._updating:
            self.c.session.set_star_dir(value)

    # ── the layers ─────────────────────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def name_of(layer) -> str:
        return layer.metadata.get("particle_name", layer.name)

    def layers_sorted(self) -> list[napari.layers.Points]:
        return [self.layers[name] for name in sorted(self.layers)]

    def _layer_choice_names(self, _widget=None) -> list[str]:
        if not getattr(self, "layers", None):
            return [NO_PARTICLE_LAYERS]
        return [self.name_of(layer) for layer in self.layers_sorted()] or [NO_PARTICLE_LAYERS]

    def selected_layer(self) -> Optional[napari.layers.Points]:
        selected_name = self.particle_layers.value
        if selected_name in ("", NO_PARTICLE_LAYERS):
            return None
        return self.layers.get(selected_name)

    def selected_particle_name(self) -> Optional[str]:
        layer = self.selected_layer()
        return layer.metadata.get("particle_name") if layer is not None else None

    def display_size_pixels(self, particle_name: str) -> float:
        box_size_angstrom = self.c.session.box_sizes.get(particle_name, DEFAULT_BOX_ANGSTROM)
        pixel_size = self.c.voxel_size_angstrom if self.c.voxel_size_angstrom > 0 else 1.0
        return max(box_size_angstrom / pixel_size, 2.0)

    def point_dims(self) -> int:
        source = self.c.source_layer
        return int(source.ndim) if source is not None and int(source.ndim) >= 2 else 2

    def apply_style(self, layer: napari.layers.Points) -> None:
        particle_name = self.name_of(layer)
        layer.size = self.display_size_pixels(particle_name)
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
        layer.projection_mode = "rescale_spherical"
        self.c.ensure_z_thickness(self.c.session.box_sizes.get(particle_name, DEFAULT_BOX_ANGSTROM))

    def restyle_all(self) -> None:
        for layer in self.layers.values():
            self.apply_style(layer)

    def bind_layer(self, layer: napari.layers.Points) -> None:
        if layer.metadata.get("_particle_events_bound"):
            return
        layer.events.name.connect(lambda _event: self.refresh_choices())
        layer.metadata["_particle_events_bound"] = True

    def store_current(self) -> None:
        """The points on screen into the session, for the tomogram on screen."""
        current = self.c.session.current
        if current is None:
            return
        for particle_name, layer in self.layers.items():
            self.c.session.store_particles(particle_name, current, layer.data)

    def restore_current(self) -> None:
        """The session's points of the tomogram on screen into the layers."""
        current = self.c.session.current
        if current is None:
            return
        dims = self.point_dims()
        empty = np.empty((0, dims), dtype=float)
        for particle_name, layer in self.layers.items():
            coords = self.c.session.particles_of(particle_name, current)
            # napari keeps the selection through a data swap: a point selected on the last tomogram would index
            # past the end of this one's (IndexError in the style's current_border_width).
            layer.selected_data = set()
            layer.data = (
                coords.copy() if coords is not None and coords.ndim == 2 and coords.shape[1] == dims else empty.copy()
            )
            layer.visible = True
            layer.metadata["tomogram_path"] = str(current)
            self.apply_style(layer)

    def sync_visibility(self) -> None:
        visible = self.c.session.current is not None
        for layer in self.layers.values():
            layer.visible = visible

    def drop_stale(self) -> None:
        """Forget the layers the user deleted, and their points."""
        stale = [name for name, layer in self.layers.items() if layer not in self.c.viewer.layers]
        for name in stale:
            self.layers.pop(name, None)
            self.c.session.particles.pop(name, None)

    def refresh_choices(self) -> None:
        layers = self.layers_sorted()
        current = self.c.session.current
        self._updating = True
        try:
            selected = self.particle_layers.value
            self.particle_layers.reset_choices()
            if not layers:
                self.particle_layers.value = NO_PARTICLE_LAYERS
                message = "No particle layers" if current is None else f"No particle layers for {current.name}"
                self.set_status(message, self._star_dir_detail())
                return
            names = [self.name_of(layer) for layer in layers]
            self.particle_layers.value = selected if selected in names else names[0]
            selected_name = self.particle_layers.value
            if selected_name in self.c.session.box_sizes:
                self.box_size.value = str(self.c.session.box_sizes[selected_name])
            where = current.name if current is not None else "no tomogram"
            self.set_status(f"{len(names)} particle layer(s) for {where}", self._star_dir_detail())
        finally:
            self._updating = False

    def create_layer(self, particle_name: str, update_status: bool = True) -> napari.layers.Points:
        """The particle layer `particle_name`, made when it does not exist; it becomes the active layer, adding
        points."""
        session = self.c.session
        if session.current is None or self.c.source_layer is None:
            raise RuntimeError("No active tomogram loaded")
        viewer = self.c.viewer
        existing = self.layers.get(particle_name)
        if existing is not None:
            self.particle_name.value = particle_name
            self.particle_layers.value = particle_name
            if particle_name in session.box_sizes:
                self.box_size.value = str(session.box_sizes[particle_name])
            viewer.layers.selection.active = existing
            existing.mode = "add"
            if update_status:
                self.set_status(
                    f"'{particle_name}' exists: selected, click to add points (new name for a new layer)",
                    self._star_dir_detail(),
                )
            return existing

        dims = self.point_dims()
        try:
            box_size_angstrom = float(self.box_size.value)
        except ValueError:  # an empty or half-typed field: the default box
            box_size_angstrom = DEFAULT_BOX_ANGSTROM
        if box_size_angstrom <= 0:
            box_size_angstrom = DEFAULT_BOX_ANGSTROM
        session.box_sizes[particle_name] = box_size_angstrom
        layer = viewer.add_points(
            data=np.empty((0, dims), dtype=float),
            name=particle_name,
            scale=self.c.spatial_scale(dims),
            units=("angstrom",) * dims,
            projection_mode="rescale_spherical",
            metadata={
                "annotation_kind": KIND,
                "particle_name": particle_name,
                "tomogram_path": str(session.current),
                "box_size_angstrom": box_size_angstrom,
            },
        )
        self.layers[particle_name] = layer
        session.particles.setdefault(particle_name, {})
        self.apply_style(layer)
        layer.mode = "add"
        layer.visible = True
        self.bind_layer(layer)
        self.c.show_current_file()
        self.sync_visibility()
        self.refresh_choices()
        self.particle_layers.value = particle_name
        self.particle_name.value = particle_name
        self.box_size.value = str(box_size_angstrom)
        viewer.layers.selection.active = layer
        if update_status:
            self.set_status(f"Created particle layer '{particle_name}'", self._star_dir_detail())
        return layer

    def load_stars(self, star_dir: Path) -> int:
        """Read the STARs of the registered tomograms from `star_dir` into particle layers; the number read."""
        star_dir = Path(star_dir).expanduser().resolve()
        if not star_dir.exists():
            self.set_status("STAR folder not found", str(star_dir))
            return 0
        names = self.c.session.load_particle_stars(star_dir)
        for particle_name in names:
            self.create_layer(particle_name, update_status=False)
        self.restore_current()
        self.refresh_choices()
        self.set_status(f"Loaded {len(names)} STAR file(s)", f"STAR dir: {star_dir}")
        return len(names)

    def save_all(self) -> int:
        """Write every tomogram's points of every layer; the number of STAR files written."""
        self.store_current()
        saved = self.c.session.save_all_particles(self.c.header_voxel_size_angstrom)
        if not self.c.closed:
            self.set_status(f"Saved {saved} particle STAR file(s)", self._star_dir_detail())
        return saved

    # ── handlers ───────────────────────────────────────────────────────────────────────────────────────────────

    def _on_create_layer(self, _value: object) -> None:
        if self.c.session.current is None or self.c.source_layer is None:
            return
        self.create_layer(self.particle_name.value.strip() or "particle", update_status=True)

    def _on_load_stars(self, _value: object) -> None:
        raw_dir = self.load_star_dir.value
        if raw_dir in (None, ""):
            raw_dir = str(self.c.session.outputs.star_dir)
        self.load_stars(Path(raw_dir))

    def _on_save_selected(self, _value: object) -> None:
        layer = self.selected_layer()
        current = self.c.session.current
        if layer is None or current is None:
            return
        self.store_current()
        particle_name = self.name_of(layer)
        star = self.c.session.save_particles(
            particle_name, current, self.c.session.pixel_size_of(current, self.c.voxel_size_angstrom)
        )
        self.set_status(f"Saved '{particle_name}'", str(star))

    def _on_save_all(self, _value: object) -> None:
        self.save_all()

    def _on_layer_selected(self, value: str) -> None:
        if self._updating or value in ("", NO_PARTICLE_LAYERS):
            return
        layer = self.selected_layer()
        if layer is None:
            return
        particle_name = self.name_of(layer)
        if particle_name in self.c.session.box_sizes:
            self._updating = True
            try:
                self.box_size.value = str(self.c.session.box_sizes[particle_name])
                self.particle_name.value = particle_name
            finally:
                self._updating = False
        self.c.viewer.layers.selection.active = layer

    def _on_box_size_edit_finished(self) -> None:
        if self._updating:
            return
        try:
            box_size = float(self.box_size.value)
        except ValueError:  # an empty or half-typed field
            return
        if box_size <= 0:
            return
        layer = self.selected_layer()
        if layer is None:
            self.c.session.box_sizes[self.particle_name.value.strip() or "particle"] = box_size
            return
        particle_name = self.name_of(layer)
        self.c.session.box_sizes[particle_name] = box_size
        layer.metadata["box_size_angstrom"] = box_size
        self.apply_style(layer)
