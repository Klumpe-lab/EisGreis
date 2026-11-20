"""What a driver may use: the one module of EisGreis that code outside it imports.

Everything else in the package is internal and changes without notice; this module changes only with a release note,
and a driver that keeps to it keeps working across viewer refactors.

    from eisgreis import api

    viewer = api.build()                       # a napari window with EisGreis's docks
    viewer.set_outputs("/data/project/notes")  # comments CSV, particle_stars/, tomogram_pngs/
    viewer.open("/data/project/tomo.mrc", title="tomo")
    viewer.reference.show_picks("ribosome", [[64, 100, 120]], box_angstrom=300, color="#ff1744")
    api.run_event_loop()

`TomogramViewer` holds one tomogram on screen and a file list; its comments and particle points are saved per
tomogram (see ``eisgreis.session``). Coordinates are Z, Y, X; lengths are Å.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from eisgreis import __version__ as version
from eisgreis.runtime import RuntimeProfile, make_profile

__all__ = [
    "RuntimeProfile",
    "TomogramViewer",
    "build",
    "make_profile",
    "napari_version",
    "run",
    "run_event_loop",
    "version",
]


def napari_version() -> str:
    """The napari this EisGreis runs on, read without importing napari."""
    from importlib.metadata import version as dist_version

    return dist_version("napari")


class ReferenceAPI:
    """Reference layers of a tomogram: picks, segmentations and other volumes, drawn while it is on screen and never
    saved. Every method acts on the tomogram on screen unless `tomogram` names another."""

    def __init__(self, layers) -> None:
        self._layers = layers

    def show_picks(
        self,
        name: str,
        coords_zyx: Sequence[Sequence[float]],
        box_angstrom: Optional[float] = None,
        color: str = "",
        *,
        source: str = "",
        pixel_size: Optional[float] = None,
        tomogram: Optional[Path] = None,
    ) -> None:
        """Picks at voxel coordinates (Z, Y, X) of the tomogram, whose voxels are `pixel_size` Å (default: its
        header's), as discs of `box_angstrom` in `color` ('#rrggbb'). Raises ValueError for coordinates that are no
        list of triples."""
        self._layers.show_picks(
            name, coords_zyx, box_angstrom, color, source=source, pixel_size=pixel_size, tomogram=tomogram
        )

    def show_segmentation(
        self,
        path: Path,
        level: float = 0.5,
        color: str = "",
        voxel_size: Optional[float] = None,
        *,
        name: Optional[str] = None,
        full_scale: float = 1.0,
        visible: bool = True,
        pixel_size: Optional[float] = None,
        tomogram: Optional[Path] = None,
    ) -> None:
        """A segmentation volume, drawn where it passes `level`, in `color`. `voxel_size` is its own voxel size in Å
        (default: the tomogram's); it shares the tomogram's corner. Raises OSError or ValueError for a file that is
        no 3-D volume."""
        self._layers.show_segmentation(
            path,
            level,
            color,
            voxel_size,
            name=name,
            full_scale=full_scale,
            visible=visible,
            pixel_size=pixel_size,
            tomogram=tomogram,
        )

    def add_volume(self, name: str, path: Path, kind: str = "", *, tomogram: Optional[Path] = None) -> None:
        """Another picture of the tomogram on its grid (a denoised one), offered by the Reference dock's *Show*."""
        self._layers.add_volume(name, path, kind, tomogram=tomogram)

    def clear(self, *, tomogram: Optional[Path] = None) -> None:
        self._layers.clear(tomogram)

    def summary(self, *, tomogram: Optional[Path] = None) -> dict[str, Any]:
        """What is shown: ``picks``, ``segmentations`` and ``volumes`` by name, and ``not_shown``, what could not be
        drawn and why."""
        return self._layers.summary(tomogram)


class View3DAPI:
    """3-D on any renderer (binned copies of the big volumes, segmentations as meshes)."""

    def __init__(self, view3d) -> None:
        self._view3d = view3d

    def set_seg_mode(self, mode: str) -> None:
        """How 3-D draws segmentations: "surfaces" (smooth meshes, one per feature) or "merged" (one labels
        volume, the fastest)."""
        self._view3d.set_seg_mode(mode)

    @property
    def seg_mode(self) -> str:
        return self._view3d.seg_mode


class TomogramViewer:
    """A napari window with EisGreis's docks around one tomogram on screen."""

    def __init__(self, controller) -> None:
        self._c = controller
        self.reference = ReferenceAPI(controller.reference)
        self.view3d = View3DAPI(controller.view3d)
        self._renderer: Optional[str] = None

    @property
    def napari(self):
        """The ``napari.Viewer``."""
        return self._c.viewer

    @property
    def current(self) -> Optional[Path]:
        """The tomogram on screen."""
        return self._c.session.current

    @property
    def tomograms(self) -> list[Path]:
        """The file list, sorted."""
        return list(self._c.session.paths)

    @property
    def outputs(self) -> dict[str, Path]:
        """Where saves go now: ``comments_csv``, ``star_dir``, ``png_dir``."""
        o = self._c.session.outputs
        return {"comments_csv": o.comments_csv, "star_dir": o.star_dir, "png_dir": o.png_dir}

    @property
    def shape(self) -> Optional[tuple[int, ...]]:
        """The grid of the volume on screen (Z, Y, X)."""
        data = self._c.original_data
        return tuple(int(n) for n in data.shape) if data is not None else None

    @property
    def pixel_size(self) -> float:
        """Å per voxel of the tomogram on screen: its header's, or what the filter dock set."""
        return float(self._c.voxel_size_angstrom)

    @property
    def renderer(self) -> str:
        """The OpenGL renderer the window draws with (asked once)."""
        if self._renderer is None:
            from eisgreis._napari import gl_renderer

            self._renderer = gl_renderer(self._c.viewer)
        return self._renderer

    def register(self, path: Path) -> Path:
        """Add a tomogram to the file list without opening it; its resolved path."""
        return self._c.register(Path(path))

    def open(self, path: Path, title: Optional[str] = None) -> None:
        """Open a tomogram: its middle slice, the whole volume in view, low-passed; `title` names the window.
        Raises OSError or ValueError for a file that cannot be read as a volume."""
        self._c.open(Path(path), title=title)

    def forget(self, path: Path) -> None:
        """Drop a tomogram from the file list with its comment, particles and reference layers."""
        self._c.forget(Path(path))

    def set_outputs(self, folder: Path) -> None:
        """Save into `folder` from now on (``tomogram_comments.csv``, ``particle_stars/``, ``tomogram_pngs/``). What
        the viewer holds is saved into the old folders first."""
        self._c.set_outputs(Path(folder))

    def save(self) -> None:
        """Write the comments CSV and every particle STAR into the current folders."""
        self._c.save()

    def reload_annotations(self) -> None:
        """Read the comments and particle STARs back from the current folders."""
        self._c.reload_annotations()

    def maximize(self) -> None:
        from eisgreis._napari import qt_window

        window = qt_window(self._c.viewer)
        if window is not None:
            window.showMaximized()


def build(
    profile: Optional[RuntimeProfile] = None, *, title: Optional[str] = None, outputs: Optional[Path] = None
) -> TomogramViewer:
    """A napari window with EisGreis's docks; saves go into `outputs` (default: the working directory)."""
    from eisgreis.app import build_viewer
    from eisgreis.session import Outputs

    controller = build_viewer(profile, title=title, outputs=Outputs.under(outputs) if outputs is not None else None)
    return TomogramViewer(controller)


def run(
    paths: Sequence[Path], profile: Optional[RuntimeProfile] = None, on_ready: Optional[Callable[[Any], None]] = None
) -> None:
    """EisGreis with `paths` in its file list and the first open, until its window closes. `on_ready` gets the
    ``napari.Viewer`` as soon as the window exists, before the docks."""
    from eisgreis.app import run_viewer

    run_viewer([Path(p) for p in paths], profile, on_ready=on_ready)


def run_event_loop() -> None:
    """Run Qt's event loop until the last window closes (napari.run)."""
    import napari

    napari.run()
