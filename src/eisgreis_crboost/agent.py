"""The viewer driven by crboost: ``open.json`` → ``eisgreis.api`` → ``opened.json``.

The agent checks the session folder for a new request every second. A request with a higher ``seq`` opens its first
volume (its other volumes join the file list) and shows its bundle, afresh on every request, so asking again for the
same tomogram picks up crboost's newer results. What the window then shows goes back into ``opened.json``.

The viewer holds one project at a time and saves into the request's ``outputs``, the project's
``Curation/napari/``. Before it opens another tomogram, what it holds is saved (comments CSV, every particle STAR),
so the STARs then read back from the folder are what was on screen. Moving to another project saves into the old
folder, forgets the old project's tomograms, and points the saves at the new one. A request without ``outputs`` (a
volume outside every project) saves into the session folder's ``outputs/``, never the working directory, and says so.

A request this link cannot read (another protocol version) is refused in the window and in ``opened.json``, and
leaves the window as it was.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from napari.utils.notifications import show_error, show_warning
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication

from eisgreis import api
from eisgreis_crboost import bundle as crboost_bundle
from eisgreis_crboost import protocol

logger = logging.getLogger(__name__)

POLL_MS = 1000


class Agent:
    """One window and what crboost asked it to show (see the module docstring)."""

    def __init__(self, viewer: api.TomogramViewer, session_dir: Path, renderer: str = "") -> None:
        self.viewer = viewer
        self.session_dir = Path(session_dir)
        self.renderer = renderer
        self.seq = -1
        self.request_mtime: Optional[int] = None
        self.outputs: Optional[Path] = None
        self.no_project_outputs = self.session_dir / protocol.OUTPUTS
        self.timer: Optional[QTimer] = None

    def watch(self) -> None:
        """Check for a new request every second, the first check once the event loop runs."""
        self.timer = QTimer()
        self.timer.timeout.connect(self.poll)
        self.timer.start(POLL_MS)

    def poll(self) -> None:
        path = self.session_dir / protocol.REQUEST
        try:
            mtime = path.stat().st_mtime_ns
        except FileNotFoundError:
            return
        if mtime == self.request_mtime:
            return
        self.request_mtime = mtime
        try:
            raw = protocol.read_json(path)
        except (OSError, ValueError) as e:
            # crboost writes the request with a rename, so a torn read is a fault on that side: say so and wait for
            # the next write.
            logger.error("unreadable %s: %s", path, e)
            show_error(f"crboost's request could not be read: {e}")
            return
        if raw is None:
            return
        seq = raw.get("seq") if isinstance(raw, dict) else None
        if isinstance(seq, int) and seq <= self.seq:
            return
        self.apply_raw(raw)

    def apply_raw(self, raw: Any) -> None:
        """Apply a request as read from ``open.json``; one this link cannot read is refused with a message."""
        try:
            request = protocol.parse_request(raw, api.version)
        except ValueError as e:  # ProtocolError included: refused, the window stays as it was
            seq = raw.get("seq") if isinstance(raw, dict) else None
            self.seq = seq if isinstance(seq, int) else self.seq
            logger.error("request refused: %s", e)
            show_error(f"crboost's request is not shown: {e}")
            self._ack(self.seq, error=str(e))
            return
        self.apply(request)

    def apply(self, request: protocol.Request) -> None:
        self.seq = request.seq
        viewer = self.viewer
        title = request.title
        outputs = request.outputs if request.outputs is not None else self.no_project_outputs
        error: Optional[str] = None
        opened: list[dict] = []
        bundle_report: Optional[dict] = None
        try:
            if not request.layers:
                raise ValueError("the request names no volume")
            path = request.layers[0].path.expanduser().resolve()
            previous = viewer.current
            new_project = outputs != self.outputs or previous is None
            if new_project or path != previous:
                if new_project:
                    self._point_at(outputs)
                else:
                    viewer.save()
                for layer in request.layers:
                    viewer.register(layer.path)
                # The load reads the whole volume and its FFT on the Qt thread (~20 s on 4 CPUs) and the window
                # freezes meanwhile: say what it is doing before it does.
                viewer.napari.title = f"Opening {title or path.stem}…"
                QApplication.processEvents()
                viewer.napari.dims.ndisplay = 2
                viewer.open(path)
                if new_project and previous is not None and previous != path:
                    viewer.forget(previous)
                viewer.reload_annotations()
            bundle_report = self._bundle(request.bundle, path)
            shape = viewer.shape
            voxel = viewer.pixel_size
            opened.append(
                {
                    "name": request.layers[0].name or path.stem,
                    "path": str(path),
                    "shape": list(shape) if shape else None,
                    "voxel_size": [voxel, voxel, voxel],
                    "shape_3d": list(shape) if shape else None,
                }
            )
            viewer.napari.title = title or path.stem
            if request.outputs is None:
                show_warning(
                    f"This tomogram is outside every crboost project: comments and particles save in {outputs}."
                )
        except Exception as e:  # the viewer is the user's window: report the failure and keep it open
            logger.exception("could not open %s", [str(layer.path) for layer in request.layers])
            error = f"{type(e).__name__}: {e}"
            show_error(f"Not opened: {error}")
        self._ack(request.seq, error=error, layers=opened, outputs=outputs, bundle=bundle_report)

    def _point_at(self, outputs: Path) -> None:
        """Save into `outputs` from now on. What the viewer holds is saved into the old folder first, then the old
        project's tomograms are forgotten; the one on screen goes once the next has opened."""
        viewer = self.viewer
        viewer.set_outputs(outputs)
        for path in viewer.tomograms:
            if path != viewer.current:
                viewer.forget(path)
        self.outputs = outputs

    def _bundle(self, raw: Optional[Path], path: Path) -> Optional[dict]:
        """Show the request's bundle with `path`'s tomogram: what was drawn and what crboost left out, or None when
        the request has none. A bundle that cannot be shown leaves the tomogram open and says why."""
        if raw is None:
            return None
        try:
            return crboost_bundle.show_bundle(self.viewer, Path(raw), expected_tomogram=path)
        except (OSError, ValueError) as e:  # ProtocolError included
            logger.exception("bundle %s not shown", raw)
            show_warning(f"crboost's picks and segmentations are not shown: {e}")
            return {"error": f"{type(e).__name__}: {e}"}

    def _ack(
        self,
        seq: int,
        *,
        error: Optional[str],
        layers: Optional[list] = None,
        outputs: Optional[Path] = None,
        bundle: Optional[dict] = None,
    ) -> None:
        layers = layers or []
        protocol.write_json(
            self.session_dir / protocol.ACK,
            {
                "seq": seq,
                "ok": error is None and bool(layers),
                "error": error,
                "layers": layers,
                "renderer": self.renderer,
                "viewer": protocol.VIEWER,
                "outputs": str(outputs) if outputs is not None else None,
                "bundle": bundle,
            },
        )


def log_napari_errors() -> None:
    """Every error napari shows as a notification (an exception in a button's handler, say) goes into the log with
    its traceback too: text on a remote desktop cannot be copied out, and napari's own traceback view fails on code
    it cannot find the source of."""
    import traceback

    from napari.utils.notifications import ErrorNotification, notification_manager

    def _log(event) -> None:
        if isinstance(event, ErrorNotification):
            exc = event.exception
            logger.error("%s", "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip())

    notification_manager.notification_ready.connect(_log)
