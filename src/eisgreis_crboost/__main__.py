"""``eisgreis-crboost``: EisGreis, driven by crboost through a session folder.

    eisgreis-crboost [files…] [--bundle FILE]… [--session-dir DIR | --root DIR] [--wait]

- In a remote desktop, by hand or started with it: the viewer makes its own session folder under ``--root``
  (``~/.crboost/napari``), registers it there (``registry.jsonl``), and crboost finds it and writes its requests into
  it. With ``--wait`` it opens no window until the first request, and after the window is closed it waits for the
  next one, so the desktop stays reachable from crboost.
- Started by crboost on a node (``--session-dir``): the folder is crboost's, which registered it. That desktop sets
  ``EISGREIS_NO_AUTOSTART``, so a ``--wait`` the user's autostart entry starts there stands down instead of becoming
  a second viewer.

Either way the viewer writes ``desktop.json`` and keeps ``alive`` fresh while it runs (``desktop.py``). Files given on
the command line open as a request would, the first on screen; a ``--bundle`` shows crboost's picks and
segmentations with its tomogram. Saves go to the request's project folder, or the session folder's ``outputs/``.

``--version`` prints what this link reads, as JSON, without starting Qt; ``--check`` loads napari and Qt (offscreen)
and prints their versions, to check an install.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import uuid
from pathlib import Path
from typing import Optional

from eisgreis import api
from eisgreis_crboost import protocol

logger = logging.getLogger("eisgreis_crboost")

_stop = threading.Event()


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="eisgreis-crboost", description=__doc__.split("\n\n")[0])
    parser.add_argument("files", nargs="*", type=Path, help="MRC volumes to open, the first on screen")
    parser.add_argument("--bundle", action="append", type=Path, default=[], help="a crboost bundle to show")
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--session-dir", type=Path, help="crboost's session folder (crboost registered it)")
    where.add_argument(
        "--root", type=Path, help=f"where to make a session folder and register it (default {protocol.default_root()})"
    )
    parser.add_argument("--wait", action="store_true", help="no window until crboost's first request")
    parser.add_argument("--profile", choices=("pc", "hpc"), help="machine settings (default: hpc under SLURM)")
    parser.add_argument("--workers", type=int, default=None, help="FFT threads (default: the profile's CPUs)")
    parser.add_argument("--version", action="store_true", help="print the versions this link reads, as JSON")
    parser.add_argument("--check", action="store_true", help="load napari and Qt (offscreen), print their versions")
    return parser.parse_args(argv)


def version_record() -> dict:
    return {
        "eisgreis": api.version,
        "protocol": list(protocol.PROTOCOLS),
        "bundle_versions": list(protocol.BUNDLE_VERSIONS),
    }


def check() -> dict:
    """Import what the viewer needs and start Qt without a display: proves the environment's libraries load."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import magicgui
    import mrcfile
    import napari
    import numpy
    import scipy
    from qtpy import API_NAME, QtCore
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    return {
        **version_record(),
        "napari": napari.__version__,
        "qt_api": API_NAME,
        "qt": QtCore.qVersion(),
        "qt_platform": app.platformName(),
        "magicgui": magicgui.__version__,
        "mrcfile": mrcfile.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "python": sys.version.split()[0],
    }


def _setup_logging(session_dir: Path) -> None:
    fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(sys.stderr), logging.FileHandler(session_dir / protocol.LOG)):
        handler.setFormatter(fmt)
        root.addHandler(handler)


def _on_signal(signum, _frame) -> None:
    """Close the window (which saves) and let the main loop end; the desktop record goes on the way out."""
    logger.info("signal %d: closing", signum)
    _stop.set()
    if "qtpy" in sys.modules:
        from qtpy.QtWidgets import QApplication

        app = QApplication.instance()
        if app is not None:
            app.closeAllWindows()


def _file_request(files: list[Path]) -> Optional[dict]:
    if not files:
        return None
    paths = [Path(f).expanduser().resolve() for f in files]
    return {
        "protocol": max(protocol.PROTOCOLS),
        "seq": 0,
        "title": paths[0].stem,
        "layers": [{"name": p.stem, "path": str(p)} for p in paths],
        "outputs": None,
        "bundle": None,
    }


def _wait_for_request(session_dir: Path, last_seq: int) -> bool:
    """Block until open.json holds a request newer than `last_seq`; False when a signal stopped the wait."""
    path = session_dir / protocol.REQUEST
    complained = False
    while not _stop.is_set():
        try:
            raw = protocol.read_json(path)
        except (OSError, ValueError) as e:
            if not complained:
                logger.error("unreadable %s: %s", path, e)
                complained = True
            raw = None
        seq = raw.get("seq") if isinstance(raw, dict) else None
        if isinstance(seq, int) and seq > last_seq:
            return True
        _stop.wait(1.0)
    return False


def _run_window(session_dir: Path, desktop, profile, first: Optional[dict], bundles: list[Path]) -> int:
    """One window until it closes: the agent applies `first` once the window is up, then every new request. The
    last request's seq."""
    from qtpy.QtCore import QTimer

    from eisgreis_crboost.agent import POLL_MS, Agent
    from eisgreis_crboost.bundle import show_bundle

    viewer = api.build(profile, title="EisGreis for crboost", outputs=session_dir / protocol.OUTPUTS)
    viewer.maximize()
    renderer = viewer.renderer
    logger.info("OpenGL renderer: %s", renderer)
    desktop.set_window(True, renderer)
    agent = Agent(viewer, session_dir, renderer)

    def _first() -> None:
        if first is not None:
            agent.apply_raw(first)
        for bundle in bundles:
            try:
                show_bundle(viewer, bundle)
            except (OSError, ValueError) as e:
                logger.exception("bundle %s not shown", bundle)
                from napari.utils.notifications import show_warning

                show_warning(f"{bundle} is not shown: {e}")

    # After the window has its maximized size, so the first tomogram fits the whole canvas. A request crboost wrote
    # into the folder is the agent's first poll's, a second later.
    QTimer.singleShot(POLL_MS, _first)
    agent.watch()
    api.run_event_loop()
    desktop.set_window(False)
    return agent.seq


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.version:
        print(json.dumps(version_record()))
        return
    if args.check:
        print(json.dumps(check()))
        return
    if args.wait and os.environ.get("EISGREIS_NO_AUTOSTART"):
        print("eisgreis-crboost: this desktop runs crboost's own viewer (EISGREIS_NO_AUTOSTART); not starting another")
        return

    if args.session_dir is not None:
        session_dir, registry = args.session_dir.expanduser().resolve(), None
    else:
        root = (args.root or protocol.default_root()).expanduser().resolve()
        session_dir, registry = root / uuid.uuid4().hex[:8], root / protocol.REGISTRY
    session_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(session_dir)
    logger.info("eisgreis %s, session folder %s", api.version, session_dir)

    # Before NumPy and SciPy are imported: they read the thread limits the profile sets.
    profile = api.make_profile(args.profile, args.workers)
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    from eisgreis_crboost.agent import log_napari_errors
    from eisgreis_crboost.desktop import Desktop

    desktop = Desktop(session_dir, registry)
    desktop.start()
    for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(signum, _on_signal)
    log_napari_errors()
    try:
        first = _file_request(args.files)
        if args.wait and first is None and not args.bundle:
            last = -1
        else:
            last = _run_window(session_dir, desktop, profile, first, args.bundle)
            if not args.wait:
                return
        while not _stop.is_set():
            logger.info("waiting for crboost's next request in %s", session_dir)
            if not _wait_for_request(session_dir, last):
                break
            last = max(last, _run_window(session_dir, desktop, profile, None, []))
    finally:
        desktop.stop()
        logger.info("stopped")


if __name__ == "__main__":
    main()
