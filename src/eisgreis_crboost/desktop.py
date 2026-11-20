"""``desktop.json``, the heartbeat and the registry line: how crboost finds a running viewer and learns what it reads.

Every viewer the link runs writes ``desktop.json`` into its session folder (the protocol and bundle versions it
reads, where it runs, its versions) and touches ``alive`` every ``HEARTBEAT_S`` seconds; both files go when it exits.
crboost takes a viewer as live while its SLURM job runs and ``alive`` is fresh, and writes the newest protocol both
sides know. A viewer that made its own session folder (one started in a desktop, by hand or with it) also appends a
line to ``registry.jsonl`` beside the session folders, which is how crboost finds it; one started in a folder crboost
made was registered by crboost.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
from datetime import datetime
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import Optional

from eisgreis import api
from eisgreis_crboost import protocol

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class Desktop:
    """The session folder's ``desktop.json`` and ``alive``, and the registry line (see the module docstring)."""

    def __init__(self, session_dir: Path, registry: Optional[Path] = None) -> None:
        self.session_dir = Path(session_dir)
        self.registry = registry
        self.started = _now()
        self.renderer: Optional[str] = None
        self.window = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def record(self) -> dict:
        try:
            napari_version = api.napari_version()
        except PackageNotFoundError:  # an environment without napari's metadata: unknown, said as such
            napari_version = None
        return {
            "protocol": list(protocol.PROTOCOLS),
            "bundle_versions": list(protocol.BUNDLE_VERSIONS),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID"),
            "node": socket.gethostname().split(".")[0],
            "display": os.environ.get("DISPLAY"),
            "pid": os.getpid(),
            "started": self.started,
            "eisgreis": api.version,
            "napari": napari_version,
            "renderer": self.renderer,
            "window": self.window,
        }

    def write(self) -> None:
        protocol.write_json(self.session_dir / protocol.DESKTOP, self.record())

    def start(self) -> None:
        """Write desktop.json and alive, start the heartbeat, and register the session when this viewer made it."""
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.write()
        self._touch()
        if self.registry is not None:
            line = {
                "kind": "desktop",
                "slurm_job_id": os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID"),
                "session_dir": str(self.session_dir),
                "started": self.started,
            }
            self.registry.parent.mkdir(parents=True, exist_ok=True)
            with self.registry.open("a") as f:
                f.write(json.dumps(line) + "\n")
        self._thread = threading.Thread(target=self._beat, name="eisgreis-heartbeat", daemon=True)
        self._thread.start()

    def set_window(self, open_: bool, renderer: Optional[str] = None) -> None:
        """Say whether a window is up, and what it draws with."""
        self.window = open_
        if renderer is not None:
            self.renderer = renderer
        self.write()

    def stop(self) -> None:
        """Stop the heartbeat and remove desktop.json and alive: crboost no longer finds this viewer."""
        self._stop.set()
        for name in (protocol.DESKTOP, protocol.ALIVE):
            try:
                (self.session_dir / name).unlink()
            except FileNotFoundError:  # already gone: the stop ran before
                pass

    def _touch(self) -> None:
        (self.session_dir / protocol.ALIVE).touch()

    def _beat(self) -> None:
        warned = False
        while not self._stop.wait(protocol.HEARTBEAT_S):
            try:
                self._touch()
                warned = False
            except OSError as e:
                if not warned:
                    logger.error("the heartbeat cannot be written into %s: %s", self.session_dir, e)
                    warned = True
