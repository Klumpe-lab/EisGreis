"""The session files crboost and the viewer exchange: their names, versions, reading and writing. No Qt, no napari.

``docs/crboost-protocol.md`` is the reference. In short, a session folder (``~/.crboost/napari/<id>/``) holds:

- ``open.json``, crboost → viewer: ``protocol``, ``seq`` (a higher one replaces the request before it), ``title``,
  ``layers`` (``[{name, path}]``, the first opened), ``outputs`` (the project's folder for saves, or null) and
  ``bundle`` (the path of ``bundle.json``, or null);
- ``bundle.json``, crboost → viewer: the tomogram's picks, segmentations and volumes
  (``format: crboost-napari-bundle``, ``version``);
- ``opened.json``, viewer → crboost: what the window shows for the request ``seq``;
- ``desktop.json`` and ``alive``, viewer → crboost: which versions the viewer reads, where it runs, and a heartbeat.

The registry, ``registry.jsonl`` beside the session folders, gets a line ``{kind: "desktop", ...}`` for a viewer that
made its own session folder. Every file is written beside its final name and renamed into place.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

PROTOCOLS = (1,)  # the open.json versions this link reads
BUNDLE_FORMAT = "crboost-napari-bundle"
BUNDLE_VERSIONS = (1,)  # the bundle.json versions this link reads
VIEWER = "eisgreis"  # opened.json's `viewer`

REQUEST = "open.json"
ACK = "opened.json"
BUNDLE = "bundle.json"
DESKTOP = "desktop.json"
ALIVE = "alive"
REGISTRY = "registry.jsonl"
LOG = "eisgreis.log"
OUTPUTS = "outputs"  # in the session folder: where saves go for a tomogram outside every project
HEARTBEAT_S = 10


class ProtocolError(ValueError):
    """A session file this link cannot read; the message says which side to update."""


def _versions(versions) -> str:
    return ", ".join(str(v) for v in versions)


def default_root() -> Path:
    """Where crboost's session folders live: the home directory, which the headnode and every desktop share."""
    return Path.home() / ".crboost" / "napari"


@dataclass(frozen=True)
class Layer:
    name: str
    path: Path


@dataclass(frozen=True)
class Request:
    """``open.json``, read."""

    protocol: int
    seq: int
    title: str
    layers: tuple[Layer, ...]
    outputs: Optional[Path]
    bundle: Optional[Path]


def parse_request(raw: Any, eisgreis_version: str = "") -> Request:
    """``open.json``'s content as a Request. Raises ProtocolError for a version this link does not read, and
    ValueError for a request it cannot make sense of."""
    if not isinstance(raw, dict):
        raise ValueError(f"the request is no JSON object ({type(raw).__name__})")
    viewer = f"EisGreis {eisgreis_version}".strip()
    protocol = raw.get("protocol")
    if protocol is None:
        raise ProtocolError(
            f"crboost's request names no protocol version, so it comes from a crboost older than {viewer} reads "
            f"(protocol {_versions(PROTOCOLS)}): update crboost"
        )
    if not isinstance(protocol, int) or protocol not in PROTOCOLS:
        side = "EisGreis" if isinstance(protocol, int) and protocol > max(PROTOCOLS) else "crboost"
        raise ProtocolError(
            f"crboost wrote protocol {protocol}; {viewer} reads protocol {_versions(PROTOCOLS)}: update {side}"
        )
    try:
        seq = int(raw.get("seq", 0))
    except (TypeError, ValueError) as e:
        raise ValueError(f"the request's seq is not a number: {raw.get('seq')!r}") from e
    layers = []
    for spec in raw.get("layers") or []:
        if not isinstance(spec, dict) or not spec.get("path"):
            raise ValueError(f"a layer of the request names no path: {spec!r}")
        path = Path(str(spec["path"]))
        layers.append(Layer(name=str(spec.get("name") or path.stem), path=path))
    outputs = raw.get("outputs")
    bundle = raw.get("bundle")
    return Request(
        protocol=protocol,
        seq=seq,
        title=str(raw.get("title") or ""),
        layers=tuple(layers),
        outputs=Path(str(outputs)) if outputs else None,
        bundle=Path(str(bundle)) if bundle else None,
    )


def read_json(path: Path) -> Optional[Any]:
    """A session file's content, or None when it does not exist. Raises ValueError for one that does not parse:
    the writer renames into place, so a torn file is a fault on the writing side."""
    try:
        text = Path(path).read_text()
    except FileNotFoundError:
        return None
    return json.loads(text)


def write_json(path: Path, content: Any) -> None:
    """Write beside the final name and rename into place, so a reader never sees half a file."""
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(content))
    os.replace(tmp, path)


def read_bundle(source: Union[str, Path, dict], eisgreis_version: str = "") -> dict:
    """A bundle from its path or its parsed content, checked for format and version. Raises ProtocolError for one
    this link does not read, ValueError for one that names no tomogram."""
    bundle = source if isinstance(source, dict) else json.loads(Path(source).expanduser().read_text())
    viewer = f"EisGreis {eisgreis_version}".strip()
    if bundle.get("format") != BUNDLE_FORMAT:
        raise ProtocolError(f"not a crboost bundle (format {bundle.get('format')!r})")
    version = bundle.get("version")
    if version not in BUNDLE_VERSIONS:
        side = "EisGreis" if isinstance(version, int) and version > max(BUNDLE_VERSIONS) else "crboost"
        raise ProtocolError(
            f"crboost wrote bundle version {version!r}; {viewer} reads {_versions(BUNDLE_VERSIONS)}: update {side}"
        )
    if not bundle.get("tomogram"):
        raise ValueError("the bundle names no tomogram")
    return bundle
