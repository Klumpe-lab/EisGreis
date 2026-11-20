"""The session files, without Qt: requests, bundles, desktop.json, and the command's --version."""

import json
import subprocess
import sys

import pytest

from eisgreis_crboost import protocol
from tests.fixtures import crboost_fixture


def test_crboost_request_fixture_is_version_1(tmp_path):
    raw = crboost_fixture("open.json", tmp_path)
    request = protocol.parse_request(raw)
    assert request.protocol == 1 and request.seq > 0
    assert request.layers[0].path.name.endswith(".mrc")
    assert request.outputs is not None and request.outputs.parts[-2:] == ("Curation", "napari")
    assert request.bundle is not None and request.bundle.name == "bundle.json"


def test_crboost_bundle_fixture_is_version_1(tmp_path):
    bundle = protocol.read_bundle(crboost_fixture("bundle.json", tmp_path))
    assert bundle["version"] == 1 and bundle["picks"] and bundle["segmentations"] and bundle["volumes"]


@pytest.mark.parametrize(
    "version, update", [(2, "update EisGreis"), (0, "update crboost"), (None, "update crboost"), ("1", "update")]
)
def test_unknown_request_versions_are_refused(tmp_path, version, update):
    raw = crboost_fixture("open.json", tmp_path)
    if version is None:
        raw.pop("protocol")
    else:
        raw["protocol"] = version
    with pytest.raises(protocol.ProtocolError, match=update):
        protocol.parse_request(raw, "0.1.0")


def test_malformed_requests():
    with pytest.raises(ValueError):
        protocol.parse_request({"protocol": 1, "seq": "soon", "layers": []})
    with pytest.raises(ValueError, match="no path"):
        protocol.parse_request({"protocol": 1, "seq": 1, "layers": [{"name": "x"}]})


def test_unknown_bundles_are_refused(tmp_path):
    bundle = crboost_fixture("bundle.json", tmp_path)
    with pytest.raises(protocol.ProtocolError, match="update EisGreis"):
        protocol.read_bundle({**bundle, "version": 2})
    with pytest.raises(protocol.ProtocolError, match="not a crboost bundle"):
        protocol.read_bundle({**bundle, "format": "something else"})
    with pytest.raises(ValueError, match="no tomogram"):
        protocol.read_bundle({**bundle, "tomogram": ""})


def test_write_json_renames_into_place(tmp_path):
    protocol.write_json(tmp_path / "opened.json", {"seq": 3})
    assert json.loads((tmp_path / "opened.json").read_text()) == {"seq": 3}
    assert [p.name for p in tmp_path.iterdir()] == ["opened.json"]


def test_desktop_record_heartbeat_and_registry(tmp_path):
    from eisgreis_crboost.desktop import Desktop

    session = tmp_path / "napari" / "abcd1234"
    registry = tmp_path / "napari" / "registry.jsonl"
    desktop = Desktop(session, registry)
    desktop.start()
    record = json.loads((session / "desktop.json").read_text())
    assert record["protocol"] == [1] and record["bundle_versions"] == [1]
    assert record["pid"] > 0 and record["node"] and record["window"] is False
    assert (session / "alive").exists()
    line = json.loads(registry.read_text().splitlines()[-1])
    assert line["kind"] == "desktop" and line["session_dir"] == str(session)
    desktop.set_window(True, "llvmpipe")
    assert json.loads((session / "desktop.json").read_text())["renderer"] == "llvmpipe"
    desktop.stop()
    assert not (session / "desktop.json").exists() and not (session / "alive").exists()


def test_version_needs_no_qt():
    """--version answers without loading napari or Qt: crboost's preflight asks it on a login node."""
    code = (
        "import sys, json; from eisgreis_crboost.__main__ import main; main(['--version']); "
        "assert not any(m.split('.')[0] in ('napari', 'qtpy', 'PyQt6') for m in sys.modules), 'Qt loaded'"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    record = json.loads(out.strip().splitlines()[-1])
    assert record["protocol"] == [1] and record["bundle_versions"] == [1] and record["eisgreis"]
