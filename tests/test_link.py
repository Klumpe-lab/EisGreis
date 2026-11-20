"""The crboost link on crboost's own session files: a request opens its tomogram, a project switch saves into the old
folder first, a bundle shows, a request outside every project saves into the session folder, and a request of a
protocol this link does not read is refused with a message."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.fixtures import build_volumes, crboost_fixture


@pytest.fixture
def agent(viewer, tmp_path):
    from eisgreis_crboost.agent import Agent

    session = tmp_path / "session"
    session.mkdir(exist_ok=True)
    return Agent(viewer, session, renderer="test")


def ack(agent) -> dict:
    return json.loads((agent.session_dir / "opened.json").read_text())


def request(tmp_path, **changes) -> dict:
    raw = crboost_fixture("open.json", tmp_path)
    build_volumes(tmp_path)
    raw.update(changes)
    return raw


def test_a_request_opens_its_tomogram(agent, tmp_path):
    raw = request(tmp_path, bundle=None)
    agent.apply_raw(raw)
    answer = ack(agent)
    assert answer["ok"] and answer["seq"] == raw["seq"] and answer["viewer"] == "eisgreis"
    layer = answer["layers"][0]
    assert layer["path"] == str(Path(raw["layers"][0]["path"]).resolve())
    assert layer["shape"] == list(agent.viewer.shape) and layer["voxel_size"][0] == pytest.approx(10.0)
    assert answer["outputs"] == raw["outputs"]
    assert agent.viewer.current == Path(raw["layers"][0]["path"]).resolve()
    assert agent.viewer.outputs["star_dir"] == Path(raw["outputs"]) / "particle_stars"
    nz = agent.viewer.shape[0]
    assert agent.viewer.napari.dims.point[0] == pytest.approx(nz // 2 * 10.0)


def test_a_project_switch_saves_into_the_old_folder_first(agent, tmp_path, mrc):
    raw = request(tmp_path, bundle=None)
    agent.apply_raw(raw)
    viewer, controller = agent.viewer, agent.viewer._c
    controller.comments_dock.comment.value = "lamella edge"
    layer = controller.particles_dock.create_layer("ribosome")
    layer.add([2.0, 3.0, 4.0])
    old_outputs = Path(raw["outputs"])
    old_tomogram = viewer.current

    other_project = tmp_path / "data" / "other_project"
    (other_project / "Curation" / "napari").mkdir(parents=True)
    other = mrc("TS_02.mrc", shape=(8, 16, 16), voxel_size=10.0, folder=other_project / "tomograms")
    agent.apply_raw(
        {
            "protocol": 1,
            "seq": raw["seq"] + 1,
            "title": "TS_02",
            "layers": [{"name": "TS_02", "path": str(other)}],
            "outputs": str(other_project / "Curation" / "napari"),
            "bundle": None,
        }
    )
    assert ack(agent)["ok"]
    rows = (old_outputs / "tomogram_comments.csv").read_text()
    assert "lamella edge" in rows and str(old_tomogram) in rows
    star = old_outputs / "particle_stars" / f"{old_tomogram.stem}__ribosome.star"
    assert star.is_file() and "20.000000 30.000000 40.000000" in star.read_text()  # voxels x 10 Å
    assert viewer.tomograms == [other.resolve()]
    assert viewer.outputs["comments_csv"] == other_project / "Curation" / "napari" / "tomogram_comments.csv"

    # Back to the first project: its comment and particles come back from its folder.
    agent.apply_raw({**raw, "seq": raw["seq"] + 2})
    assert ack(agent)["ok"]
    assert controller.comments_dock.comment.value == "lamella edge"
    np.testing.assert_allclose(controller.particles_dock.layers["ribosome"].data, [[2.0, 3.0, 4.0]])


def test_a_bundle_shows_with_its_tomogram(agent, tmp_path):
    raw = request(tmp_path)
    agent.apply_raw(raw)
    answer = ack(agent)
    assert answer["ok"], answer
    report = answer["bundle"]
    assert report["picks"] and all(name.startswith("crboost · ") for name in report["picks"])
    assert report["segmentations"] and not report["not_shown"]
    assert len(report["volumes"]) >= 2
    names = [layer.name for layer in agent.viewer.napari.layers]
    assert all(name in names for name in report["picks"] + report["segmentations"])


def test_a_bundle_of_another_tomogram_is_reported(agent, tmp_path, mrc):
    raw = request(tmp_path)
    other = mrc("elsewhere.mrc", shape=(8, 16, 16), voxel_size=10.0)
    raw["layers"] = [{"name": "elsewhere", "path": str(other)}]
    agent.apply_raw(raw)
    answer = ack(agent)
    assert answer["ok"] and "the bundle is" in answer["bundle"]["error"]


def test_outside_every_project_saves_into_the_session_folder(agent, tmp_path):
    raw = request(tmp_path, outputs=None, bundle=None)
    agent.apply_raw(raw)
    answer = ack(agent)
    assert answer["ok"] and answer["outputs"] == str(agent.session_dir / "outputs")
    assert agent.viewer.outputs["png_dir"] == agent.session_dir / "outputs" / "tomogram_pngs"


@pytest.mark.parametrize("version", [2, None])
def test_an_unknown_protocol_is_refused_with_a_message(agent, tmp_path, version):
    raw = request(tmp_path, bundle=None)
    if version is None:
        raw.pop("protocol")
    else:
        raw["protocol"] = version
    agent.apply_raw(raw)
    answer = ack(agent)
    assert not answer["ok"] and answer["seq"] == raw["seq"]
    assert ("update EisGreis" if version == 2 else "update crboost") in answer["error"]
    assert agent.viewer.current is None


def test_polling_applies_each_new_request_once(agent, tmp_path):
    from eisgreis_crboost import protocol

    raw = request(tmp_path, bundle=None)
    protocol.write_json(agent.session_dir / "open.json", raw)
    agent.poll()
    first = ack(agent)
    (agent.session_dir / "opened.json").unlink()
    agent.poll()  # the same request: nothing to do
    assert not (agent.session_dir / "opened.json").exists()
    protocol.write_json(agent.session_dir / "open.json", {**raw, "seq": raw["seq"] + 5})
    agent.poll()
    assert ack(agent)["seq"] == first["seq"] + 5
