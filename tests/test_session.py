"""The viewer's state without Qt: comments, per-tomogram particles, where saves go."""

import numpy as np
import pytest

from eisgreis.io.star import load_points_from_star
from eisgreis.session import Outputs, Session


def test_comments_csv_round_trip(tmp_path, mrc):
    a, b = mrc("a.mrc"), mrc("b.mrc")
    session = Session(Outputs.under(tmp_path / "out"))
    session.register(a)
    session.register(b)
    session.current = a.resolve()
    session.set_comment("  thick ice  ")
    csv_path = session.save_comments_csv()
    assert csv_path == tmp_path / "out" / "tomogram_comments.csv"
    assert csv_path.read_text().splitlines()[0] == "path,comment"

    again = Session(Outputs.under(tmp_path / "out"))
    again.load_comments_csv(csv_path)
    assert again.comment(a.resolve()) == "thick ice"
    assert again.comment(b.resolve()) == ""


def test_plain_two_column_csv(tmp_path, mrc):
    a = mrc("a.mrc")
    plain = tmp_path / "plain.csv"
    plain.write_text(f"path,note\n{a},crystalline\n")
    session = Session(Outputs.under(tmp_path))
    session.load_comments_csv(plain)
    assert session.comment(a.resolve()) == "crystalline"


def test_particles_saved_per_tomogram_and_read_back(tmp_path, mrc):
    a, b = mrc("a.mrc", voxel_size=5.0), mrc("b.mrc", voxel_size=10.0)
    session = Session(Outputs.under(tmp_path / "out"))
    a, b = session.register(a), session.register(b)
    session.pixel_sizes = {a: 5.0, b: 10.0}
    session.box_sizes["ribosome"] = 300.0
    session.store_particles("ribosome", a, np.array([[1.0, 2.0, 3.0]]))
    session.store_particles("ribosome", b, np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]))
    assert session.save_all_particles(fallback_pixel_size=1.0) == 2

    star_a = tmp_path / "out" / "particle_stars" / "a__ribosome.star"
    coords, box = load_points_from_star(star_a)
    np.testing.assert_allclose(coords, [[5.0, 10.0, 15.0]])  # Å = pixels x 5 Å
    assert box == pytest.approx(300.0)

    again = Session(Outputs.under(tmp_path / "out"))
    again.register(a)
    again.register(b)
    names = again.load_particle_stars(tmp_path / "out" / "particle_stars")
    assert sorted(names) == ["ribosome", "ribosome"]
    np.testing.assert_allclose(again.particles_of("ribosome", b), [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    assert again.pixel_sizes[a] == pytest.approx(5.0, abs=1e-4)


def test_output_folders(tmp_path):
    session = Session(Outputs.under(tmp_path / "first"))
    session.set_star_dir(str(tmp_path / "elsewhere"))
    assert session.outputs.star_dir == (tmp_path / "elsewhere").resolve()
    session.set_star_dir("")
    assert session.outputs.star_dir == tmp_path / "first" / "particle_stars"
    session.set_outputs(Outputs.under(tmp_path / "project" / "Curation" / "napari"))
    assert session.outputs.png_dir == tmp_path / "project" / "Curation" / "napari" / "tomogram_pngs"
    assert session.outputs.comments_csv.name == "tomogram_comments.csv"


def test_file_labels_and_forget(tmp_path, mrc):
    one = mrc("tomo.mrc", folder=tmp_path / "one")
    two = mrc("tomo.mrc", folder=tmp_path / "two")
    session = Session(Outputs.under(tmp_path))
    one, two = session.register(one), session.register(two)
    assert list(session.file_labels()) == ["tomo.mrc", "tomo.mrc [2]"]
    session.current = two
    session.set_comment("keep?")
    session.store_particles("x", two, np.zeros((1, 3)))
    session.forget(two)
    assert session.paths == [one] and session.current is None
    assert two not in session.comments and session.particles_of("x", two) is None
