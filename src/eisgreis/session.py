"""What the viewer holds apart from what it draws: the tomograms, the one on screen, their pixel sizes, the comments,
the particle points of every tomogram, and where saves go. No Qt and no napari: the docks and the viewer read and
change it, and it can be tested on its own.

What it saves:

- comments: one CSV with a ``path`` and a ``comment`` column, a row for every tomogram;
- particles: one STAR per tomogram and particle name, ``<tomogram stem>__<name>.star`` in the STAR folder, the
  points as ``_rlnCoordinateZ/Y/X`` in Å from the volume's corner, and ``_rlnBoxSize`` in Å;
- PNGs, into the PNG folder (see ``scalebar.py``).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np

from eisgreis.io.mrc import read_voxel_size_from_path
from eisgreis.io.star import load_points_from_star, sanitize_star_token, save_points_to_star


@dataclass
class Outputs:
    """Where the viewer saves: the comments CSV, the STAR folder, the PNG folder."""

    comments_csv: Path
    star_dir: Path
    png_dir: Path

    @classmethod
    def under(cls, folder: Path) -> "Outputs":
        folder = Path(folder).expanduser()
        return cls(folder / "tomogram_comments.csv", folder / "particle_stars", folder / "tomogram_pngs")


class Session:
    """The viewer's state (see the module docstring)."""

    def __init__(self, outputs: Optional[Outputs] = None) -> None:
        self.paths: list[Path] = []  # registered tomograms, sorted
        self.current: Optional[Path] = None  # the one on screen
        self.comments: dict[Path, str] = {}
        self.pixel_sizes: dict[Path, float] = {}  # Å, the header's or what the filter dock set
        self.particles: dict[str, dict[Path, np.ndarray]] = {}  # name -> tomogram -> points (pixels, Z Y X)
        self.box_sizes: dict[str, float] = {}  # name -> box (Å)
        # `default_outputs` is where saves go unless the docks name another CSV or STAR folder (`outputs`).
        self.default_outputs = outputs or Outputs.under(Path.cwd())
        self.outputs = replace(self.default_outputs)

    # ── tomograms ──────────────────────────────────────────────────────────────────────────────────────────────

    def register(self, path: Path) -> Path:
        """Add `path` to the file list (once); its resolved path."""
        resolved = Path(path).expanduser().resolve()
        if resolved not in self.paths:
            self.paths.append(resolved)
            self.paths.sort()
        return resolved

    def forget(self, path: Path) -> None:
        """Drop `path` from the file list, its comment and its particles, so no later save writes them."""
        path = Path(path).expanduser().resolve()
        if path in self.paths:
            self.paths.remove(path)
        self.comments.pop(path, None)
        for per_path in self.particles.values():
            per_path.pop(path, None)
        if self.current == path:
            self.current = None

    def file_labels(self) -> dict[str, Path]:
        """The file list's labels: the file name, and `name [n]` for the n-th of the same name."""
        labels: dict[str, Path] = {}
        counts: dict[str, int] = {}
        for path in self.paths:
            base = path.name
            counts[base] = counts.get(base, 0) + 1
            labels[base if counts[base] == 1 else f"{base} [{counts[base]}]"] = path
        return labels

    def label_for(self, path: Path) -> Optional[str]:
        for label, candidate in self.file_labels().items():
            if candidate == path:
                return label
        return None

    def pixel_size_of(self, path: Path, fallback: float) -> float:
        return self.pixel_sizes.get(path, fallback)

    # ── where saves go ─────────────────────────────────────────────────────────────────────────────────────────

    def set_outputs(self, outputs: Outputs) -> None:
        self.default_outputs = outputs
        self.outputs = replace(outputs)

    def set_comments_csv(self, value: Optional[str]) -> None:
        """The comments CSV the docks name; empty means the default one."""
        raw = (value or "").strip()
        self.outputs.comments_csv = Path(raw).expanduser().resolve() if raw else self.default_outputs.comments_csv

    def set_star_dir(self, value: Optional[str]) -> None:
        """The STAR folder the docks name; empty means the default one."""
        raw = (value or "").strip()
        self.outputs.star_dir = Path(raw).expanduser().resolve() if raw else self.default_outputs.star_dir

    # ── comments ───────────────────────────────────────────────────────────────────────────────────────────────

    def set_comment(self, text: str) -> None:
        if self.current is not None:
            self.comments[self.current] = text.strip()

    def comment(self, path: Optional[Path] = None) -> str:
        path = path or self.current
        return self.comments.get(path, "") if path is not None else ""

    def load_comments_csv(self, csv_path: Path) -> None:
        """Read comments from a CSV with ``path`` and ``comment`` columns, or from a plain two-column one."""
        loaded: dict[Path, str] = {}
        with Path(csv_path).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames and {"path", "comment"}.issubset(set(reader.fieldnames)):
                for row in reader:
                    path_value = (row.get("path") or "").strip()
                    if not path_value:
                        continue
                    loaded[Path(path_value).expanduser().resolve()] = (row.get("comment") or "").strip()
            else:
                handle.seek(0)
                for row in csv.reader(handle):
                    if len(row) < 2:
                        continue
                    path_value = row[0].strip()
                    if path_value.lower() == "path":
                        continue
                    loaded[Path(path_value).expanduser().resolve()] = row[1].strip()
        self.comments.update(loaded)

    def save_comments_csv(self, csv_path: Optional[Path] = None) -> Path:
        """Write every tomogram's comment (registered or commented) to `csv_path`, the comments CSV by default."""
        csv_path = Path(csv_path) if csv_path is not None else self.outputs.comments_csv
        rows = [
            {"path": str(path), "comment": self.comments.get(path, "")}
            for path in sorted(set(self.paths) | set(self.comments))
        ]
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "comment"])
            writer.writeheader()
            writer.writerows(rows)
        return csv_path

    # ── particles ──────────────────────────────────────────────────────────────────────────────────────────────

    def star_path(self, path: Path, particle_name: str) -> Path:
        return self.outputs.star_dir / f"{path.stem}__{sanitize_star_token(particle_name)}.star"

    def store_particles(self, particle_name: str, path: Path, coords: np.ndarray) -> None:
        self.particles.setdefault(particle_name, {})[path] = np.asarray(coords, dtype=float).copy()

    def particles_of(self, particle_name: str, path: Optional[Path]) -> Optional[np.ndarray]:
        if path is None:
            return None
        return self.particles.get(particle_name, {}).get(path)

    def save_particles(self, particle_name: str, path: Path, pixel_size: float) -> Path:
        """Write one tomogram's points of `particle_name` to its STAR (Å); `pixel_size` converts them."""
        coords = self.particles_of(particle_name, path)
        if coords is None:
            coords = np.empty((0, 3), dtype=float)
        star = self.star_path(path, particle_name)
        save_points_to_star(
            coords * pixel_size, star, particle_name, box_size_angstrom=self.box_sizes.get(particle_name)
        )
        return star

    def save_all_particles(self, fallback_pixel_size: float) -> int:
        """Write every tomogram's points of every particle name, each tomogram's at its pixel size
        (`fallback_pixel_size` for one that has none); the number of STAR files written."""
        saved = 0
        for particle_name, per_path in self.particles.items():
            for path in per_path:
                self.save_particles(particle_name, path, self.pixel_size_of(path, fallback_pixel_size))
                saved += 1
        return saved

    def load_particle_stars(self, star_dir: Path) -> list[str]:
        """Read ``<tomogram stem>__<name>.star`` files of the registered tomograms back into points; the particle
        name of every file read, in order."""
        by_stem = {path.stem: path for path in self.paths}
        names: list[str] = []
        for star_path in sorted(Path(star_dir).glob("*.star")):
            stem = star_path.stem
            if "__" not in stem:
                continue
            tomogram_stem, particle_token = stem.split("__", 1)
            tomogram_path = by_stem.get(tomogram_stem)
            if tomogram_path is None:
                continue
            coords_angstrom, box_size_angstrom = load_points_from_star(star_path)
            pixel_size = self.pixel_sizes.get(tomogram_path)
            if pixel_size is None:
                pixel_size = read_voxel_size_from_path(tomogram_path) or 1.0
                self.pixel_sizes[tomogram_path] = pixel_size
            particle_name = particle_token.replace("_", " ").strip() or particle_token
            coords_pixels = coords_angstrom / pixel_size if coords_angstrom.size else coords_angstrom
            self.particles.setdefault(particle_name, {})[tomogram_path] = coords_pixels
            if box_size_angstrom is not None:
                self.box_sizes[particle_name] = box_size_angstrom
            names.append(particle_name)
        return names
