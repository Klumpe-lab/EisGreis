from __future__ import annotations

from pathlib import Path
import re

import numpy as np


def sanitize_star_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip())
    token = token.strip("_")
    return token or "annotations"


def save_points_to_star(
    coordinates: np.ndarray,
    output_path: Path,
    layer_name: str,
    box_size_angstrom: float | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ndim = coordinates.shape[1] if coordinates.ndim == 2 else 0
    title = sanitize_star_token(layer_name)

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"data_{title}\n\n")
        handle.write("loop_\n")
        if ndim >= 3:
            handle.write("_rlnCoordinateZ #1\n")
            handle.write("_rlnCoordinateY #2\n")
            handle.write("_rlnCoordinateX #3\n")
            if box_size_angstrom is not None:
                handle.write("_rlnBoxSize #4\n")
            for coord in coordinates:
                z, y, x = coord[:3]
                if box_size_angstrom is None:
                    handle.write(f"{z:.6f} {y:.6f} {x:.6f}\n")
                else:
                    handle.write(f"{z:.6f} {y:.6f} {x:.6f} {box_size_angstrom:.6f}\n")
        elif ndim >= 2:
            handle.write("_rlnCoordinateY #1\n")
            handle.write("_rlnCoordinateX #2\n")
            if box_size_angstrom is not None:
                handle.write("_rlnBoxSize #3\n")
            for coord in coordinates:
                y, x = coord[:2]
                if box_size_angstrom is None:
                    handle.write(f"{y:.6f} {x:.6f}\n")
                else:
                    handle.write(f"{y:.6f} {x:.6f} {box_size_angstrom:.6f}\n")
        else:
            handle.write("_rlnCoordinateX #1\n")
            if box_size_angstrom is not None:
                handle.write("_rlnBoxSize #2\n")
            for coord in coordinates:
                if box_size_angstrom is None:
                    handle.write(f"{coord[0]:.6f}\n")
                else:
                    handle.write(f"{coord[0]:.6f} {box_size_angstrom:.6f}\n")


def load_points_from_star(output_path: Path) -> tuple[np.ndarray, float | None]:
    lines = output_path.read_text(encoding="utf-8").splitlines()
    headers: list[str] = []
    rows: list[list[float]] = []
    in_loop = False

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("data_"):
            continue
        if stripped == "loop_":
            in_loop = True
            continue
        if not in_loop:
            continue
        if stripped.startswith("_"):
            headers.append(stripped.split()[0])
            continue
        try:
            rows.append([float(value) for value in stripped.split()])
        except ValueError:
            continue

    if not headers or not rows:
        return np.empty((0, 0), dtype=float), None

    header_index = {name: idx for idx, name in enumerate(headers)}
    coords: list[list[float]] = []
    for row in rows:
        if "_rlnCoordinateZ" in header_index:
            coords.append(
                [
                    row[header_index["_rlnCoordinateZ"]],
                    row[header_index["_rlnCoordinateY"]],
                    row[header_index["_rlnCoordinateX"]],
                ]
            )
        elif "_rlnCoordinateY" in header_index:
            coords.append(
                [
                    row[header_index["_rlnCoordinateY"]],
                    row[header_index["_rlnCoordinateX"]],
                ]
            )
        elif "_rlnCoordinateX" in header_index:
            coords.append([row[header_index["_rlnCoordinateX"]]])

    box_size_angstrom = None
    if "_rlnBoxSize" in header_index and rows:
        box_size_angstrom = rows[0][header_index["_rlnBoxSize"]]

    return np.asarray(coords, dtype=float), box_size_angstrom
