"""crboost's bundle: one tomogram's picks, segmentations and volumes, shown as EisGreis's reference layers.

crboost writes a bundle per tomogram (its ``services/visualization/napari_bundle.py``); ``docs/crboost-protocol.md``
describes the format, version 1:

    {
      "format": "crboost-napari-bundle", "version": 1,
      "tomogram": "/abs/path/tomogram.mrc",   # the volume the coordinates refer to
      "tomo_name": "...", "project": "/abs/project",
      "shape_zyx": [512, 1024, 1024], "voxel_size": 6.2,   # that volume's grid (Å)
      "picks": [
        {"name": "ribosome · candidates (tmextractcand) (182)", "species": "ribosome",
         "source": "candidates tmextractcand", "color": "#ff1744", "box_size": 310.0,
         "coordinates_zyx": [[z, y, x], ...]}           # voxels of "tomogram"
      ],
      "segmentations": [
        {"name": "membrane (segment)", "path": "/abs/seg.mrc", "voxel_size": 6.2,
         "shape_zyx": [512, 1024, 1024], "level": 63.5, "full_scale": 127, "color": "#f50057",
         "visible": true}                               # shown where value > level (p = 0.5)
      ],
      "volumes": [                                     # pictures of this tomogram on its grid
        {"name": "reconstruction (tsReconstruct)", "kind": "reconstruction", "path": "/abs/rec.mrc"},
        {"name": "denoised · cryoCARE (denoisePredict)", "kind": "denoised", "path": "/abs/den.mrc"}
      ],
      "notes": ["what crboost left out, and why"]
    }

Every layer is named ``crboost · <name>``, so it reads as crboost's result, not an annotation of the user's.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from napari.utils.notifications import show_warning

from eisgreis import api
from eisgreis_crboost import protocol

logger = logging.getLogger(__name__)

PREFIX = "crboost · "


def show_bundle(
    viewer: api.TomogramViewer, source: Union[Path, dict], expected_tomogram: Optional[Path] = None
) -> dict:
    """Show the bundle at `source` with its tomogram, replacing what an earlier bundle showed for it; what was
    drawn, what was not and why (``not_shown``), and crboost's own ``notes``. Raises ProtocolError for a bundle
    this link does not read, ValueError for one of another tomogram than `expected_tomogram`."""
    bundle = protocol.read_bundle(source, api.version)
    tomogram = Path(bundle["tomogram"]).expanduser().resolve()
    if expected_tomogram is not None and tomogram != Path(expected_tomogram).expanduser().resolve():
        raise ValueError(f"the bundle is {bundle['tomogram']}'s")
    reference = viewer.reference
    reference.clear(tomogram=tomogram)
    pixel_size = float(bundle.get("voxel_size") or 0) or None
    problems: list[str] = []
    for seg in bundle.get("segmentations") or []:
        try:
            reference.show_segmentation(
                Path(seg["path"]),
                float(seg.get("level", 0.5)),
                seg.get("color", ""),
                seg.get("voxel_size") or pixel_size,
                name=PREFIX + str(seg.get("name") or Path(seg["path"]).stem),
                full_scale=float(seg.get("full_scale") or 1.0),
                visible=bool(seg.get("visible", True)),
                pixel_size=pixel_size,
                tomogram=tomogram,
            )
        except (KeyError, OSError, ValueError) as e:
            logger.exception("segmentation %s not shown", seg.get("path"))
            problems.append(f"{seg.get('name')}: {e}")
    for picks in bundle.get("picks") or []:
        try:
            reference.show_picks(
                PREFIX + str(picks.get("name") or "picks"),
                picks.get("coordinates_zyx") or [],
                picks.get("box_size"),
                picks.get("color", ""),
                source=str(picks.get("source") or ""),
                pixel_size=pixel_size,
                tomogram=tomogram,
            )
        except (TypeError, ValueError) as e:
            logger.exception("picks %s not shown", picks.get("name"))
            problems.append(f"{picks.get('name')}: {e}")
    for volume in bundle.get("volumes") or []:
        reference.add_volume(
            str(volume["name"]), Path(volume["path"]), str(volume.get("kind") or ""), tomogram=tomogram
        )
    if problems:
        show_warning("crboost layers not shown: " + "; ".join(problems))
    summary = reference.summary(tomogram=tomogram)
    return {**summary, "not_shown": list(summary.get("not_shown") or []) + problems, "notes": bundle.get("notes") or []}
