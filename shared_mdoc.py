from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class MdocTiltEntry:
    z_value: int
    tilt_angle: float
    sub_frame_path: Optional[str]
    datetime: Optional[str]

    @property
    def basename(self) -> Optional[str]:
        if not self.sub_frame_path:
            return None
        return Path(str(self.sub_frame_path).replace("\\", "/")).name

    @property
    def stem(self) -> Optional[str]:
        basename = self.basename
        return None if basename is None else Path(basename).stem


def parse_mdoc(path: Path) -> list[MdocTiltEntry]:
    current: dict[str, str] = {}
    entries: list[MdocTiltEntry] = []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("[") and line.endswith("]"):
                if current:
                    entry = _entry_from_section(current)
                    if entry is not None:
                        entries.append(entry)
                current = {"__header__": line}
                continue

            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            current[key.strip()] = value.strip()

    if current:
        entry = _entry_from_section(current)
        if entry is not None:
            entries.append(entry)

    if not entries:
        raise ValueError(f"No [ZValue] sections found in {path}")

    return entries


def resolve_tilt_image_path(entry: MdocTiltEntry, tilt_directory: Path) -> Path:
    if not tilt_directory.is_dir():
        raise NotADirectoryError(f"Tilt directory does not exist: {tilt_directory}")

    candidates = [path for path in tilt_directory.iterdir() if path.is_file()]
    by_name = {path.name.lower(): path for path in candidates}
    by_stem = {path.stem.lower(): path for path in candidates}

    basename = entry.basename
    if basename:
        exact = by_name.get(basename.lower())
        if exact is not None:
            return exact

    stem = entry.stem
    if stem:
        exact_stem = by_stem.get(stem.lower())
        if exact_stem is not None:
            return exact_stem

        stem_matches = [path for path in candidates if stem.lower() in path.stem.lower()]
        if len(stem_matches) == 1:
            return stem_matches[0]

    raise FileNotFoundError(
        f"Could not match tilt image for mdoc entry {entry.z_value} "
        f"({basename or 'missing SubFramePath'}) in {tilt_directory}"
    )


def sort_entries_by_tilt_angle(entries: list[MdocTiltEntry]) -> list[MdocTiltEntry]:
    return sorted(entries, key=lambda entry: (entry.tilt_angle, entry.z_value))


def _entry_from_section(section: dict[str, str]) -> Optional[MdocTiltEntry]:
    header = section.get("__header__", "")
    if not header.startswith("[ZValue"):
        return None

    z_value = _parse_z_value(header)
    tilt_angle = float(section["TiltAngle"])
    return MdocTiltEntry(
        z_value=z_value,
        tilt_angle=tilt_angle,
        sub_frame_path=section.get("SubFramePath"),
        datetime=section.get("DateTime"),
    )


def _parse_z_value(header: str) -> int:
    inner = header.strip("[]")
    _, value = inner.split("=", 1)
    return int(value.strip())
