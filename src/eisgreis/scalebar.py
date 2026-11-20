"""The scale bar in nanometres, on screen and burned into an exported PNG.

On screen the bar takes a round length (1, 2, 5, 10, 15, 20, 25, 50, 75, 100, 125, 150, 200, 500 or 750 nm, times
a power of 1000) no longer than a quarter of the canvas, and always reads in nm. The PNG export writes the slice on
screen at full resolution, in the layer's contrast and colour map, with a white 100 nm bar in its bottom-right
corner.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PREFERRED_NM = (1, 2, 5, 10, 15, 20, 25, 50, 75, 100, 125, 150, 200, 500, 750)
PNG_SCALE_BAR_NM = 100.0


def preferred_length_nm(desired_nm: float) -> float:
    """The longest round length no longer than `desired_nm`; the shortest one when every one is longer."""
    if not np.isfinite(desired_nm) or desired_nm <= 0:
        return 1.0
    candidates = sorted(value * 1000.0**power for power in range(-6, 7) for value in PREFERRED_NM)
    smaller = [value for value in candidates if value <= desired_nm]
    return smaller[-1] if smaller else candidates[0]


def format_nm(length_nm: float) -> str:
    if 0.01 <= abs(length_nm) < 1_000_000:
        text = f"{length_nm:.3f}".rstrip("0").rstrip(".")
    else:
        text = f"{length_nm:.3g}"
    return f"{text} nm"


def length_and_label(desired_nm: float) -> tuple[float, str]:
    """For the on-screen bar: the length to draw (nm) and its label."""
    length = preferred_length_nm(desired_nm)
    return length, format_nm(length)


def slice_rgb(viewer, layer) -> np.ndarray:
    """The 2-D slice of `layer` on screen at full resolution, with the layer's contrast and colormap (uint8 RGB)."""
    data = layer.data
    if data.ndim < 2:
        raise ValueError("Layer has fewer than 2 dimensions")
    offset = viewer.dims.ndim - data.ndim
    layer_order = [axis - offset for axis in viewer.dims.order if axis - offset >= 0]
    row_axis, col_axis = layer_order[-2], layer_order[-1]

    position = layer.world_to_data(viewer.dims.point)
    index: list[object] = []
    for axis in range(data.ndim):
        if axis in (row_axis, col_axis):
            index.append(slice(None))
        else:
            index.append(int(np.clip(round(float(position[axis])), 0, data.shape[axis] - 1)))
    plane = np.asarray(data[tuple(index)], dtype=np.float32)
    if row_axis > col_axis:
        plane = plane.T

    low, high = (float(v) for v in layer.contrast_limits)
    scaled = np.clip((plane - low) / max(high - low, 1e-12), 0.0, 1.0)
    if layer.gamma != 1:
        scaled = scaled ** float(layer.gamma)
    rgba = layer.colormap.map(scaled.ravel()).reshape(plane.shape + (4,))
    return (np.clip(rgba[..., :3], 0.0, 1.0) * 255).round().astype(np.uint8)


def slice_suffix(viewer) -> str:
    """The slice's part of a PNG's name: ``z<index>`` in a volume."""
    point = list(viewer.dims.point)
    if len(point) >= 3:
        return f"z{int(round(point[0]))}"
    if len(point) == 2:
        return "slice"
    return "_".join(f"d{idx}_{int(round(value))}" for idx, value in enumerate(point))


def burn_scale_bar(rgb: np.ndarray, pixel_size_angstrom: float, length_nm: float):
    """White bar with label in the bottom-right corner, black outline for contrast. Returns (PIL image, bar px)."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    height, width = rgb.shape[:2]
    short_side = min(height, width)
    bar_px = length_nm * 10.0 / pixel_size_angstrom
    margin = max(round(0.04 * short_side), 6)
    bar_h = max(round(0.012 * short_side), 3)
    font_size = max(round(0.04 * short_side), 12)
    try:
        font = ImageFont.load_default(size=font_size)
    except TypeError:  # Pillow < 10.1 has one size only
        font = ImageFont.load_default()

    x1 = width - margin
    x0 = x1 - bar_px
    y1 = height - margin
    y0 = y1 - bar_h
    outline = max(1, bar_h // 3)
    draw.rectangle([x0 - outline, y0 - outline, x1 + outline, y1 + outline], fill=(0, 0, 0))
    draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255))
    label = f"{length_nm:g} nm"
    draw.text(
        ((x0 + x1) / 2, y0 - outline - max(2, bar_h // 2)),
        label,
        font=font,
        fill=(255, 255, 255),
        anchor="md",
        stroke_width=max(1, font_size // 12),
        stroke_fill=(0, 0, 0),
    )
    return image, bar_px


def save_slice_png(viewer, layer, output_dir: Path, stem: str, pixel_size_angstrom: float) -> tuple[Path, str]:
    """Write the slice of `layer` on screen as ``<stem>__<slice>.png`` in `output_dir`; (path, a line describing it)."""
    rgb = slice_rgb(viewer, layer)
    pixel_size = pixel_size_angstrom if pixel_size_angstrom > 0 else 1.0
    image, bar_px = burn_scale_bar(rgb, pixel_size, PNG_SCALE_BAR_NM)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{stem}__{slice_suffix(viewer)}.png"
    image.save(output_path)
    detail = f"{output_path} ({rgb.shape[1]}x{rgb.shape[0]} px, {PNG_SCALE_BAR_NM:g} nm = {bar_px:.1f} px)"
    return output_path, detail
