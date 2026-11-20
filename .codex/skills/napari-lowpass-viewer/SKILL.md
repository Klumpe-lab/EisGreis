---
name: napari-lowpass-viewer
description: Use when working on EisGreis, the napari tomogram viewer in this repo, especially lazy loading of tomograms, MRC-like file handling, Fourier filtering, the comments and particle docks, or the crboost link. Covers src/eisgreis (the viewer) and src/eisgreis_crboost (the link to crboost).
---

# EisGreis

This skill is for maintaining the viewer in this repo, a uv project (`pyproject.toml`, `uv.lock`):

- `src/eisgreis/`: the viewer. `app.py` builds it (the controller), `session.py` holds its state without Qt,
  `docks/` has one module per dock, `io/` reads MRC, STAR and mdoc files, `filters.py` the low-pass and bandpass,
  `api.py` is what code outside the viewer may use, `_napari.py` every use of napari's private modules.
- `src/eisgreis_crboost/`: the link to crboost. It uses `eisgreis.api` only; the viewer never imports it.

## When to use

Use this skill when the task involves:

- lazy loading of tomograms or image stacks
- MRC-like file registration and switching in the napari UI
- low-pass or Gaussian bandpass Fourier filtering
- the comments dock, particle annotation layers and STAR export
- keeping the viewer startup responsive with many `.mrc` files

## Working rules

1. Prefer updating `filters.py` or `io/` instead of duplicating logic in a dock.
2. Keep startup lazy: register file choices first and only read/filter image data when a file becomes active.
3. Keep comments in their own dock and persist them through CSV import/export.
4. The filter dock uses a numeric low-pass input field, not a slider.
5. Keep particle annotations as separate napari `Points` layers with per-tomogram points and one STAR file per
   particle layer and tomogram.
6. State goes into `session.py`, widgets into `docks/`; anything of napari's that is private goes into `_napari.py`.
7. `eisgreis` never imports `eisgreis_crboost`; the link only uses `eisgreis.api` (`tests/test_boundary.py`).

## Validation

- `uv run ruff check .` and `uv run ruff format --check .`
- `uv run pytest` (viewer tests need a display; headless: `xvfb-run -a uv run pytest` with `QT_QPA_PLATFORM=offscreen`)
- For runtime checks, `uv run eisgreis /path/to/volume.mrc`.

## Notes

- The viewer supports: `.mrc`, `.mrcs`, `.map`, `.st`, `.rec`, `.preali`, `.ali`
- Files opened through napari `File/Open File(s)` also appear in the file selector.
- Comments are saved as CSV keyed by resolved file path.
- Particle annotations are saved as STAR files keyed by tomogram stem and layer name.
