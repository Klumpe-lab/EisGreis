---
name: napari-lowpass-viewer
description: Use when working on the Napari_Lorenz napari viewers, especially lazy loading of tomograms, MRC-like file handling, shared Fourier filtering, or the low-pass and cryolo-style filter widgets. Covers the main viewer script, the cryolo-style script, and the shared filter/MRC helper modules.
---

# Napari Lowpass Viewer

This skill is for maintaining the napari viewer scripts in this repo:

- `napari_mrc_lowpass.py`
- `napari_cryolo_bandpass.py`
- `shared_filters.py`
- `shared_mrc.py`
- `shared_star.py`

## When to use

Use this skill when the task involves:

- lazy loading of tomograms or image stacks
- MRC-like file registration and switching in the napari UI
- low-pass or Gaussian bandpass Fourier filtering
- sharing filter or MRC I/O code across the viewer scripts
- particle annotation layers and STAR export for tomograms
- keeping the napari viewer startup responsive with many `.mrc` files

## Current architecture

- `napari_mrc_lowpass.py` is the main multi-file viewer.
- It should register many files quickly, show a status summary, and only load/filter the selected file.
- `napari_cryolo_bandpass.py` provides a cryolo-style Qt widget for Gaussian LP/HP filtering on already-open image layers.
- `shared_filters.py` owns Fourier-domain filtering helpers.
- `shared_mrc.py` owns plugin-based MRC loading and voxel-size extraction from file paths.
- `shared_star.py` owns STAR-writing helpers for particle annotation layers.

## Working rules

1. Prefer updating `shared_filters.py`, `shared_mrc.py`, or `shared_star.py` instead of duplicating logic in a viewer script.
2. Keep startup lazy in `napari_mrc_lowpass.py`.
   It should register file choices first and only read/filter image data when a file becomes active.
3. Keep comments in their own dock widget and persist them through CSV import/export.
4. The main viewer uses a numeric low-pass input field, not a slider.
5. Preserve `NUMBA_DISABLE_JIT=1` near the top of viewer scripts unless the environment issue is explicitly resolved.
6. Prefer `napari-mrcfile-reader` for image loading paths handled by the main viewer.
7. Keep particle annotations as separate napari `Points` layers with per-tomogram visibility and one STAR file per particle layer.
8. When changing filter behavior, update both the UI-facing script and the shared helper if the logic is meant to be reused.

## Validation

- Syntax check with:
  - `python3 -m py_compile shared_filters.py`
  - `python3 -m py_compile shared_mrc.py`
  - `python3 -m py_compile shared_star.py`
  - `python3 -m py_compile napari_mrc_lowpass.py`
  - `python3 -m py_compile napari_cryolo_bandpass.py`
- For runtime checks, prefer launching through the `napari-lorenz` conda environment.

## Notes

- The main viewer supports: `.mrc`, `.mrcs`, `.map`, `.st`, `.rec`, `.preali`, `.ali`
- Files opened through napari `File/Open File(s)` should also appear in the viewer selector.
- Comments are saved as CSV keyed by resolved file path.
- Particle annotations are saved as STAR files keyed by tomogram path and layer name.
