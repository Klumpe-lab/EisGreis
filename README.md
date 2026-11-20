# EisGreis

A napari viewer optimized to look through your tomograms, while keeping notes and marking interesting particles.

## Install and run

With [uv](https://docs.astral.sh/uv/), on macOS or Linux:

```bash
git clone https://github.com/Klumpe-lab/EisGreis.git
cd EisGreis
uv sync
uv run eisgreis /path/to/volume1.mrc /path/to/volume2.mrc
```

The first file opens; the others wait in the file list. Run it in the folder where the comments CSV, the particle
STARs and the PNGs should go. On a cluster, `install.sh` makes a lab install with launchers for remote desktops:
[docs/install.md](docs/install.md).

`eisgreis --profile pc|hpc` picks the machine's settings (without it, `hpc` where SLURM is present):

```bash
# Workstation / laptop: all cores, FFT kept in RAM for quick cutoff changes
uv run eisgreis --profile pc /path/to/volume.mrc

# HPC remote desktop: only the CPUs given to the job (SLURM_CPUS_PER_TASK or CPU affinity), the FFT cache only with
# at least 24 GB of job memory (--spectrum-cache / --no-spectrum-cache override that); it prints the OpenGL renderer
# and warns when rendering is software-only
uv run eisgreis --profile hpc /path/to/volume.mrc
```

Both accept `--workers N`.

Measured on a 1024 x 1024 x 512 tomogram on an M3 Max:

| | Open tomogram | Change cutoff | Peak RAM |
|---|---|---|---|
| First viewer | ~30 s | ~18 s | 19.4 GB |
| EisGreis | ~0.7 s | ~0.5 s | 11.3 GB (9.2 GB without FFT cache) |

Ask for about 16 GB of memory per HPC desktop session for tomograms this size.

## What it does

The main filter dock lets you add multiple MRC files, switch between them with a file selector, and apply a low-pass
value in Angstrom using a numeric entry field. A status line shows how many tomograms are registered and how many are
currently loaded. The filter dropdown switches between `hard`, `gaussian`, `butterworth`, and `cosine`.

Comments live in a separate dock widget. Each tomogram can have a free-text comment, comments are saved to CSV, can
be reloaded from a previous session, and are auto-saved on viewer close to `tomogram_comments.csv` by default unless
you choose a different CSV path in the comments dock.

Particle annotations live in their own dock widget. Create multiple napari points layers per tomogram, give each one
a particle name such as `ribosome`, and save each layer to its own STAR file in `particle_stars/` by default. The STAR
filename includes both the tomogram stem and the layer name, and layers are auto-saved when you switch tomograms or
close the viewer.

The scale bar and the measurement layer read in nm; *Save current view PNG* writes the slice on screen, at full
resolution, with a 100 nm scale bar burned in.

Reference layers show picks, segmentations and other volumes of a tomogram beside your own layers, without saving
them; 3-D (Ctrl+Y) works on any renderer, from binned copies of the big volumes, with segmentations as smooth
surfaces.

Run the tilt-series browser with:

```bash
uv run eisgreis-tilts
```

Or open a specific `.mdoc` plus the folder containing the tilt images:

```bash
uv run eisgreis-tilts --mdoc /path/to/series.mdoc --tilt-dir /path/to/tilts
```

The tilt-series browser reads SerialEM `.mdoc` metadata, sorts the entries by `TiltAngle` like the RELION bad-tilt
exclusion workflow, matches each entry against files in the chosen tilt folder, and lets you step through the images
while marking each tilt as kept or excluded.

`uv run eisgreis-bandpass` opens a crYOLO-style Gaussian low-pass and high-pass filter on any open image layer.

## For code

`eisgreis.api` is the one module code outside the viewer imports: `build()`, `run()`, and `TomogramViewer` with
`open`, `set_outputs`, `save`, `reload_annotations`, `reference.show_picks` / `show_segmentation` / `add_volume`,
and `view3d.set_seg_mode`. Everything else is internal.

`eisgreis_crboost` is the link to crboost, the cryo-ET pipeline server: `eisgreis-crboost` shows the tomograms crboost
asks for. It uses `eisgreis.api` only, and the viewer never imports it; the two sides talk through versioned files
([docs/crboost-protocol.md](docs/crboost-protocol.md)).

Tests: `uv run pytest` (the viewer's tests need a display for OpenGL; headless, `xvfb-run -a uv run pytest` with
`QT_QPA_PLATFORM=offscreen`). Lint: `uv run ruff check .` and `uv run ruff format .`.

## Licence

BSD-3-Clause, as napari's: [LICENSE](LICENSE).
