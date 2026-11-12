# Napari Conda Environment

This is a napari viewer optimized to look through your data, while keeping notes and marking interesting particles

Create the environment with:

```bash
conda env create -f environment.yml
conda activate napari-lorenz
```

Run the MRC viewer with:

```bash
python napari_mrc_lowpass.py
```

Open one or more files immediately on launch with:

```bash
python napari_mrc_lowpass.py /path/to/volume1.mrc /path/to/volume2.mrc
```

### Fast viewer: PC and HPC launchers

`napari_mrc_lowpass_fast.py` is a faster copy of the viewer with the same features. It uses `shared_filters_fast.py` and `shared_mrc_fast.py`, and the original scripts are unchanged. Start it through the launcher for the machine you are on:

```bash
# Workstation / laptop: all cores, FFT kept in RAM for quick cutoff changes
python napari_mrc_lowpass_pc.py /path/to/volume.mrc

# HPC remote desktop (run in a terminal inside the desktop session)
python napari_mrc_lowpass_hpc.py /path/to/volume.mrc
# with a GPU visualisation node and VirtualGL:
vglrun python napari_mrc_lowpass_hpc.py /path/to/volume.mrc
```

The HPC launcher only uses the CPUs given to the job (`SLURM_CPUS_PER_TASK` or CPU affinity). It keeps the FFT cache only if the job has at least 24 GB of memory, which you can override with `--spectrum-cache` or `--no-spectrum-cache`. It also prints the OpenGL renderer and warns when rendering is software-only. Both launchers accept `--workers N`.

Measured on a 1024 x 1024 x 512 tomogram on an M3 Max:

| | Open tomogram | Change cutoff | Peak RAM |
|---|---|---|---|
| Original viewer | ~30 s | ~18 s | 19.4 GB |
| Fast viewer | ~0.7 s | ~0.5 s | 11.3 GB (9.2 GB without FFT cache) |

Ask for about 16 GB of memory per HPC desktop session for tomograms this size.

The main filter dock lets you add multiple MRC files, switch between them with a file selector, and apply a low-pass value in Angstrom using a numeric entry field. A status line shows how many tomograms are registered and how many are currently loaded. The filter dropdown switches between `hard`, `gaussian`, `butterworth`, and `cosine`.

Comments live in a separate dock widget. Each tomogram can have a free-text comment, comments are saved to CSV, can be reloaded from a previous session, and are auto-saved on viewer close to `tomogram_comments.csv` by default unless you choose a different CSV path in the comments dock.

Particle annotations live in their own dock widget. Create multiple napari points layers per tomogram, give each one a particle name such as `ribosome`, and save each layer to its own STAR file in `particle_stars/` by default. The STAR filename includes both the tomogram stem and the layer name, and layers are auto-saved when you switch tomograms or close the viewer.

Run the tilt-series browser with:

```bash
python napari_tilt_series_excluder.py
```

Or open a specific `.mdoc` plus the folder containing the tilt images:

```bash
python napari_tilt_series_excluder.py --mdoc /path/to/series.mdoc --tilt-dir /path/to/tilts
```

The tilt-series browser reads SerialEM `.mdoc` metadata, sorts the entries by `TiltAngle` like the RELION bad-tilt exclusion workflow, matches each entry against files in the chosen tilt folder, and lets you step through the images while marking each tilt as kept or excluded.

This environment now starts from the `napari-mrcfile-reader` plugin stack, installed into a conda-based napari environment via `pip`.
