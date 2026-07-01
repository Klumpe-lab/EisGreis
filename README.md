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

Open a file immediately on launch with:

```bash
python napari_mrc_lowpass.py /path/to/volume.mrc
```

The dock widget exposes a low-pass slider in Angstrom, using the voxel size stored in the MRC header, and a dropdown to switch between `hard`, `gaussian`, `butterworth`, and `cosine` filters.
