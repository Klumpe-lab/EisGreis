# Installing EisGreis

## On your computer

With [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/Klumpe-lab/EisGreis.git
cd EisGreis
uv sync
uv run eisgreis /path/to/tomogram.mrc
```

`uv sync` builds `.venv` from `uv.lock` with uv's own Python (`.python-version`), the same on macOS and Linux.
`uv run eisgreis-tilts` and `uv run eisgreis-bandpass` start the other two tools.

## On a cluster

`install.sh` makes the checkout an install for a whole lab, for use in remote desktops (an Open OnDemand Remote
Desktop, say):

```bash
git clone https://github.com/Klumpe-lab/EisGreis.git /path/to/software/EisGreis
/path/to/software/EisGreis/install.sh
```

It puts everything inside the checkout:

- a pinned uv in `.uv/bin`, and uv's own Python in `.uv/python` (never the host's: an environment built on the host's
  Python does not start inside a desktop image that lacks it);
- the environment in `.uv/env/<python>-<12 hex of the sha256 of uv.lock>`, built under a temporary name, renamed when
  complete and never changed afterwards. EisGreis itself is installed editable, so the launchers run the checkout's
  code;
- `xlibs/`: five X libraries pip's Qt needs for its xcb platform (`libxcb-cursor.so.0`, `libxcb-icccm.so.4`,
  `libxcb-image.so.0`, `libxcb-keysyms.so.1`, `libxcb-render-util.so.0`), which some desktop images lack. They are
  conda-forge's xcb-util builds (MIT licensed), fetched by pinned URL and checked by sha256; no conda is involved.

Updating is `git pull` and `install.sh` again. A pull that changes `uv.lock` builds a new environment beside the old
one, and a viewer that is running keeps its own; remove old environments by hand once nothing runs them.

### The launchers

`eisgreis` and `eisgreis-crboost` at the top of the install start the viewer from its environment (`launch.sh`):

- `PYTHONPATH`, `PYTHONHOME` and `QT_PLUGIN_PATH` are cleared, so cluster modules and other environments stay out;
- `LD_LIBRARY_PATH` is filtered, not cleared: a GPU desktop's GL arrangement stays (its `glvnd` folder and
  `/.singularity.d/libs`, where apptainer's `--nv` puts the NVIDIA libraries), every other entry goes;
- `xlibs/` goes on `LD_LIBRARY_PATH` only when the system lacks one of the five (`ldconfig -p`), so a desktop image
  that has them uses its own;
- on a full GPU (`nvidia-smi -L` names a GPU and no MIG device) with a display, the viewer runs under VirtualGL
  (`vglrun -d egl`) and draws on the card; on a MIG slice or a CPU node Mesa draws in software. `EISGREIS_VGL=no`
  turns VirtualGL off.

In a remote desktop's terminal, in the folder where comments, STARs and PNGs should go:

```bash
/path/to/software/EisGreis/eisgreis /path/to/tomogram.mrc
```

Ask for about 16 GB for a 1k x 1k x 512 tomogram; with 24 GB or more the viewer keeps its FFT in memory, and a new
low-pass cutoff takes one inverse FFT.

### With crboost

crboost's conf.yaml names the install (`napari.eisgreis`); `./crboost eisgreis-install /path/to/software/EisGreis`
clones a release into the folder, runs `install.sh` and prints that line. crboost runs `eisgreis-crboost` and talks to
it through files in `~/.crboost/napari/` (`docs/crboost-protocol.md`). In a desktop, `eisgreis-crboost` makes the
desktop one crboost can show tomograms in; `eisgreis-crboost --wait` does the same with no window until crboost's first
request, which is what crboost's User settings put into the desktop's autostart.
