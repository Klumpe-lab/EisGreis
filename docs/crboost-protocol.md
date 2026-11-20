# The crboost protocol

crboost, the cryo-ET pipeline server, shows tomograms in EisGreis. The two share no code: crboost runs an EisGreis
install's `eisgreis-crboost` and talks to it only through the files below. This page is their reference; crboost's
writers cite it. The link's code is `src/eisgreis_crboost/` (`protocol.py` holds the names and versions).

Version 1.

## The session folder

`~/.crboost/napari/<id>/`. The home directory is the one place the headnode and every desktop share. Every file is
written beside its final name and renamed into place, so a reader never sees half of one: a file that does not parse
is a fault on the writing side.

A session folder comes from one of two places:

- **crboost** makes it when it starts the viewer on a node of its own (a VNC desktop, `eisgreis-crboost
  --session-dir <folder>`), and registers it.
- **The viewer** makes it when it starts in a user's own remote desktop (`eisgreis-crboost`, by hand, or `--wait` from
  the desktop's autostart), and registers it: a line `{"kind": "desktop", "slurm_job_id", "session_dir", "started"}`
  appended to `~/.crboost/napari/registry.jsonl`.

A desktop crboost starts for its own session sets `EISGREIS_NO_AUTOSTART` in it: the user's autostart entry, which
the desktop runs too, then stands down rather than register a second viewer.

## `open.json`, crboost → viewer

```json
{
  "protocol": 1,
  "seq": 1795108724038,
  "title": "TS_01",
  "layers": [{"name": "TS_01", "path": "/abs/TS_01_10.00Apx.mrc"}],
  "outputs": "/abs/project/Curation/napari",
  "bundle": "/home/user/.crboost/napari/1a2b3c4d/bundle.json"
}
```

- `protocol`: the version of this file. crboost writes the newest version both sides read (below).
- `seq`: a clock in ms. A request with a higher `seq` replaces the one before it; the viewer applies each once.
- `title`: the window's title.
- `layers`: the volumes; the first is opened, the rest join the file list.
- `outputs`: where the viewer saves (comments CSV, `particle_stars/`, `tomogram_pngs/`): the project's
  `Curation/napari/`, or null for a volume outside every project. The viewer then saves into the session folder's
  `outputs/`, never its working directory, and says so.
- `bundle`: the path of `bundle.json`, or null.

The viewer holds one project at a time. Before it opens another tomogram it saves what it holds into the current
folder; a request with other `outputs` saves into the old folder, forgets the old project's tomograms, and points the
saves at the new one. Then it reads the comments and STARs back from the folder.

A request without `protocol`, or with a version the viewer does not read, is refused: the viewer shows an error and
answers with `ok: false` and the reason, naming the side to update.

## `bundle.json`, crboost → viewer

One tomogram's picks, segmentations and other volumes, shown as reference layers named `crboost · <name>`, never saved
(`src/eisgreis_crboost/bundle.py`). Built by crboost's `services/visualization/napari_bundle.py`.

```json
{
  "format": "crboost-napari-bundle", "version": 1,
  "tomogram": "/abs/TS_01_10.00Apx.mrc", "tomo_name": "TS_01", "project": "/abs/project",
  "shape_zyx": [24, 40, 36], "voxel_size": 10.0,
  "picks": [{"name": "ribosome · candidates (tmextractcand) (2)", "species": "ribosome",
             "source": "candidates tmextractcand", "color": "#ff1744", "box_size": 300.0,
             "coordinates_zyx": [[12.0, 20.0, 18.0], [15.0, 16.0, 23.0]]}],
  "segmentations": [{"name": "membrane (segment)", "path": "/abs/TS_01__membrane.mrc", "voxel_size": 20.0,
                     "shape_zyx": [12, 20, 18], "level": 63.5, "full_scale": 127.0, "color": "#f50057",
                     "visible": true}],
  "volumes": [{"name": "reconstruction (tsReconstruct)", "kind": "reconstruction", "path": "/abs/rec.mrc"},
              {"name": "denoised · cryocare (denoisePredict)", "kind": "denoised", "path": "/abs/den.mrc"}],
  "notes": ["what crboost left out, and why"]
}
```

- `tomogram` is the volume the coordinates refer to; a bundle of another tomogram than the request's is reported, not
  drawn.
- Picks are voxels (Z, Y, X) of `tomogram`, whose voxel is `voxel_size` Å; `box_size` is in Å.
- A segmentation keeps its own grid: `voxel_size` relative to the tomogram's says how much coarser it is, and both
  grids share the volume's corner. It shows where its value passes `level`.
- `volumes` are pictures of the same tomogram on its grid; with two or more, the viewer's Reference dock switches
  between them.

## `opened.json`, viewer → crboost

```json
{
  "seq": 1795108724038, "ok": true, "error": null,
  "layers": [{"name": "TS_01", "path": "/abs/TS_01_10.00Apx.mrc", "shape": [24, 40, 36],
              "voxel_size": [10.0, 10.0, 10.0], "shape_3d": [24, 40, 36]}],
  "renderer": "llvmpipe (LLVM 21.1.8, 256 bits)", "viewer": "eisgreis",
  "outputs": "/abs/project/Curation/napari",
  "bundle": {"picks": ["crboost · ribosome · candidates (tmextractcand) (2)"],
             "segmentations": ["crboost · membrane (segment)"],
             "volumes": ["reconstruction (tsReconstruct)", "denoised · cryocare (denoisePredict)"],
             "not_shown": [], "notes": ["what crboost left out, and why"]}
}
```

What the window shows for request `seq`. `outputs` is the folder the viewer saves into (the session folder's
`outputs/` for a request without one). `bundle` is null without one, or `{"error": …}` when it could not be shown;
`not_shown` lists what the viewer could not draw, `notes` what crboost left out.

## `desktop.json` and `alive`, viewer → crboost

```json
{
  "protocol": [1], "bundle_versions": [1],
  "slurm_job_id": "1832128", "node": "cbe-g5-8", "display": ":1", "pid": 4242,
  "started": "2026-11-19T18:30:00+01:00", "eisgreis": "0.1.0", "napari": "0.10.0",
  "renderer": "NVIDIA B200/PCIe/SSE2", "window": true
}
```

Every viewer writes it, whichever started it, and touches `alive` every 10 s; both go when it exits. `window` is false
while a viewer started with `--wait` has no window yet. crboost takes a viewer as live while its SLURM job runs (any
job: a desktop's is the portal's) and `alive` is under 30 s old; the newest live one is the user's viewer.

## Versions

In `desktop.json` the viewer says which `open.json` versions (`protocol`) and bundle versions it reads, and crboost
writes the newest one both know. Before a viewer has started, crboost asks the install (`eisgreis-crboost
--version`, JSON, no Qt). A side that meets a version it does not know says so in its UI and never falls back
silently; crboost's dialog reads, for instance, "EisGreis 0.1.0 at /path reads protocol 1, this crboost writes 2:
update EisGreis".

A change of either format comes with a new version, and with a PR here that adds the new fixtures to
`tests/fixtures/crboost/` (crboost's `scripts/eisgreis_fixtures.py` writes them with crboost's own writers) and the
version the link reads.
