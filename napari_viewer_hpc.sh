#!/usr/bin/env bash
# Start the fast napari MRC viewer on the HPC cluster.
#
# Run it from a terminal inside an OpenOnDemand Remote Desktop session, in the
# folder where comments CSV, STAR files and PNGs should be written:
#     napari-viewer [volume.mrc ...] [--workers N] [--no-spectrum-cache]
#
# Expected layout (the environment sits next to the repository):
#     <install dir>/Napari_Lorenz/   this repository
#     <install dir>/env/             conda env from environment.yml
#     <install dir>/napari-viewer -> Napari_Lorenz/napari_viewer_hpc.sh
# Set NAPARI_LORENZ_ENV to use an environment somewhere else.
set -euo pipefail

script="$(readlink -f "${BASH_SOURCE[0]}")"
repo="$(dirname "$script")"
env_dir="${NAPARI_LORENZ_ENV:-$(dirname "$repo")/env}"

if [[ ! -x "$env_dir/bin/python" ]]; then
    echo "napari-viewer: conda environment not found at $env_dir" >&2
    exit 1
fi

# Keep cluster modules (e.g. EESSI's Python and libraries) from shadowing the
# environment; its Python finds its own libraries without these.
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH QT_PLUGIN_PATH
export PATH="$env_dir/bin:$PATH"
export CONDA_PREFIX="$env_dir"

exec "$env_dir/bin/python" "$repo/napari_mrc_lowpass_hpc.py" "$@"
