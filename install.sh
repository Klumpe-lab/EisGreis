#!/bin/sh
# The cluster install of EisGreis, in this checkout: its Python environment, and the X libraries Qt needs in a remote
# desktop that lacks them. Prints the environment's path on stdout; progress goes to stderr.
#
# The environment is named by what goes into it, <python>-<12 hex of the sha256 of uv.lock>. It is built under a
# temporary name, renamed when complete, and never changed afterwards: an update (git pull, then this script) that
# changes the lock builds a new environment beside the old one, and a viewer that is running keeps its own. EisGreis
# itself is installed editable, so the launchers run this checkout's code.
#
#   ./install.sh          find or build the environment and the X libraries; print the environment's path
#   ./install.sh --path   print the environment's path, never build
#
# What goes in is data: uv.lock (what), .python-version (which Python, uv's own build, never the host's), UV_VERSION
# below (which uv). EISGREIS_UV names a uv binary of your own instead of the pinned one fetched into .uv/bin.
set -eu
here=$(cd "$(dirname "$0")" && pwd -P)

UV_VERSION=0.13.7

# The five X libraries pip's Qt needs for its xcb platform, which some remote desktops lack (Rocky 9's
# ood-desktop image among them): conda-forge's xcb-util builds, fetched by pinned URL and checked by sha256.
# package  sha256  file in its lib/  name the loader asks for
CONDA_FORGE=https://conda.anaconda.org/conda-forge/linux-64
XLIBS="
xcb-util-cursor-0.1.6-hb03c661_0.conda c2be9cae786fdb2df7c2387d2db31b285cf90ab3bfabda8fa75a596c3d20fc67
    libxcb-cursor.so.0.0.0 libxcb-cursor.so.0
xcb-util-wm-0.4.2-hb711507_0.conda 31d44f297ad87a1e6510895740325a635dd204556aa7e079194a0034cdd7e66a
    libxcb-icccm.so.4.0.0 libxcb-icccm.so.4
xcb-util-image-0.4.0-hb83e432_3.conda ca6ef0d589c8508356bb19836c3b09fa1fd8a28073be22454e8b3707208a7ffc
    libxcb-image.so.0.0.0 libxcb-image.so.0
xcb-util-keysyms-0.4.1-hb711507_0.conda 546e3ee01e95a4c884b6401284bb22da449a2f4daf508d038fdfa0712fe4cc69
    libxcb-keysyms.so.1.0.0 libxcb-keysyms.so.1
xcb-util-renderutil-0.3.10-hb83e432_1.conda 3c93689b6420ea6442618bfb0711902947739858cfeb2bf6a76ab2aace9dc9c0
    libxcb-render-util.so.0.0.0 libxcb-render-util.so.0
"
XLIB_NAMES="libxcb-cursor.so.0 libxcb-icccm.so.4 libxcb-image.so.0 libxcb-keysyms.so.1 libxcb-render-util.so.0"

log() { printf 'install: %s\n' "$*" >&2; }
die() { log "$*"; exit 1; }

sha256() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

python=$(sed -n '1p' "$here/.python-version" 2>/dev/null | tr -d '[:space:]')
printf '%s' "$python" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$' || die ".python-version must hold an exact x.y.z, got '$python'"
[ -f "$here/uv.lock" ] || die "no uv.lock in $here"
name=$python-$(sha256 "$here/uv.lock" | cut -c1-12)
env_root=$here/.uv/env
env=$env_root/$name
uv=${EISGREIS_UV:-$here/.uv/bin/uv}

# Nothing the shell loaded may reach a build.
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE VIRTUAL_ENV CONDA_PREFIX

# uv with this checkout's caches and Pythons, and nobody's own uv configuration.
run_uv() {
    UV_NO_CONFIG=1 UV_CACHE_DIR="$here/.uv/cache" UV_PYTHON_INSTALL_DIR="$here/.uv/python" \
        UV_PYTHON_PREFERENCE=only-managed "$uv" "$@"
}

ensure_uv() {
    [ -z "${EISGREIS_UV:-}" ] || return 0
    [ "$("$uv" --version 2>/dev/null | cut -d' ' -f2)" = "$UV_VERSION" ] && return 0
    log "fetching uv $UV_VERSION into $here/.uv/bin"
    # Into a scratch folder and renamed, so two first runs never write the same binary.
    fetch=$here/.uv/fetch-$$
    rm -rf "$fetch"
    mkdir -p "$here/.uv/bin"
    curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_UNMANAGED_INSTALL="$fetch" sh >&2 || true
    [ -x "$fetch/uv" ] && mv -f "$fetch/uv" "$uv"
    rm -rf "$fetch"
    [ "$("$uv" --version 2>/dev/null | cut -d' ' -f2)" = "$UV_VERSION" ] ||
        die "could not fetch uv $UV_VERSION; put a uv $UV_VERSION binary at $uv, or point EISGREIS_UV at one"
}

have_xlibs() {
    for soname in $XLIB_NAMES; do
        [ -f "$here/xlibs/$soname" ] || return 1
    done
}

fetch_xlibs() {
    case "$(uname -s)-$(uname -m)" in
        Linux-x86_64) ;;
        *) return 0 ;;
    esac
    have_xlibs && return 0
    for tool in curl unzip zstd tar; do
        command -v "$tool" >/dev/null 2>&1 || die "fetching the X libraries needs $tool"
    done
    log "fetching the five X libraries into $here/xlibs"
    tmp=$here/xlibs.tmp-$$
    rm -rf "$tmp"
    mkdir -p "$tmp/out"
    set -- $XLIBS
    while [ $# -ge 4 ]; do
        pkg=$1 sum=$2 file=$3 soname=$4
        shift 4
        curl -LsSf -o "$tmp/$pkg" "$CONDA_FORGE/$pkg" || die "could not fetch $CONDA_FORGE/$pkg"
        [ "$(sha256 "$tmp/$pkg")" = "$sum" ] || die "$pkg does not have the sha256 this script pins"
        mkdir "$tmp/x"
        (cd "$tmp/x" && unzip -q "../$pkg" 'pkg-*.tar.zst' && zstd -dq --stdout pkg-*.tar.zst | tar -x "lib/$file") ||
            die "could not unpack lib/$file from $pkg"
        cp "$tmp/x/lib/$file" "$tmp/out/$soname"
        rm -rf "$tmp/x" "$tmp/$pkg"
    done
    chmod -R a+rX "$tmp/out"
    rm -rf "$here/xlibs"
    mv "$tmp/out" "$here/xlibs"
    rm -rf "$tmp"
}

case ${1:-} in
    --path) printf '%s\n' "$env"; exit 0 ;;
    "") ;;
    *) die "unknown option $1 (see the top of this file)" ;;
esac

# A shared install is written by whoever runs this first: group-writable.
umask 002
fetch_xlibs
if [ -x "$env/bin/python3" ]; then
    printf '%s\n' "$env"
    exit 0
fi
if ! mkdir -p "$env_root" 2>/dev/null || [ ! -w "$env_root" ]; then
    die "no environment for this uv.lock ($name) and $env_root is not writable by you: ask whoever installed EisGreis to run $here/install.sh"
fi
ensure_uv

tmp=$env_root/$name.tmp-$$
cleanup() { rm -rf "$tmp"; }
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

log "building environment $name (about two minutes; viewers already running are not affected)"
# --no-bin: uv would otherwise put a python shim in ~/.local/bin, outside this checkout.
run_uv python install --no-bin "$python" >&2
run_uv venv --relocatable --python "$python" "$tmp" >&2
(cd "$here" && UV_PROJECT_ENVIRONMENT="$tmp" run_uv sync --locked --no-dev --python "$python" >&2)
"$tmp/bin/python3" -I -c "import eisgreis.api, eisgreis_crboost.protocol" ||
    die "the environment was built but EisGreis does not import from it"

{
    echo "name      $name"
    echo "python    $python"
    echo "uv        $("$uv" --version)"
    echo "lock sha  $(sha256 "$here/uv.lock")"
    echo "built     $(date -u +%Y-%m-%dT%H:%M:%SZ) by $(id -un)@$(hostname)"
    echo "checkout  $(git -C "$here" rev-parse --short HEAD 2>/dev/null || echo unknown)"
} > "$tmp/BUILD"
chmod -R a+rX "$tmp"

# One rename publishes it. If another run finished the same environment first, theirs stands and ours goes.
if mv -T "$tmp" "$env" 2>/dev/null; then
    log "built $name"
elif [ -x "$env/bin/python3" ]; then
    log "another run finished $name first; using it"
else
    die "could not move the finished build to $env"
fi
printf '%s\n' "$env"
