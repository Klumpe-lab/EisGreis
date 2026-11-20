# The launchers' common part, sourced by ./eisgreis and ./eisgreis-crboost with $here (this checkout) and $command
# (the environment's script to run) set. It runs the command from the environment install.sh built, with nothing the
# shell loaded in the way, and so that Qt starts in a remote desktop:
#
# - PYTHONPATH, PYTHONHOME and QT_PLUGIN_PATH are cleared, so cluster modules and other environments stay out;
# - LD_LIBRARY_PATH is filtered, not cleared: a GPU desktop's GL arrangement stays (its glvnd folder and
#   /.singularity.d/libs, where apptainer's --nv puts the NVIDIA libraries), every other entry goes;
# - xlibs/ goes on LD_LIBRARY_PATH only when the system lacks one of the five X libraries Qt needs (ldconfig -p);
# - on a full GPU (nvidia-smi names a GPU and no MIG device) with a display, the command runs under VirtualGL
#   (vglrun -d egl) and draws on the card; elsewhere Mesa draws in software. EISGREIS_VGL=no turns VirtualGL off.

env=$("$here/install.sh" --path) || exit 1
if [ ! -x "$env/bin/$command" ]; then
    echo "$command: no environment for this checkout's uv.lock ($env); run $here/install.sh" >&2
    exit 1
fi

unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE QT_PLUGIN_PATH VIRTUAL_ENV CONDA_PREFIX

keep=
old_ifs=$IFS
IFS=:
for dir in ${LD_LIBRARY_PATH:-}; do
    case $dir in
        */glvnd | */glvnd/ | /.singularity.d/libs | /.singularity.d/libs/) keep=${keep:+$keep:}$dir ;;
    esac
done
IFS=$old_ifs

if [ "$(uname -s)" = Linux ]; then
    cache=$( (ldconfig -p || /sbin/ldconfig -p || /usr/sbin/ldconfig -p) 2>/dev/null)
    for lib in libxcb-cursor.so.0 libxcb-icccm.so.4 libxcb-image.so.0 libxcb-keysyms.so.1 libxcb-render-util.so.0; do
        if ! printf '%s\n' "$cache" | grep -qF "$lib ("; then
            if [ ! -f "$here/xlibs/$lib" ]; then
                echo "$command: the system lacks $lib and $here/xlibs has none; run $here/install.sh" >&2
                exit 1
            fi
            keep=$here/xlibs${keep:+:$keep}
            break
        fi
    done
fi
if [ -n "$keep" ]; then
    export LD_LIBRARY_PATH="$keep"
else
    unset LD_LIBRARY_PATH
fi

vgl=
if [ -n "${DISPLAY:-}" ] && [ "${EISGREIS_VGL:-auto}" != no ] && command -v nvidia-smi >/dev/null 2>&1; then
    gpus=$(nvidia-smi -L 2>/dev/null || true)
    if printf '%s\n' "$gpus" | grep -q '^GPU ' && ! printf '%s\n' "$gpus" | grep -q 'MIG'; then
        vgl=$(command -v vglrun 2>/dev/null || true)
        [ -n "$vgl" ] || { [ -x /opt/VirtualGL/bin/vglrun ] && vgl=/opt/VirtualGL/bin/vglrun; }
    fi
fi
if [ -n "$vgl" ]; then
    exec "$vgl" -d egl "$env/bin/$command" "$@"
fi
exec "$env/bin/$command" "$@"
