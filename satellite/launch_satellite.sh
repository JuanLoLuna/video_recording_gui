#!/usr/bin/env bash
# Start Thalamus on the SATELLITE computer (Linux).
# Python used, first match wins:
#   1. THALAMUS_PYTHON (full path to a python)
#   2. the conda environment THALAMUS_CONDA_ENV (default "thalamus")
#   3. the virtual environment THALAMUS_VENV (default ~/thalamus_venv)
set -euo pipefail
# Real GenTL producers are compiled libraries (ELF); Braille tables also end in .cti.
gentl_dirs() {
  find "$@" -xdev -name '*.cti' -type f 2>/dev/null | while read -r f; do
    [[ "$(head -c4 "$f" | od -An -c | tr -d ' ')" == '177ELF' ]] && dirname "$f"
  done | sort -u | paste -sd: -
}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${THALAMUS_PYTHON:-}"
if [[ -z "$PY" ]] && command -v conda >/dev/null 2>&1; then
  ENV_NAME="${THALAMUS_CONDA_ENV:-thalamus}"
  CAND="$(conda info --base)/envs/$ENV_NAME/bin/python"
  [[ -x "$CAND" ]] && PY="$CAND"
fi
if [[ -z "$PY" ]]; then
  PY="${THALAMUS_VENV:-$HOME/thalamus_venv}/bin/python"
fi
[[ -x "$PY" ]] || { echo "No Thalamus python found (tried $PY). Set THALAMUS_PYTHON." >&2; exit 1; }
echo "Using $PY"
# GENICAM node finds camera drivers through GENICAM_GENTL64_PATH (genicam_node.cpp).
if [[ -z "${GENICAM_GENTL64_PATH:-}" ]]; then
  CTI_DIRS="$(gentl_dirs /opt /usr/lib /usr/local/lib)"
  [[ -n "$CTI_DIRS" ]] && export GENICAM_GENTL64_PATH="$CTI_DIRS" && echo "GENICAM_GENTL64_PATH=$CTI_DIRS"
fi
exec "$PY" -m thalamus.pipeline -c "$HERE/configs/satellite_linux.json" --open -p 50060 -u 50061 -d 50062
