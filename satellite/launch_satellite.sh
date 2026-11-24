#!/usr/bin/env bash
# Start Thalamus on the SATELLITE computer (Linux).
# Python used, first match wins:
#   1. THALAMUS_PYTHON (full path to a python)
#   2. the conda environment THALAMUS_CONDA_ENV (default "thalamus")
#   3. the virtual environment THALAMUS_VENV (default ~/thalamus_venv)
set -euo pipefail
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
exec "$PY" -m thalamus.pipeline -c "$HERE/configs/satellite_linux.json" --open -p 50060 -u 50061 -d 50062
