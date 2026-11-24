#!/usr/bin/env bash
# Start Thalamus on the SATELLITE computer (Linux), the way Jarl set it up on 2026-10-09:
# thalamus-contrib loaded (-C, gives the THORCAM node for the Thorlabs DCC1240 / IDS UI-124xSE-M),
# gRPC open to the network (--open) and encrypted with TLS client certificates.
#
# Settings (environment variables, all optional):
#   THALAMUS_PYTHON   python to use. Default: ~/venv-thalamus (Jarl's, Thalamus 1.0.53 + thalamus-contrib 0.0.12),
#                     else the conda env THALAMUS_CONDA_ENV (default "thalamus"), else ~/thalamus_venv.
#   THALAMUS_CONFIG   config file. Default: configs/satellite_thorcam.json next to this script.
#   THALAMUS_CERTS    folder with ca.pem, thalamus.internal.pem, thalamus.internal-key.pem. Default: ~/certs.
#                     If it is missing, Thalamus starts WITHOUT encryption and a warning is printed.
#   THALAMUS_PORT     gRPC port (default 50050, as Jarl used; user-interface port is PORT+1).
#   DRY_RUN=1         print the command instead of starting Thalamus.
#
# After it opens: tick Running on the node "thorcam1" yourself; Thalamus starts every node switched off.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Real GenTL producers are compiled libraries (ELF); Braille tables also end in .cti.
gentl_dirs() {
  find "$@" -xdev -name '*.cti' -type f 2>/dev/null | while read -r f; do
    [[ "$(head -c4 "$f" | od -An -c | tr -d ' ')" == '177ELF' ]] && dirname "$f"
  done | sort -u | paste -sd: -
}

PY="${THALAMUS_PYTHON:-}"
if [[ -z "$PY" && -x "$HOME/venv-thalamus/bin/python" ]]; then
  PY="$HOME/venv-thalamus/bin/python"
fi
if [[ -z "$PY" ]] && command -v conda >/dev/null 2>&1; then
  CAND="$(conda info --base)/envs/${THALAMUS_CONDA_ENV:-thalamus}/bin/python"
  [[ -x "$CAND" ]] && PY="$CAND"
fi
[[ -z "$PY" ]] && PY="${THALAMUS_VENV:-$HOME/thalamus_venv}/bin/python"
[[ -x "$PY" ]] || { echo "No Thalamus python found (tried $PY). Set THALAMUS_PYTHON." >&2; exit 1; }

CONFIG="${THALAMUS_CONFIG:-$HERE/configs/satellite_thorcam.json}"
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 1; }
PORT="${THALAMUS_PORT:-50050}"
CERTS="${THALAMUS_CERTS:-$HOME/certs}"

ARGS=(-m thalamus.pipeline -c "$CONFIG" --open -p "$PORT" -u "$((PORT + 1))" -d "$((PORT + 2))")
if "$PY" -c "import thalamus.contrib" >/dev/null 2>&1; then
  ARGS+=(-C)
else
  echo "WARNING: thalamus-contrib is not installed in $PY: no THORCAM node. Install with: $PY -m pip install thalamus-contrib" >&2
fi
if [[ -f "$CERTS/ca.pem" && -f "$CERTS/thalamus.internal.pem" && -f "$CERTS/thalamus.internal-key.pem" ]]; then
  ARGS+=(--ca "$CERTS/ca.pem" --cert "$CERTS/thalamus.internal.pem" --key "$CERTS/thalamus.internal-key.pem")
else
  echo "WARNING: certificates not found in $CERTS: starting WITHOUT encryption (receiver must also run without TLS)." >&2
fi

# GENICAM node (other camera types) finds drivers through GENICAM_GENTL64_PATH (genicam_node.cpp).
if [[ -z "${GENICAM_GENTL64_PATH:-}" ]]; then
  CTI_DIRS="$(gentl_dirs /opt /usr/lib /usr/local/lib)"
  [[ -n "$CTI_DIRS" ]] && export GENICAM_GENTL64_PATH="$CTI_DIRS"
fi

echo "Using $PY"
echo "Command: $PY ${ARGS[*]}"
[[ "${DRY_RUN:-0}" == "1" ]] && exit 0
cd "$HOME"   # Thalamus writes its log files (thalamus_*.log) in the current folder
exec "$PY" "${ARGS[@]}"
