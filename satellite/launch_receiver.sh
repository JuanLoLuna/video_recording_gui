#!/usr/bin/env bash
# Start Thalamus on the receiving computer (Linux). Set THALAMUS_VENV to override.
# Ports 50070-50072 avoid the Sleeve clock service on 50050.
set -euo pipefail
THALAMUS_VENV="${THALAMUS_VENV:-$HOME/thalamus_venv}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$THALAMUS_VENV/bin/python" -m thalamus.pipeline -c "$HERE/configs/receiver.json" -p 50070 -u 50071 -d 50072
