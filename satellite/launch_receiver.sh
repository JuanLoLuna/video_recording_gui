#!/usr/bin/env bash
# Start Thalamus on the receiving computer (Linux). Set THALAMUS_VENV to override.
# Ports 50070-50072 avoid the Sleeve clock service on 50050.
# -C loads thalamus-contrib (needed for the satellite's H264 stream).
# Certificates: copy ca.pem, thalamus.internal.pem, thalamus.internal-key.pem from the satellite's
# ~/certs into satellite/certs (git ignores that folder). The satellite requires TLS client certificates.
set -euo pipefail
THALAMUS_VENV="${THALAMUS_VENV:-$HOME/thalamus_venv}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CERTS="${THALAMUS_CERTS:-$HERE/certs}"
exec "$THALAMUS_VENV/bin/python" -m thalamus.pipeline -C -c "$HERE/configs/receiver.json" -p 50070 -u 50071 -d 50072 \
  --ca "$CERTS/ca.pem" --cert "$CERTS/thalamus.internal.pem" --key "$CERTS/thalamus.internal-key.pem"
