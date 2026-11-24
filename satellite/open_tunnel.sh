#!/usr/bin/env bash
# Let Juan's Claude session on cortex log in to this computer for setup help.
# Opens a reverse SSH tunnel: cortex localhost:22022 -> this computer's SSH (port 22).
# Close it any time with Ctrl+C. Nothing stays open after that.
# Usage: ~/repos/video_recording_gui/satellite/open_tunnel.sh [cortex user]
set -euo pipefail
CORTEX_USER="${1:-juan}"
CORTEX_HOST="cortex.ophanim.org"
PORT=22022
CORTEX_KEY="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGfJTibSAD6bBr+Xv7QRybbjII2YjmJZs69q5jLDAePt cortex-juan-claude-setup"

# 1. This computer must run an SSH server.
if ! systemctl is-active --quiet ssh 2>/dev/null; then
  echo "Installing/starting the SSH server (asks for your password)..."
  sudo apt-get install -y openssh-server
  sudo systemctl enable --now ssh
fi

# 2. Allow cortex's key to log in here (only the public key; remove the line later to revoke).
mkdir -p ~/.ssh && chmod 700 ~/.ssh
touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
grep -q "AAAAC3NzaC1lZDI1NTE5AAAAIGfJTibSAD6bBr+Xv7QRybbjII2YjmJZs69q5jLDAePt" ~/.ssh/authorized_keys \
  || echo "$CORTEX_KEY" >> ~/.ssh/authorized_keys

# 3. Open the tunnel. Asks for the cortex password of $CORTEX_USER.
echo
echo "Opening tunnel to $CORTEX_HOST as $CORTEX_USER (user here: $(whoami))."
echo "Leave this window open while Claude works. Press Ctrl+C to close."
exec ssh -N -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes \
  -R "$PORT:localhost:22" "$CORTEX_USER@$CORTEX_HOST"
