#!/usr/bin/env bash
# Turn remote help from Juan's Claude session on cortex on or off.
#
#   remote_access.sh up      allow it and open the tunnel (window stays open; Ctrl+C closes the tunnel)
#   remote_access.sh down    close everything again and undo every change
#   remote_access.sh status  show what is on
#
# What "up" changes on this computer (all undone by "down"):
#   1. /etc/hosts.allow: first line "sshd: 127.0.0.1 [::1] : ALLOW" (SSH only from this computer itself;
#      IT's rules below it stay as they are). A backup is kept in /etc/hosts.allow.before-remote-access.
#   2. ~/.ssh/authorized_keys: cortex's public key (line ending in "cortex-juan-claude-setup").
#   3. SSH server on (installed if missing).
# Then it opens a reverse tunnel: cortex localhost:22022 -> this computer's SSH.
# Your cortex password is typed into ssh itself; this script never reads or stores it.
set -euo pipefail

CORTEX_USER="${CORTEX_USER:-juan}"
CORTEX_HOST="cortex.ophanim.org"
PORT=22022
KEY_ID="AAAAC3NzaC1lZDI1NTE5AAAAIGfJTibSAD6bBr+Xv7QRybbjII2YjmJZs69q5jLDAePt"
CORTEX_KEY="ssh-ed25519 $KEY_ID cortex-juan-claude-setup"
ALLOW_LINE="sshd: 127.0.0.1 [::1] : ALLOW"
ALLOW_FILE=/etc/hosts.allow

allow_line_present() { grep -qxF "$ALLOW_LINE" "$ALLOW_FILE" 2>/dev/null; }

remove_allow_lines() {
  # Our line, plus the two earlier hand-added variants.
  sudo sed -i -e '/^sshd: 127\.0\.0\.1 \[::1\]\( : ALLOW\)\?$/d' "$ALLOW_FILE"
}

status() {
  echo "SSH server:       $(systemctl is-active ssh 2>/dev/null || true) (socket: $(systemctl is-active ssh.socket 2>/dev/null || true))"
  if allow_line_present && [[ "$(head -1 "$ALLOW_FILE")" == "$ALLOW_LINE" ]]; then
    echo "hosts.allow:      local SSH allowed (first line)"
  elif allow_line_present; then
    echo "hosts.allow:      line present but NOT first (will not work)"
  else
    echo "hosts.allow:      local SSH not allowed"
  fi
  grep -q "$KEY_ID" ~/.ssh/authorized_keys 2>/dev/null && echo "cortex key:       present" || echo "cortex key:       absent"
  pgrep -u "$USER" -f "ssh .*-R $PORT:localhost:22" >/dev/null && echo "tunnel:           open" || echo "tunnel:           closed"
}

up() {
  echo "Turning remote access ON (sudo may ask for this computer's password)."
  # 1. SSH server
  if ! dpkg -s openssh-server >/dev/null 2>&1; then
    sudo apt-get install -y openssh-server
  fi
  sudo systemctl enable --now ssh.socket ssh >/dev/null 2>&1 || sudo systemctl enable --now ssh
  # 2. hosts.allow: our line first
  [[ -f "$ALLOW_FILE.before-remote-access" ]] || sudo cp -p "$ALLOW_FILE" "$ALLOW_FILE.before-remote-access"
  remove_allow_lines
  sudo sed -i "1i $ALLOW_LINE" "$ALLOW_FILE"
  # 3. cortex key
  mkdir -p ~/.ssh && chmod 700 ~/.ssh
  touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
  grep -q "$KEY_ID" ~/.ssh/authorized_keys || echo "$CORTEX_KEY" >> ~/.ssh/authorized_keys
  # Local check before opening the tunnel
  if timeout 10 bash -c 'exec 3<>/dev/tcp/127.0.0.1/22 && head -c 4 <&3' 2>/dev/null | grep -q SSH; then
    echo "Local SSH check:  OK"
  else
    echo "Local SSH check:  FAILED (server not answering). Run: sudo journalctl -u ssh -n 20" >&2
    exit 1
  fi
  echo
  echo "Opening tunnel to $CORTEX_HOST as $CORTEX_USER (this computer's user: $(whoami))."
  echo "Type your cortex password if asked. Leave this window open; Ctrl+C closes the tunnel."
  echo "When finished: $0 down"
  exec ssh -N -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes \
    -R "$PORT:localhost:22" "$CORTEX_USER@$CORTEX_HOST"
}

down() {
  echo "Turning remote access OFF (sudo may ask for this computer's password)."
  pkill -u "$USER" -f "ssh .*-R $PORT:localhost:22" 2>/dev/null && echo "Tunnel closed." || true
  [[ -f ~/.ssh/authorized_keys ]] && sed -i "/$KEY_ID/d" ~/.ssh/authorized_keys
  remove_allow_lines
  sudo systemctl disable --now ssh ssh.socket >/dev/null 2>&1 || true
  echo
  status
}

case "${1:-}" in
  up) up ;;
  down) down ;;
  status) status ;;
  *) echo "Usage: $0 up|down|status"; exit 2 ;;
esac
