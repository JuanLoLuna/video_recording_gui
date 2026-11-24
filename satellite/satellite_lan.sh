#!/usr/bin/env bash
# Wired link from the satellite to the receiving laptop (Ethernet cable, no internet on it).
#
#   satellite_lan.sh up      give the Ethernet port 192.168.50.10/24 (no gateway, Wi-Fi untouched)
#                            and open Thalamus's port 50050 in the firewall for 192.168.50.0/24 only
#   satellite_lan.sh down    undo both
#   satellite_lan.sh status  show the link, address, firewall rule and whether the laptop answers
#
# The laptop gets 192.168.50.20/24 on its Ethernet port (see README). sudo asks for this computer's password.
set -euo pipefail
CON=satellite-lan
ADDR=192.168.50.10/24
NET=192.168.50.0/24
PORT=50050
LAPTOP=192.168.50.20
IFACE="${SATELLITE_LAN_IFACE:-$(nmcli -t -f DEVICE,TYPE device | awk -F: '$2=="ethernet"{print $1; exit}')}"

status() {
  echo "Ethernet port:   ${IFACE:-none found} ($(nmcli -t -f DEVICE,STATE device | awk -F: -v d="$IFACE" '$1==d{print $2}'))"
  echo "Address:         $(ip -4 -br addr show "$IFACE" 2>/dev/null | awk '{print $3}')"
  nmcli -t -f NAME connection show | grep -qx "$CON" && echo "Connection:      $CON exists" || echo "Connection:      $CON absent"
  if sudo -n true 2>/dev/null; then
    sudo ufw status | grep -E "^$PORT/tcp .*$NET" >/dev/null && echo "Firewall:        port $PORT open for $NET" || echo "Firewall:        port $PORT not open for $NET"
  else
    echo "Firewall:        (run with sudo rights to check)"
  fi
  ping -c1 -W1 "$LAPTOP" >/dev/null 2>&1 && echo "Laptop $LAPTOP:  answers" || echo "Laptop $LAPTOP:  no answer"
  ss -ltn | grep -q ":$PORT " && echo "Thalamus:        listening on $PORT" || echo "Thalamus:        not running"
}

up() {
  [[ -n "$IFACE" ]] || { echo "No Ethernet port found." >&2; exit 1; }
  if ! nmcli -t -f NAME connection show | grep -qx "$CON"; then
    sudo nmcli connection add type ethernet ifname "$IFACE" con-name "$CON" \
      ipv4.method manual ipv4.addresses "$ADDR" ipv4.never-default yes ipv6.method disabled autoconnect yes >/dev/null
  fi
  sudo nmcli connection up "$CON" >/dev/null 2>&1 || echo "Cable not plugged in yet: the address is applied once it is."
  sudo ufw allow from "$NET" to any port "$PORT" proto tcp comment "Thalamus satellite" >/dev/null
  status
}

down() {
  nmcli -t -f NAME connection show | grep -qx "$CON" && sudo nmcli connection delete "$CON" >/dev/null
  sudo ufw delete allow from "$NET" to any port "$PORT" proto tcp >/dev/null 2>&1 || true
  status
}

case "${1:-}" in
  up) up ;; down) down ;; status) status ;;
  *) echo "Usage: $0 up|down|status"; exit 2 ;;
esac
