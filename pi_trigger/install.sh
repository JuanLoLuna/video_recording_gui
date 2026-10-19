#!/bin/bash
# Install the camera trigger service on a Raspberry Pi 4 (Raspberry Pi OS).
#   sudo bash install.sh
# Run it from the folder that holds pi_trigger_server.py and pi-trigger.service.
# Safe to run again. Boot files are backed up (*.bak-pi-trigger) before any change.
# Everything it prints is also saved to install-report.txt next to this script.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORT="$HERE/install-report.txt"
exec > >(tee "$REPORT") 2>&1

if [[ $EUID -ne 0 ]]; then echo "Run with sudo: sudo bash $0"; exit 1; fi
BOOT=/boot/firmware; [[ -d $BOOT ]] || BOOT=/boot
CONFIG=$BOOT/config.txt
CMDLINE=$BOOT/cmdline.txt
echo "== pi-trigger install $(date -Is) on $(hostname), boot files in $BOOT"

# --- config.txt: hardware PWM on GPIO18, analog audio off, USB gadget; in the final [all] section
cp -n "$CONFIG" "$CONFIG.bak-pi-trigger" || true
last_section=$(grep -E '^\[' "$CONFIG" | tail -1 || true)
if [[ "$last_section" != "[all]" ]]; then
  printf '\n[all]\n' >> "$CONFIG"; echo "config.txt: added a final [all] section"
fi
for line in "dtoverlay=dwc2,dr_mode=peripheral" "dtoverlay=pwm,pin=18,func=2" "dtparam=audio=off"; do
  if grep -qxF "$line" "$CONFIG"; then echo "config.txt: already has $line"
  else echo "$line" >> "$CONFIG"; echo "config.txt: added $line"; fi
done
if grep -qxF "dtparam=audio=on" "$CONFIG"; then
  sed -i 's/^dtparam=audio=on$/dtparam=audio=off/' "$CONFIG"; echo "config.txt: audio on -> off"
fi

# --- cmdline.txt (ONE line): load the serial gadget with TWO ports, login console on the first
cp -n "$CMDLINE" "$CMDLINE.bak-pi-trigger" || true
cmd="$(head -n1 "$CMDLINE")"
for word in "modules-load=dwc2,g_serial" "g_serial.n_ports=2" "systemd.wants=serial-getty@ttyGS0.service"; do
  if [[ " $cmd " == *" $word "* ]]; then echo "cmdline.txt: already has $word"
  else cmd="$cmd $word"; echo "cmdline.txt: added $word"; fi
done
printf '%s\n' "$cmd" > "$CMDLINE"
echo "cmdline.txt now: $cmd"

# --- the service
install -d /opt/pi-trigger /var/log/pi-trigger
install -m 0755 "$HERE/pi_trigger_server.py" /opt/pi-trigger/pi_trigger_server.py
install -m 0644 "$HERE/pi-trigger.service" /etc/systemd/system/pi-trigger.service
systemctl daemon-reload
systemctl enable pi-trigger.service
systemctl restart pi-trigger.service || true
sleep 1
systemctl --no-pager --lines=5 status pi-trigger.service || true

echo
echo "== checks"
ls /sys/class/pwm/ || true
ls -l /dev/ttyGS* 2>/dev/null || echo "no /dev/ttyGS* yet: they appear after the reboot (n_ports=2)"
echo
echo "Done. Reboot now:  sudo reboot"
echo "After the reboot the host sees TWO USB serial ports: the first is the login, the second is the trigger."
echo "Report saved to $REPORT"
