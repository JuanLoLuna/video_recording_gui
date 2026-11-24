#!/usr/bin/env bash
# Find how this computer was already set up to stream video (read-only).
# Writes ~/existing_setup.txt and prints a short summary.
# Usage: ~/repos/video_recording_gui/satellite/find_existing_setup.sh
OUT="${1:-$HOME/existing_setup.txt}"
{
  echo "== host / user"; hostname; whoami; date
  echo "== uEye install"; dpkg -l 2>/dev/null | grep -i -E 'ueye|ids' | awk '{print $2, $3}'
  echo "== uEye services"; systemctl list-units --all 2>/dev/null | grep -i -E 'ueye|ids'
  echo "== uEye cameras"; (command -v ueyesetid >/dev/null && timeout 10 ueyesetid -l) 2>&1 | head -20
  echo "== conda envs"; (conda env list 2>/dev/null || ls ~/miniforge3/envs ~/miniconda3/envs ~/anaconda3/envs 2>/dev/null)
  echo "== Thalamus installs (site-packages)"; find / -xdev -type d -path '*site-packages/thalamus' 2>/dev/null
  echo "== Thalamus source checkouts"; find / -xdev -maxdepth 5 -type d -iname 'thalamus*' -not -path '*/site-packages/*' 2>/dev/null | head -20
  echo "== Thalamus / camera configs (json with \"nodes\")"
  find /home /opt /srv -xdev -name '*.json' -size -2M 2>/dev/null | xargs -r grep -l '"nodes"' 2>/dev/null | head -20
  echo "== node types in those configs"
  find /home /opt /srv -xdev -name '*.json' -size -2M 2>/dev/null | xargs -r grep -l '"nodes"' 2>/dev/null | head -20 \
    | xargs -r grep -h -o -E '"type": *"[A-Z0-9_]+"' 2>/dev/null | sort | uniq -c
  echo "== ueye / camera python code"; find /home /opt -xdev -name '*.py' -size -1M 2>/dev/null | xargs -r grep -l -i -E 'pyueye|ueye|is_InitCamera|v4l2loopback' 2>/dev/null | grep -v site-packages | head -20
  echo "== python packages with ueye/genicam"; find / -xdev -type d \( -name 'pyueye*' -o -name 'harvesters*' -o -name 'ids_peak*' \) 2>/dev/null | head
  echo "== v4l2loopback"; lsmod | grep -i v4l2loopback; ls /etc/modules-load.d /etc/modprobe.d 2>/dev/null | grep -i v4l
  echo "== autostart / services / shortcuts"; ls ~/.config/autostart /etc/xdg/autostart 2>/dev/null | grep -v -i -E 'gnome|ubuntu|org\.|snap|tracker|xdg|print|ibus|geoclue|spice|pulseaudio|user-dirs|at-spi'
  ls ~/Desktop ~/.local/share/applications 2>/dev/null
  systemctl list-unit-files --type=service 2>/dev/null | grep -i -E 'thalamus|camera|stream|ueye'
  echo "== shell history mentioning thalamus / ueye / camera (all users readable)"
  for h in /home/*/.bash_history /home/*/.zsh_history /root/.bash_history; do
    [[ -r "$h" ]] && grep -a -i -E 'thalamus|ueye|genicam|ffmpeg|v4l2|camera|pyueye|peak' "$h" | sed "s#^#$h: #" | tail -40
  done
  echo "== files changed 2026-10-01..2026-10-15 in home folders (top 60)"
  find /home -xdev -type f -newermt 2026-10-01 ! -newermt 2026-10-16 -not -path '*/.cache/*' -not -path '*/site-packages/*' -not -path '*/.mozilla/*' 2>/dev/null | head -60
} > "$OUT" 2>&1
echo "Report written to $OUT ($(wc -l < "$OUT") lines)"
grep -A3 '== Thalamus source checkouts' "$OUT" | tail -3
grep -A8 '== node types' "$OUT" | tail -8
echo "Send $OUT to Juan's Claude session."
