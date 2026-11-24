#!/usr/bin/env bash
# One-command camera check for the satellite (Linux).
# Finds IDS / Thorlabs drivers, GenTL (.cti) files and the camera on USB,
# writes everything to ~/satellite_check.txt and prints a short verdict.
# Usage: ~/repos/video_recording_gui/satellite/check_camera.sh
OUT="${1:-$HOME/satellite_check.txt}"
# Real GenTL producers are compiled libraries (ELF); Braille tables also end in .cti.
gentl_dirs() {
  find "$@" -xdev -name '*.cti' -type f 2>/dev/null | while read -r f; do
    [[ "$(head -c4 "$f" | od -An -c | tr -d ' ')" == '177ELF' ]] && dirname "$f"
  done | sort -u | paste -sd: -
}
{
  echo "== date / host"; date; hostname
  echo "== IDS / Thorlabs packages"; dpkg -l 2>/dev/null | grep -i -E 'ids|ueye|peak|thorlabs' || echo "(none)"
  echo "== programs"; compgen -c 2>/dev/null | grep -i -E 'peak|ueye|idscamera' | sort -u || true
  echo "== GENICAM_GENTL64_PATH"; echo "${GENICAM_GENTL64_PATH:-(empty)}"
  echo "== GenTL driver folders (.cti libraries)"; gentl_dirs / || true
  echo "== uEye / IDS services"; systemctl list-units --all 2>/dev/null | grep -i -E 'ueye|ids' || echo "(none)"
  echo "== camera on USB (IDS vendor 1409, Thorlabs 1313)"; lsusb | grep -E '1409:|1313:' || echo "(none)"
  echo "== video devices"; ls -l /dev/video* 2>&1
  echo "== Thalamus"
  if command -v conda >/dev/null 2>&1; then
    PY="$(conda info --base)/envs/${THALAMUS_CONDA_ENV:-thalamus}/bin/python"
    "$PY" -c "import thalamus, importlib.metadata as m; print('thalamus', m.version('thalamus_neuro'))" 2>&1
  else
    echo "conda not found"
  fi
} > "$OUT" 2>&1

CTI_DIRS="$(gentl_dirs /)"
echo "Report written to $OUT"
echo
lsusb | grep -q -E '1409:|1313:' && echo "OK   camera is on USB" || echo "FAIL camera not on USB: check cable/port"
grep -q -i 'ids-peak\|ids_peak' "$OUT" && echo "OK   IDS peak is installed" || echo "MISS IDS peak not found (only needed for the GenTL route)"
if [[ -n "$CTI_DIRS" ]]; then
  echo "OK   GenTL driver files in: $CTI_DIRS"
  [[ -z "${GENICAM_GENTL64_PATH:-}" ]] && echo "NOTE GENICAM_GENTL64_PATH is empty; launch_satellite.sh sets it from the folders above"
else
  echo "MISS no .cti file: Thalamus's GENICAM node cannot see the camera yet"
fi
echo
echo "Send $OUT to Juan's Claude session."
