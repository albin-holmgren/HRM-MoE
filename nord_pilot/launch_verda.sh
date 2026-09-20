#!/usr/bin/env bash
# Run only on the dedicated Verda test VM. Does not provision/pay for anything.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ${1:-} != --execute ]]; then
  echo 'Dry run: arm independent Verda API guard, pull pinned image, run GPU gates, retain artifacts, delete VM retaining volumes.'
  echo 'Execution requires: --execute INSTANCE_UUID HOSTNAME PRIVATE_CREDENTIALS_JSON STORAGE_USD_PER_HOUR TAX_FRACTION'
  exit 0
fi
[[ $# == 6 ]] || { echo 'Incorrect argument count'; exit 2; }
[[ $EUID == 0 ]] || { echo 'Run as root on the dedicated test VM'; exit 2; }
instance=$2
host_name=$3
credentials=$(realpath "$4")
storage_hourly=$5
tax_fraction=$6
command -v systemd-run >/dev/null
command -v docker >/dev/null
command -v python3 >/dev/null
# Require credentials outside the repo/container mount.
case "$credentials" in "$PWD"/*) echo 'Place credentials outside the source tree'; exit 2;; esac
state="/var/lib/nord-pilot/$instance"
[[ ! -e "$state" ]] || { echo 'Previous guard state exists; review before a new run'; exit 2; }
mkdir -p "$state"
chmod 700 "$state"
unit="nord-hrm-guard-$instance"
systemd-run --unit="$unit" --property=Restart=on-failure --property=RestartSec=10 \
  /usr/bin/python3 "$PWD/nord_pilot/verda_guard.py" --execute --instance "$instance" --hostname "$host_name" \
  --credentials "$credentials" --state-dir "$state" --storage-hourly "$storage_hourly" --tax-fraction "$tax_fraction"
for _ in $(seq 1 30); do
  [[ -f "$state/armed.json" ]] && break
  sleep 1
done
[[ -f "$state/armed.json" ]] && systemctl is-active --quiet "$unit" || {
  echo 'Guard failed to arm. No GPU work launched. Delete this VM retaining volumes through Verda console.'; exit 2;
}
trap 'touch "$state/finished"' EXIT
image=$(cat nord_pilot/image.txt)
# Timeout includes pull time; billing watchdog already runs independently.
timeout 1800 docker pull "$image"
mkdir -p pilot_runs
run="pilot_runs/$(date -u +%Y%m%dT%H%M%SZ)"
printf '%s\n' "$run" > "$state/output-path.txt"
timeout --signal=TERM --kill-after=60 5400 docker run --rm --name nord-hrm-pilot-test \
  --gpus '"device=0"' --shm-size=4g --network=none \
  -v "$PWD:/workspace" -w /workspace "$image" \
  bash nord_pilot/gpu_test.sh "$run" > "$state/container.log" 2>&1
# Results stay on the retained OS volume, including if the watchdog interrupts us.
sync
