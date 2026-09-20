#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
unset NORD_REFERENCE_ATTENTION
out=${1:?Pass a fresh output directory on retained storage}
test ! -e "$out" || { echo 'Refusing to overwrite a run'; exit 2; }
mkdir -p "$out"
python -m nord_pilot.gpu_gate | tee "$out/kernel-gate.json"
common=(--device cuda --config nord_pilot/configs/gpu.json --batch-tokens 2048 --minutes 20)
python nord_pilot/run.py "${common[@]}" --out "$out/full" --steps 40 | tee "$out/full.log"
python nord_pilot/run.py "${common[@]}" --out "$out/resumed" --steps 20 | tee "$out/first-half.log"
python nord_pilot/run.py "${common[@]}" --out "$out/resumed" --steps 40 --resume "$out/resumed/latest.pt" | tee "$out/resume.log"
python -m nord_pilot.compare "$out/full/latest.pt" "$out/resumed/latest.pt" --device cuda | tee "$out/recovery-gate.json"
python nord_pilot/infer.py --device cuda --export "$out/resumed/export.pt" | tee "$out/inference.json"
python nord_pilot/run.py "${common[@]}" --out "$out/long-context" --steps 10 --stress-context | tee "$out/long-context.log"
python nord_pilot/run.py "${common[@]}" --out "$out/resumed" --steps 200 --resume "$out/resumed/latest.pt" | tee "$out/learning.log"
python - "$out" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);s=json.loads((p/'resumed/summary.json').read_text())
assert s['status']=='completed' and s['last_train_loss']<s['first_train_loss'], 'No learning on technical fixture'
assert s['final_valid_loss']<s['initial_valid_loss'], 'No validation loss reduction on fixture'
print('TECHNICAL PILOT PASSED. This is not a capability benchmark or production model.')
(p/'PASS.json').write_text(json.dumps({'technical_pilot':'pass','capability':'not assessed'},indent=2))
PY
