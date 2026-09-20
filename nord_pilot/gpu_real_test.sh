#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export WANDB_MODE=disabled HF_HUB_OFFLINE=1
unset NORD_REFERENCE_ATTENTION
out=${1:?Pass a fresh output directory}
test ! -e "$out" || { echo 'Refusing to overwrite a run'; exit 2; }
mkdir -p "$out"
python -m nord_pilot.gpu_gate | tee "$out/kernel-gate.json"
common=(--device cuda --config data-real/gpu-real.json --data-dir data-real --batch-tokens 2048 --minutes 20 --schedule-steps 600 --eval-every 100 --early-stop-patience 3 --checkpoint-every 100)
python nord_pilot/run.py "${common[@]}" --out "$out/full" --steps 20 | tee "$out/full.log"
python nord_pilot/run.py "${common[@]}" --out "$out/resumed" --steps 10 | tee "$out/first-half.log"
python nord_pilot/run.py "${common[@]}" --out "$out/resumed" --steps 20 --resume "$out/resumed/latest.pt" | tee "$out/resume.log"
python -m nord_pilot.compare "$out/full/latest.pt" "$out/resumed/latest.pt" --device cuda | tee "$out/recovery-gate.json"
python nord_pilot/run.py "${common[@]}" --out "$out/resumed" --steps 600 --resume "$out/resumed/latest.pt" | tee "$out/learning.log"
python nord_pilot/infer.py --device cuda --data-dir data-real --export "$out/resumed/best-export.pt" --tokens 48 | tee "$out/inference.json"
python - "$out" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);s=json.loads((p/'resumed/summary.json').read_text())
assert s['status'] in ('completed','early_stopped')
initial=json.loads((p/'full/manifest.json').read_text())['initial_valid_loss']
assert s['best_valid_loss']<initial, 'No validation improvement over initialization'
assert s['final_valid_loss']<s['initial_valid_loss'], 'No validation improvement during continuation'
(p/'PASS.json').write_text(json.dumps({'real_data_technical_pilot':'pass','initial_valid_loss':initial,'best_valid_loss':s['best_valid_loss'],'capability':'not established'},indent=2))
print('REAL-DATA TECHNICAL TEST PASSED. General capability not established.')
PY
