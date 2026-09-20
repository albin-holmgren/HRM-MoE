#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export WANDB_MODE=disabled HF_HUB_OFFLINE=1
unset NORD_REFERENCE_ATTENTION
out=${1:?Pass a fresh output directory}
parent=${2:?Pass the trusted step-600 export}
test ! -e "$out" || { echo 'Refusing to overwrite a run'; exit 2; }
test -f "$parent"
mkdir -p "$out"
python -m nord_pilot.gpu_gate | tee "$out/kernel-gate.json"
common=(--device cuda --config nord_pilot/configs/gpu-real-69m.json --data-dir data-long --batch-tokens 2048 --minutes 35 --schedule-steps 9000 --eval-every 1000 --early-stop-patience 3 --min-delta .005 --checkpoint-every 1000 --init-export "$parent")
python nord_pilot/run.py "${common[@]}" --out "$out/full" --steps 20 | tee "$out/full.log"
python nord_pilot/run.py "${common[@]}" --out "$out/resumed" --steps 10 | tee "$out/first-half.log"
python nord_pilot/run.py "${common[@]}" --out "$out/resumed" --steps 20 --resume "$out/resumed/latest.pt" | tee "$out/resume.log"
python -m nord_pilot.compare "$out/full/latest.pt" "$out/resumed/latest.pt" --device cuda | tee "$out/recovery-gate.json"
python nord_pilot/run.py "${common[@]}" --out "$out/resumed" --steps 9000 --resume "$out/resumed/latest.pt" | tee "$out/learning.log"
python nord_pilot/infer.py --device cuda --data-dir data-long --export "$out/resumed/best-export.pt" --tokens 48 | tee "$out/inference.json"
python -m nord_pilot.evaluate_corpus --data-dir data-long --test-file test-fresh.jsonl --initial-export "$parent" --trained-export "$out/resumed/best-export.pt" --out "$out/fresh-test-evaluation.json" | tee "$out/evaluation.log"
python - "$out" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);s=json.loads((p/'resumed/summary.json').read_text());evaluation=json.loads((p/'fresh-test-evaluation.json').read_text())
assert s['status'] in ('completed','early_stopped')
initial=json.loads((p/'full/manifest.json').read_text())['initial_valid_loss']
assert s['best_valid_loss']<initial and evaluation[1]['test_loss']<evaluation[0]['test_loss']
(p/'PASS.json').write_text(json.dumps({'long_real_data_technical_pilot':'pass','initial_valid_loss':initial,'best_valid_loss':s['best_valid_loss'],'fresh_test_loss':[evaluation[0]['test_loss'],evaluation[1]['test_loss']],'capability':'not established'},indent=2))
print('LONG REAL-DATA TECHNICAL TEST PASSED. General capability not established.')
PY
