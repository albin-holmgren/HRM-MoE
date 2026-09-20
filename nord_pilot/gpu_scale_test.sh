#!/usr/bin/env bash
# One-GPU scaled technical run on the built real corpus. Technical pass only.
#
# Runs on the paid Verda VM; it provisions nothing by itself. Its job is to answer the two
# questions the $500 plan actually rests on, cheaply and in this order:
#   1. Kernel gates still pass on this image (FA3 prefix attention + Triton experts).
#   2. The validator loss improves on held-out documents the model never trained on.
# The second is the gate that separates "the pipeline trains" from "the number is real".
#
# Usage: gpu_scale_test.sh <out-dir> <config> <corpus-dir> <minutes> <steps> <subprocess-minutes>
set -euo pipefail
cd "$(dirname "$0")/.."
export WANDB_MODE=disabled HF_HUB_OFFLINE=1
# The paid GPU must exercise native FA3, not the reference attention the CPU tests use.
unset NORD_REFERENCE_ATTENTION
out=${1:?Pass a fresh output directory}
config=${2:?Pass a config under nord_pilot/configs}
data=${3:?Pass the built corpus directory}
# A 25-minute window at this scale is roughly 15-25M tokens, i.e. thousands of steps, so the
# step cap is set above what the clock allows and the wall clock does the stopping. The
# learning-rate schedule is sized to the expected step count instead.
minutes=${4:-25}
steps=${5:-20000}
schedule_steps=${7:-8000}
child_minutes=${6:-4}
test ! -e "$out" || { echo 'Refusing to overwrite a run'; exit 2; }
test -f "$config"
test -f "$data/tokenizer.json" && test -f "$data/train.jsonl" && test -f "$data/valid.jsonl" && test -f "$data/test-fresh.jsonl"
mkdir -p "$out"

python -m nord_pilot.gpu_gate | tee "$out/kernel-gate.json"

# Recovery equivalence is checked first, at short duration, so a resume bug costs seconds
# rather than a full billed run. 20 steps vs 10+10 must still agree exactly.
gate=(--device cuda --config "$config" --data-dir "$data" --batch-tokens 2048 --minutes "$child_minutes" \
  --schedule-steps "$schedule_steps" --eval-every 5 --checkpoint-every 10 --early-stop-patience 0 --valid-records 256)
python nord_pilot/run.py "${gate[@]}" --out "$out/full" --steps 20 --save-initial-export "$out/initial-export.pt" | tee "$out/full.log"
python nord_pilot/run.py "${gate[@]}" --out "$out/resumed" --steps 10 | tee "$out/first-half.log"
python nord_pilot/run.py "${gate[@]}" --out "$out/resumed" --steps 20 --resume "$out/resumed/latest.pt" | tee "$out/resume.log"
python -m nord_pilot.compare "$out/full/latest.pt" "$out/resumed/latest.pt" --device cuda | tee "$out/recovery-gate.json"

# Full schedule. It initializes from the same step-0 weights the baseline holds, so the
# before/after comparison is against exactly the weights this run started from instead of a
# half-trained checkpoint.
# eval-every is 250 steps so validation is a small fraction of the wall clock rather than a
# steady tax, and the final held-out score is bounded separately below.
train=(--device cuda --config "$config" --data-dir "$data" --batch-tokens 2048 --minutes "$minutes" \
  --schedule-steps "$schedule_steps" --eval-every 250 --checkpoint-every 500 --early-stop-patience 4 --min-delta .005 \
  --valid-records 1024 --init-export "$out/initial-export.pt")
python nord_pilot/run.py "${train[@]}" --out "$out/trained" --steps "$steps" | tee "$out/learning.log"

python nord_pilot/infer.py --device cuda --data-dir "$data" --export "$out/trained/best-export.pt" --tokens 48 | tee "$out/inference.json"

# Bounded held-out scoring: a partial pass reaches roughly +-0.003 nats on the loss while
# scoring the full 151M-token split would cost hours per model on one GPU.
python -m nord_pilot.evaluate_corpus --data-dir "$data" --test-file test-fresh.jsonl \
  --initial-export "$out/initial-export.pt" --trained-export "$out/trained/best-export.pt" \
  --test-records 12000 --disjoint-limit 2000000 --out "$out/fresh-test-evaluation.json" | tee "$out/evaluation.log"

python - "$out" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
summary=json.loads((p/'trained/summary.json').read_text())
manifest=json.loads((p/'trained/manifest.json').read_text())
evaluation=json.loads((p/'fresh-test-evaluation.json').read_text())
initial=manifest['initial_valid_loss']
assert summary['status'] in ('completed','early_stopped'), summary['status']
assert summary['best_valid_loss']<initial, 'No validation improvement over untrained initialization'
assert evaluation[1]['test_loss']<evaluation[0]['test_loss'], 'Held-out loss did not improve over untrained initialization'
record={'scaled_real_corpus_technical_run':'pass','parameters':manifest['parameters'],'steps_run':summary['step'],
 'initial_valid_loss':initial,'best_valid_loss':summary['best_valid_loss'],'best_step':summary['best_step'],
 'fresh_test_loss':{'untrained_initialization':evaluation[0]['test_loss'],'trained':evaluation[1]['test_loss']},
 'fresh_test_records_scored':evaluation[1]['test_records_scored'],'leakage_check':evaluation[1]['leakage_check'],
 'capability':'not established','claim':'loss reduction on held-out real text; not an assistant-quality or benchmark claim'}
(p/'PASS.json').write_text(json.dumps(record,indent=2))
print('SCALED REAL-CORPUS TECHNICAL RUN PASSED. General capability not established.')
PY
