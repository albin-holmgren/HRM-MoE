#!/usr/bin/env bash
# One-GPU scaled technical run on the built real corpus. Technical pass only.
#
# Runs on the paid Verda VM; it provisions nothing by itself. Its job is to answer the two
# questions the $500 plan actually rests on, cheaply and in this order:
#   1. Kernel gates still pass on this image (FA3 prefix attention + Triton experts).
#   2. The validator loss improves on held-out documents the model never trained on.
# The second is the gate that separates "the pipeline trains" from "the number is real".
#
# Usage: gpu_scale_test.sh <out-dir> <config> <corpus-dir> <minutes> [steps] [subprocess-minutes] [schedule-steps] [parent-export] [peak-lr] [seed]
# Set NORD_PLAN_ONLY=1 to print the exact trainer invocations instead of running them. The
# plan is built here rather than restated elsewhere, so a replay of this mode is a check of
# the real thing: each printed command is fed back through the trainer's own --check-only
# validator on a CPU machine before any GPU is provisioned.
set -euo pipefail
cd "$(dirname "$0")/.."
export WANDB_MODE=disabled HF_HUB_OFFLINE=1
# The paid GPU must exercise native FA3, not the reference attention the CPU tests use.
unset NORD_REFERENCE_ATTENTION
out=${1:?Pass a fresh output directory}
config=${2:?Pass a config under nord_pilot/configs}
data=${3:?Pass the built corpus directory}
# A 25-minute window at this scale is thousands of steps now that the batch probe lifted the
# rate, so the step cap is set above what the clock allows and the wall clock does the
# stopping. schedule-steps sizes the linear-warmup cosine the runner actually applies; set it
# near the step count you expect the clock to reach, since a schedule longer than the run
# leaves the final steps at a high learning rate.
minutes=${4:-25}
steps=${5:-20000}
schedule_steps=${7:-8000}
child_minutes=${6:-4}
# A trusted parent export turns this into a weights-only continuation: every invocation below
# initializes from those weights instead of random, so the paid run extends an existing lineage
# rather than starting a new one. run.py verifies the parent config and tokenizer hash before
# loading it, and the held-out comparison below then reads parent-versus-continued.
parent=${8:-}
# Held as one string and appended with word splitting rather than as a possibly-empty array:
# this script also runs under macOS bash 3.2, where expanding an empty array in a quoted
# "${array[@]}" position raises "unbound variable" under set -u.
init_args=""
[[ -n "$parent" ]] && init_args="--init-export $parent"
# A continuation starts from weights that already sit at the floor of an earlier cosine, so it
# restarts the schedule from a lower peak than a fresh run. Re-entering at the fresh-run peak
# would step the model back up the loss curve it just descended, spending paid steps undoing
# progress instead of adding it. Empty means use the runner default (2e-4).
lr=${9:-}
lr_args=""
[[ -n "$lr" ]] && lr_args="--lr $lr"
# The training order is a seeded permutation of the corpus, so a second run on the same corpus
# with the same seed would replay the records the first run already consumed. Changing the seed
# keeps the run reproducible while sampling a different slice of the same bounded prefix.
seed=${10:-42}
seed_args="--seed $seed"
# Early stopping is measured against the loss this run started from. With a parent that
# baseline is a trained checkpoint, and a restart at the peak learning rate legitimately
# spends its first hundred-plus steps above that baseline before descending through it, so
# patience would end the paid session at its first four checks. Continuations therefore run
# patience-free: the step cap and the wall-clock cap still bound them, and the best export and
# the parent-versus-continued comparison below still decide whether the run produced evidence.
patience=4
[[ -n "$parent" ]] && patience=0

# Recovery equivalence is checked first, at short duration, so a resume bug costs seconds
# rather than a full billed run. 20 steps vs 10+10 must still agree exactly. The schedule is
# longer than these caps, so the caps are what stop them.
gate=(--device cuda --config "$config" --data-dir "$data" --batch-tokens 2048 --minutes "$child_minutes" \
  --schedule-steps "$schedule_steps" --eval-every 5 --checkpoint-every 10 --early-stop-patience 0 --valid-records 256)
gate+=($init_args $lr_args $seed_args)

# Batch size is the one knob that converts the wall-clock budget directly into learning. This
# run is stopped by the clock, so input tokens per second is what decides how many updates the
# money buys, and the earlier pilot measured the rate at only 0.87% of H100 peak because one
# maximum-length record per step leaves the GPU launch-bound. A larger batch should lift that
# rate, but it has to be measured on the hardware rather than assumed, so the probe runs first
# and trains the real run with whatever it selects. The 2048 batch the pilot already validated
# is the fallback if every candidate fails.
gate_batch=2048
train_batch=$gate_batch

# Full schedule. Without a parent it initializes from the same step-0 weights the baseline
# holds, so the before/after comparison is against exactly the weights this run started from
# rather than a half-trained checkpoint. With a parent it continues those weights, and the
# initial export written before the first update is the parent itself, so the comparison is
# parent-versus-continued.
# eval-every is 250 steps so validation is a small fraction of the wall clock rather than a
# steady tax, and the final held-out score is bounded separately below.
# The step cap is the ceiling the wall clock is not expected to reach; schedule-steps sets how
# far the cosine decays and should track the step count the minute budget actually reaches.
plan=(
  "python nord_pilot/batch_probe.py --config $config --data-dir $data --out $out/batch-probe --steps 8 --minutes 0.15 --child-timeout 60 --schedule-steps $schedule_steps"
  "python nord_pilot/run.py ${gate[*]} --out $out/full --steps 20 --save-initial-export $out/initial-export.pt"
  "python nord_pilot/run.py ${gate[*]} --out $out/resumed --steps 10"
  "python nord_pilot/run.py ${gate[*]} --out $out/resumed --steps 20 --resume $out/resumed/latest.pt"
  "python nord_pilot/run.py --device cuda --config $config --data-dir $data --batch-tokens $train_batch --minutes $minutes --schedule-steps $schedule_steps --eval-every 250 --checkpoint-every 500 --early-stop-patience $patience --min-delta .005 --valid-records 1024 --init-export ${parent:-$out/initial-export.pt} $lr_args $seed_args --out $out/trained --steps $steps"
)
if [[ "${NORD_PLAN_ONLY:-0}" == 1 ]]; then
  printf '%s\n' "${plan[@]}"
  exit 0
fi

test ! -e "$out" || { echo 'Refusing to overwrite a run'; exit 2; }
[[ -z "$parent" ]] || test -f "$parent"
test -f "$config"
test -f "$data/tokenizer.json" && test -f "$data/train.jsonl" && test -f "$data/valid.jsonl" && test -f "$data/test-fresh.jsonl"
mkdir -p "$out"

python -m nord_pilot.gpu_gate | tee "$out/kernel-gate.json"

# Measured, then read back. The probe logs every candidate, so the batch actually used is
# visible in the artifacts rather than only in the choice it produced.
python nord_pilot/batch_probe.py --config "$config" --data-dir "$data" --out "$out/batch-probe" \
  --steps 8 --minutes 0.15 --child-timeout 60 --schedule-steps "$schedule_steps" \
  | tee "$out/batch-probe.log"
if [[ -f "$out/batch-probe/chosen-batch.txt" ]]; then
  train_batch=$(cat "$out/batch-probe/chosen-batch.txt")
fi
echo "Training batch size: $train_batch"

train=(--device cuda --config "$config" --data-dir "$data" --batch-tokens "$train_batch" --minutes "$minutes" \
  --schedule-steps "$schedule_steps" --eval-every 250 --checkpoint-every 500 --early-stop-patience "$patience" --min-delta .005 \
  --valid-records 1024 --init-export "${parent:-$out/initial-export.pt}")
train+=($lr_args $seed_args)

python nord_pilot/run.py "${gate[@]}" --out "$out/full" --steps 20 --save-initial-export "$out/initial-export.pt" | tee "$out/full.log"
python nord_pilot/run.py "${gate[@]}" --out "$out/resumed" --steps 10 | tee "$out/first-half.log"
python nord_pilot/run.py "${gate[@]}" --out "$out/resumed" --steps 20 --resume "$out/resumed/latest.pt" | tee "$out/resume.log"
python -m nord_pilot.compare "$out/full/latest.pt" "$out/resumed/latest.pt" --device cuda | tee "$out/recovery-gate.json"

# The wall clock is expected to stop this run before the step cap, and the runner reports that
# as 'bounded_stop' with exit code 3. That is the designed outcome of a money-bounded run, not
# a failure, so it is captured here instead of aborting the script under set -e. Everything the
# gate needs is still written on that path: summary, best export, and held-out evaluation.
set +e
python nord_pilot/run.py "${train[@]}" --out "$out/trained" --steps "$steps" | tee "$out/learning.log"
train_rc=${PIPESTATUS[0]}
set -e
if [[ "$train_rc" != 0 && "$train_rc" != 3 ]]; then
  echo "Training stopped for a reason other than its own clock budget (exit $train_rc)"
  exit "$train_rc"
fi

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
# A wall-clock-bounded run is the expected shape here: the step cap sits above what the minute
# budget can reach on purpose. 'bounded_stop' is therefore accepted as a legitimate stop, and
# the two assertions below are what actually decide whether the run produced evidence.
assert summary['status'] in ('completed','early_stopped','bounded_stop'), summary['status']
assert summary['best_valid_loss']<initial, 'No validation improvement over untrained initialization'
assert evaluation[1]['test_loss']<evaluation[0]['test_loss'], 'Held-out loss did not improve over untrained initialization'
record={'scaled_real_corpus_technical_run':'pass','stop_reason':summary['status'],'parameters':manifest['parameters'],'steps_run':summary['step'],
 'initial_valid_loss':initial,'best_valid_loss':summary['best_valid_loss'],'best_step':summary['best_step'],
 'fresh_test_loss':{'untrained_initialization':evaluation[0]['test_loss'],'trained':evaluation[1]['test_loss']},
 'fresh_test_records_scored':evaluation[1]['test_records_scored'],'leakage_check':evaluation[1]['leakage_check'],
 'capability':'not established','claim':'loss reduction on held-out real text; not an assistant-quality or benchmark claim'}
(p/'PASS.json').write_text(json.dumps(record,indent=2))
print('SCALED REAL-CORPUS TECHNICAL RUN PASSED. General capability not established.')
PY
