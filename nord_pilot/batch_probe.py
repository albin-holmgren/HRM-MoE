"""Measure on-GPU throughput at several batch sizes before the scaled run commits to one.

The scaled run is bounded by wall clock, so input tokens per second is what decides how much
learning the money buys. The 2026-09-20 pilot measured 17,167.84 input tokens/s at
--batch-tokens 2048 and the cost model records that as 0.87% of H100 bf16 peak: the model
applies 32 sequential block passes per token, so at one maximum-length record per step it is
launch-bound rather than arithmetic-bound. A larger batch should lift the rate, but that is a
measurement, not an assumption, so it is measured here on the GPU that is already billing.

Nothing here can make a run worse. Every candidate is the same trainer with a different batch,
the default is the batch the earlier pilot actually validated, and a candidate that fails or
out-of-memory is simply never chosen. --check-only replays every candidate invocation through
the trainer's own validator and touches no GPU.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_BATCH = 2048


def invocation(python, config, data, out, batch, steps, minutes, schedule_steps):
    """The trainer call one candidate batch size is measured with."""
    return [python, str(ROOT / 'run.py'), '--device', 'cuda', '--config', str(config),
            '--data-dir', str(data), '--out', str(out), '--batch-tokens', str(batch),
            '--steps', str(steps), '--minutes', str(minutes),
            '--schedule-steps', str(schedule_steps),
            # Validation is switched off inside the measured window: this probe measures step
            # time, and an evaluation in the middle would be attributed to the batch size.
            '--eval-every', '100000', '--checkpoint-every', '100000',
            '--early-stop-patience', '0', '--valid-records', '8']


def choose_batch(rows, default=DEFAULT_BATCH):
    """Fastest measured batch size wins; rows with no measurement are ignored."""
    measured = [row for row in rows if row.get('tokens_per_second')]
    if not measured:
        return default
    return int(max(measured, key=lambda row: row['tokens_per_second'])['batch_tokens'])


def steady_state_rate(metrics_path):
    """Pooled tokens/s over every step after the first, or None if none was recorded.

    The first step of a run pays for CUDA context setup and any kernel compilation the Triton
    cache has not already stored. Attributing that to the batch size would rank candidates by
    how much warmup they happened to absorb, and the first candidate run always absorbs the
    most because it warms the shared cache for the rest. Dropping the first step measures the
    rate the training loop actually sustains. If only one step fits the window its timing is
    still used, because a single step is better evidence than none.
    """
    rows = [json.loads(line) for line in Path(metrics_path).read_text().splitlines() if line.strip()]
    if not rows:
        return None
    steady = rows[1:] or rows
    seconds = sum(row['seconds'] for row in steady)
    if seconds <= 0:
        return None
    return sum(row['input_tokens'] for row in steady) / seconds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--candidates', default='2048,8192,16384,32768')
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--minutes', type=float, default=0.5)
    ap.add_argument('--schedule-steps', type=int, default=8000)
    ap.add_argument('--child-timeout', type=float, default=180.0)
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--check-only', action='store_true',
                    help='Validate every candidate invocation with the trainer and exit')
    a = ap.parse_args()
    candidates = [int(part) for part in a.candidates.split(',') if part.strip()]
    if not candidates:
        raise SystemExit('No batch-size candidates given')
    if a.check_only:
        failures = 0
        for batch in candidates:
            argv = invocation(a.python, a.config, a.data_dir, a.out / f'probe-{batch}', batch,
                              a.steps, a.minutes, a.schedule_steps)
            proc = subprocess.run(argv + ['--check-only'], capture_output=True, text=True,
                                  cwd=ROOT.parent)
            ok = proc.returncode == 0
            print(f"{'OK    ' if ok else 'FAILED'} probe --batch-tokens {batch}")
            if not ok:
                print('   ', (proc.stderr or proc.stdout).strip().splitlines()[-1][:300])
                failures += 1
        print(json.dumps({'probe_candidates': candidates,
                          'validated': len(candidates) - failures, 'failures': failures}))
        raise SystemExit(1 if failures else 0)
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for batch in candidates:
        run_dir = a.out / f'probe-{batch}'
        if run_dir.exists():
            # Never reuse a directory: an earlier probe's numbers would be read as this one's.
            rows.append({'batch_tokens': batch, 'status': 'skipped_existing_directory'})
            continue
        argv = invocation(a.python, a.config, a.data_dir, run_dir, batch, a.steps, a.minutes,
                          a.schedule_steps)
        started = time.monotonic()
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, cwd=ROOT.parent,
                                  timeout=a.child_timeout)
            code, note = proc.returncode, None
        except subprocess.TimeoutExpired:
            # subprocess.run kills the child on timeout, so a hung candidate cannot bill on.
            code, note = None, 'timeout'
        row = {'batch_tokens': batch, 'exit_code': code,
               'seconds': round(time.monotonic() - started, 2)}
        metrics_path = run_dir / 'metrics.jsonl'
        if metrics_path.exists():
            row['steady_state_tokens_per_second'] = steady_state_rate(metrics_path)
        summary_path = run_dir / 'summary.json'
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            peak = summary['peak_memory_bytes']
            # The chosen rate is the steady-state one when it exists, so a candidate is not
            # rewarded for warmup it absorbed on behalf of the candidates that ran after it.
            row.update({'steps_run': summary['step'], 'status': summary['status'],
                        'reported_tokens_per_second': summary['tokens_per_training_second'],
                        'tokens_per_second': row.get('steady_state_tokens_per_second')
                            or summary['tokens_per_training_second'],
                        'peak_memory_gb': round(peak / 2 ** 30, 2) if peak else None})
        else:
            # The usual cause is an out-of-memory abort, which leaves no summary behind.
            row.update({'status': note or 'no_summary'})
        rows.append(row)
        print(json.dumps(row), flush=True)
    chosen = choose_batch(rows)
    (a.out / 'batch-probe.json').write_text(json.dumps(
        {'chosen_batch_tokens': chosen, 'validated_default': DEFAULT_BATCH,
         'measurements': rows}, indent=2))
    (a.out / 'chosen-batch.txt').write_text(str(chosen))
    print(json.dumps({'chosen_batch_tokens': chosen}))


if __name__ == '__main__':
    main()
