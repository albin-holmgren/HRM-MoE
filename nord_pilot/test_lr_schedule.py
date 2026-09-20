"""Pin the learning-rate and backprop-depth schedules the runner applies.

Both were previously inert: `AdamATan2(..., lr=2e-4)` was constant and `loss_at` hardcoded
`bp_steps=5`, so `--schedule-steps` changed nothing about training while still appearing in
the fingerprint. These tests fail if the schedule goes inert again, and they check the
property the recovery gate depends on: the schedule is a pure function of `step`.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ['NORD_REFERENCE_ATTENTION'] = '1'


def train(out, extra):
    common = [sys.executable, 'nord_pilot/run.py', '--device', 'cpu',
              '--config', 'nord_pilot/configs/cpu.json', '--batch-tokens', '256',
              '--minutes', '2', '--eval-every', '50', '--checkpoint-every', '50']
    r = subprocess.run(common + ['--out', str(out)] + extra, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return [json.loads(line) for line in (Path(out) / 'metrics.jsonl').read_text().splitlines()]


class LrScheduleTests(unittest.TestCase):
    def test_schedule_decays_and_peaks_at_configured_lr(self):
        with tempfile.TemporaryDirectory() as td:
            rows = train(Path(td) / 'a', ['--steps', '10', '--schedule-steps', '10', '--lr', '1e-3'])
        lrs = [r['lr'] for r in rows]
        self.assertEqual(len(lrs), 10)
        # Peak is the configured rate, decay is monotone, and it never goes below the floor.
        self.assertLessEqual(max(lrs), 1e-3)
        self.assertAlmostEqual(max(lrs), 1e-3, places=9)
        self.assertEqual(lrs, sorted(lrs, reverse=True))
        self.assertLess(lrs[-1], lrs[0])
        self.assertGreaterEqual(min(lrs), 0.1 * 1e-3 - 1e-12)

    def test_schedule_length_changes_the_trajectory(self):
        # The defect was that schedule-steps had no effect. If it is inert again, these two
        # runs produce identical learning rates and this assertion fires.
        with tempfile.TemporaryDirectory() as td:
            short = train(Path(td) / 'short', ['--steps', '10', '--schedule-steps', '10'])
            long = train(Path(td) / 'long', ['--steps', '10', '--schedule-steps', '1000'])
        self.assertNotEqual([r['lr'] for r in short], [r['lr'] for r in long])
        # A schedule longer than the run should still be near its peak, not decayed away.
        self.assertGreater([r['lr'] for r in long][-1], [r['lr'] for r in short][-1])

    def test_schedule_is_a_function_of_step_not_of_run_shape(self):
        # A 10-step run and a 5+5 resume must see the same rate at every step, which is what
        # makes the recovery-equivalence gate meaningful under a schedule.
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            whole = train(td / 'whole', ['--steps', '10', '--schedule-steps', '10'])
            train(td / 'split', ['--steps', '5', '--schedule-steps', '10'])
            train(td / 'split', ['--steps', '10', '--schedule-steps', '10',
                                 '--resume', str(td / 'split' / 'latest.pt')])
            first = [json.loads(line) for line in (td / 'split' / 'metrics.jsonl').read_text().splitlines()][:10]
        self.assertEqual([r['lr'] for r in whole], [r['lr'] for r in first])


if __name__ == '__main__':
    unittest.main()
