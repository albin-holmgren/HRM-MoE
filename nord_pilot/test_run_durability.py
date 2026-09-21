"""Pin the crash-durability properties the third scaled run did not have.

That run trained for 3400 steps, measured a real improvement over its parent, and then died
out of memory. The loss numbers were downloaded; the weights were not, because best-export.pt
was only written on the path after the training loop had already exited normally. These tests
hold the replacement behaviour in place from both directions: the math the run depends on is
unchanged, and an interrupted run still leaves usable weights behind.
"""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

os.environ['NORD_REFERENCE_ATTENTION'] = '1'
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent


class SlicedCrossEntropyTest(unittest.TestCase):
    def test_slicing_the_loss_changes_nothing_but_the_allocation(self):
        # The slicing exists to cap the largest transient in a step. It is only safe because the
        # value it returns is the same one the whole-batch reduction produced, so this compares
        # them directly rather than assuming the reduction is associative in floating point.
        torch.manual_seed(0)
        logits = torch.randn(20000, 512, dtype=torch.bfloat16)
        labels = torch.randint(0, 512, (20000,))
        labels[::7] = -100
        reference = F.cross_entropy(logits.float(), labels, ignore_index=-100)
        total = None
        targets = None
        for start in range(0, logits.shape[0], 8192):
            piece, target = logits[start:start + 8192], labels[start:start + 8192]
            part = F.cross_entropy(piece.float(), target, ignore_index=-100, reduction='sum')
            total = part if total is None else total + part
            count = (target != -100).sum()
            targets = count if targets is None else targets + count
        self.assertAlmostEqual(float(reference), float(total / targets), places=5)


class InterruptedRunTest(unittest.TestCase):
    def test_an_interrupted_run_still_writes_weights_and_a_summary(self):
        # The failure mode that cost the session: a run that stops early keeps its metrics but
        # loses its weights. SIGTERM is the portable stand-in for any stop that is not the step
        # cap, and it must still leave an export, a best export and a summary on disk.
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / 'interrupted'
            command = [sys.executable, 'nord_pilot/run.py', '--device', 'cpu',
                       '--config', 'nord_pilot/configs/cpu.json', '--batch-tokens', '256',
                       '--out', str(out), '--steps', '4000', '--schedule-steps', '4000',
                       '--eval-every', '2', '--checkpoint-every', '4', '--minutes', '5']
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True)
            # Wait for validation to have recorded something, so the interruption lands after
            # the run has a best checkpoint to preserve rather than before it starts.
            deadline = time.time() + 120
            while time.time() < deadline:
                metrics = out / 'metrics.jsonl'
                if metrics.exists() and len(metrics.read_text().splitlines()) >= 6:
                    break
                if process.poll() is not None:
                    break
                time.sleep(0.5)
            process.send_signal(signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=180)
            self.assertIsNotNone(process.returncode, stdout + stderr)
            best = out / 'best-export.pt'
            self.assertTrue(best.exists(), f'no best export after interruption: {stdout} {stderr}')
            payload = torch.load(best, map_location='cpu', weights_only=False)
            summary = json.loads((out / 'summary.json').read_text())
            initial_valid = json.loads((out / 'manifest.json').read_text())['initial_valid_loss']
            # The export has to carry the best weights the run actually found, not the initial
            # ones and not an empty file, or the interruption has cost the session its model.
            self.assertGreater(payload['step'], 0)
            self.assertEqual(payload['step'], summary['best_step'])
            self.assertAlmostEqual(payload['validation_loss'], summary['best_valid_loss'], places=9)
            self.assertLess(summary['best_valid_loss'], initial_valid)
            self.assertTrue((out / 'latest-weights.pt').exists(),
                            'the weights-only checkpoint was not written')
            self.assertEqual(summary['status'], 'bounded_stop')
            self.assertLess(summary['step'], 4000)


if __name__ == '__main__':
    unittest.main()
