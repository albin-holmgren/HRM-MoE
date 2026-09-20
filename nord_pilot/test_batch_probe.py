"""The batch probe decides what the paid run trains with, so its choice is pinned here."""
import json
import tempfile
import unittest
from pathlib import Path

from nord_pilot.batch_probe import DEFAULT_BATCH, choose_batch, steady_state_rate


class ChooseBatchTest(unittest.TestCase):
    def test_fastest_measured_candidate_wins(self):
        rows = [{'batch_tokens': 2048, 'tokens_per_second': 17167.0},
                {'batch_tokens': 8192, 'tokens_per_second': 61000.0},
                {'batch_tokens': 16384, 'tokens_per_second': 58000.0}]
        self.assertEqual(choose_batch(rows), 8192)

    def test_falls_back_to_the_validated_default_when_nothing_measured(self):
        rows = [{'batch_tokens': 8192, 'status': 'no_summary'},
                {'batch_tokens': 16384, 'status': 'timeout'}]
        self.assertEqual(choose_batch(rows), DEFAULT_BATCH)

    def test_the_default_is_kept_when_no_candidate_beats_it(self):
        rows = [{'batch_tokens': 2048, 'tokens_per_second': 17167.0},
                {'batch_tokens': 16384, 'tokens_per_second': 17167.0}]
        self.assertEqual(choose_batch(rows), 2048)


class SteadyStateRateTest(unittest.TestCase):
    def _write(self, rows):
        path = Path(tempfile.mkdtemp()) / 'metrics.jsonl'
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        return path

    def test_the_first_step_is_excluded_from_the_rate(self):
        # A slow warmup step must not drag the measured rate down, or a candidate would be
        # ranked by how much compilation it happened to absorb.
        path = self._write([{'seconds': 10.0, 'input_tokens': 1000},
                            {'seconds': 1.0, 'input_tokens': 2000},
                            {'seconds': 1.0, 'input_tokens': 2000}])
        self.assertEqual(steady_state_rate(path), 2000.0)

    def test_a_single_step_still_produces_a_rate(self):
        path = self._write([{'seconds': 2.0, 'input_tokens': 4000}])
        self.assertEqual(steady_state_rate(path), 2000.0)

    def test_no_steps_means_no_rate(self):
        path = self._write([])
        self.assertIsNone(steady_state_rate(path))


if __name__ == '__main__':
    unittest.main()
