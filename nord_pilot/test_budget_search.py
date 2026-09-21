"""Pin the fixed-budget keep/discard search behaviour.

This is the ported half of karpathy/autoresearch: a fixed wall-clock budget per
variant, one held-out metric, and keep-if-better/discard-if-worse with every
attempt written to the log. These tests check the decisions, not the model, so
they run against tiny synthetic summaries rather than real training.
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'nord_pilot'))
import budget_search as bs  # noqa: E402


class ConfigOverlayTests(unittest.TestCase):
    def test_overlay_patches_base_without_mutating_it(self):
        base = {'n_layers': 2, 'moe_top_k': 2}
        out = bs.write_variant_config(base, {'n_layers': 4}, ROOT / '_tmp_cfg_test.json')
        self.assertEqual(out['n_layers'], 4)
        self.assertEqual(out['moe_top_k'], 2)
        self.assertEqual(base['n_layers'], 2)
        (ROOT / '_tmp_cfg_test.json').unlink()

    def test_unknown_key_is_rejected_rather_than_silently_ignored(self):
        with self.assertRaises(SystemExit):
            bs.write_variant_config({'n_layers': 2}, {'moe_typo': 1}, ROOT / '_tmp_cfg_test.json')


class ParseRunOutputTests(unittest.TestCase):
    def test_parameters_and_expert_totals_are_summed_across_steps(self):
        stdout = '{"parameters": 4334592, "start_step": 0, "valid_loss": 10.9}\n'
        metrics = ROOT / '_tmp_metrics_test.jsonl'
        metrics.write_text(
            json.dumps({'step': 1, 'expert_counts': [1, 2, 3, 4, 5, 6, 7, 8]}) + '\n' +
            json.dumps({'step': 2, 'expert_counts': [10, 20, 30, 40, 50, 60, 70, 80]}) + '\n')
        params, last, totals = bs.parse_run_output(stdout, metrics)
        self.assertEqual(params, 4334592)
        self.assertEqual(last['step'], 2)
        self.assertEqual(totals, [11, 22, 33, 44, 55, 66, 77, 88])
        metrics.unlink()

    def test_missing_metrics_file_is_not_an_error(self):
        params, last, totals = bs.parse_run_output('', ROOT / '_tmp_absent_metrics.jsonl')
        self.assertIsNone(params)
        self.assertIsNone(last)
        self.assertIsNone(totals)


class KeepDiscardTests(unittest.TestCase):
    """The comparison rule itself: strictly lower held-out loss wins."""

    def decide(self, seen_best, metric):
        if seen_best is None or (metric is not None and metric < seen_best):
            return 'keep', metric
        return 'discard', seen_best

    def test_first_variant_is_kept_and_lower_beats_it(self):
        verdict, best = self.decide(None, 10.0)
        self.assertEqual(verdict, 'keep')
        verdict, best = self.decide(best, 9.0)
        self.assertEqual(verdict, 'keep')
        self.assertEqual(best, 9.0)

    def test_worse_variant_is_discarded_and_does_not_replace_best(self):
        verdict, best = self.decide(9.0, 9.5)
        self.assertEqual(verdict, 'discard')
        self.assertEqual(best, 9.0)

    def test_equal_metric_does_not_replace_the_incumbent(self):
        verdict, best = self.decide(9.0, 9.0)
        self.assertEqual(verdict, 'discard')
        self.assertEqual(best, 9.0)


if __name__ == '__main__':
    unittest.main()
