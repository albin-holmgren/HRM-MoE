"""Regression tests for the streaming held-out evaluator.

`evaluate_corpus.py` used to read the held-out split and the train/valid leak check with
`read_text().splitlines()`. The rehearsal corpus was about a megabyte, so that passed
locally, but on the built corpus train.jsonl is 18 GB and the evaluator would exhaust
memory before scoring a single batch on a paid GPU. These tests pin the streaming
behavior so the in-memory shape cannot come back unnoticed.

Only the pure row helpers are exercised here; `main()` needs a Hopper GPU and is covered
by the GPU run itself.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Importing the evaluator pulls in the model tree, which imports the native FA3 module
# unless reference attention is selected. The helpers under test never touch it.
os.environ.setdefault('NORD_REFERENCE_ATTENTION', '1')

from nord_pilot.evaluate_corpus import assert_disjoint, stream


def write(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return path


class StreamingEvaluatorTests(unittest.TestCase):
    def test_stream_reads_every_row_without_loading_the_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = write(Path(td) / 'test.jsonl',
                         [{'doc_id': f'd{i}', 'token_ids': [i]} for i in range(500)])
            rows = list(stream(path))
            self.assertEqual(len(rows), 500)
            self.assertEqual(rows[-1]['doc_id'], 'd499')

    def test_stream_skips_blank_lines(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'test.jsonl'
            path.write_text('{"doc_id":"a"}\n\n{"doc_id":"b"}\n')
            self.assertEqual([r['doc_id'] for r in stream(path)], ['a', 'b'])

    def test_disjoint_split_passes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            train = write(root / 'train.jsonl', [{'doc_id': f't{i}'} for i in range(200)])
            test_ids = {f'x{i}' for i in range(20)}
            assert_disjoint(test_ids, train)

    def test_shared_document_id_fails(self):
        with tempfile.TemporaryDirectory() as td:
            train = write(Path(td) / 'train.jsonl', [{'doc_id': 't1'}, {'doc_id': 'leaked'}])
            with self.assertRaises(ValueError):
                assert_disjoint({'leaked'}, train)

    def test_limit_reports_a_bounded_spot_check(self):
        """A capped inspection must say so instead of claiming a full audit."""
        with tempfile.TemporaryDirectory() as td:
            train = write(Path(td) / 'train.jsonl', [{'doc_id': f't{i}'} for i in range(500)])
            report = assert_disjoint(set(), train, limit=25)
            self.assertEqual(report, {'records_inspected': 25, 'complete': False})

    def test_full_inspection_is_marked_complete(self):
        with tempfile.TemporaryDirectory() as td:
            train = write(Path(td) / 'train.jsonl', [{'doc_id': f't{i}'} for i in range(7)])
            report = assert_disjoint(set(), train)
            self.assertEqual(report, {'records_inspected': 7, 'complete': True})

    def test_leak_inside_the_inspected_window_still_fails(self):
        with tempfile.TemporaryDirectory() as td:
            train = write(Path(td) / 'train.jsonl', [{'doc_id': 'leaked'}] + [{'doc_id': f't{i}'} for i in range(500)])
            with self.assertRaises(ValueError):
                assert_disjoint({'leaked'}, train, limit=10)


if __name__ == '__main__':
    unittest.main()
