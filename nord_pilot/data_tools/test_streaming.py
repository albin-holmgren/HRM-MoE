"""Tests for the line-indexed streaming corpus reader.

The reader replaced an in-memory list of every record. Its value depends on three
properties that a careless rewrite would break silently: offsets must identify the same
records as splitting the file, the training order must be reproducible from the seed, and
the digest must match a plain file hash so resume fingerprints keep rejecting changed data.
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from nord_pilot.data_tools.streaming import JsonlCorpus, index_lines_and_digest


def tokenizer():
    tok = Tokenizer(models.BPE(unk_token='[UNK]'))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel()
    tok.train_from_iterator([' '.join(f'word{i} text here' for i in range(200))],
                            trainers.BpeTrainer(vocab_size=512,
                                                special_tokens=['[PAD]', '[UNK]', '[BOS]', '[SEP]', '[EOS]'],
                                                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                                show_progress=False))
    return tok


class StreamingTests(unittest.TestCase):
    def write(self, root, name, rows, trailing_newline=True):
        path = root / name
        text = ''.join(json.dumps(r) + '\n' for r in rows)
        if not trailing_newline and text.endswith('\n'):
            text = text[:-1]
        path.write_text(text)
        return path

    def rows(self, count, end=True):
        return [{'id': i, 'kind': 'pretrain', 'document_end': end,
                 'token_ids': [10 + (i % 50), 20 + (i % 50), 30]} for i in range(count)]

    def corpus(self, root, rows, seed=7):
        path = self.write(root, 'train.jsonl', rows)
        self.write(root, 'valid.jsonl', rows[:4])
        build = JsonlCorpus({'train': path, 'valid': root / 'valid.jsonl'}, tokenizer(), 64, 512, seed)
        self.addCleanup(build.close)
        return build

    def test_offsets_match_line_splitting(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows = self.rows(25)
            path = self.write(root, 'train.jsonl', rows)
            offsets, digest = index_lines_and_digest(path)
            self.assertEqual(len(offsets), 25)
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
            data = path.read_bytes()
            for position, offset in enumerate(offsets):
                line = data[offset:data.index(b'\n', offset)]
                self.assertEqual(json.loads(line)['id'], position)

    def test_trailing_newline_does_not_add_a_record(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = self.write(root, 'train.jsonl', self.rows(5))
            with_newline, _ = index_lines_and_digest(path)
            truncated = self.write(root, 'truncated.jsonl', self.rows(5), trailing_newline=False)
            without_newline, _ = index_lines_and_digest(truncated)
            self.assertEqual(len(with_newline), 5)
            self.assertEqual(len(without_newline), 5)

    def test_training_order_is_reproducible_from_seed(self):
        with tempfile.TemporaryDirectory() as td:
            first = self.corpus(Path(td), self.rows(30), seed=11)
            second = self.corpus(Path(td), self.rows(30), seed=11)
            third = self.corpus(Path(td), self.rows(30), seed=12)
            same = [first.record('train', i) for i in range(30)]
            self.assertEqual(same, [second.record('train', i) for i in range(30)])
            self.assertNotEqual(same, [third.record('train', i) for i in range(30)])

    def test_validation_keeps_file_order(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows = self.rows(8)
            path = self.write(root, 'valid.jsonl', rows)
            corpus = JsonlCorpus({'valid': path}, tokenizer(), 64, 512, seed=3)
            self.addCleanup(corpus.close)
            self.assertTrue(np.array_equal(corpus.order['valid'], np.arange(8)))

    def test_every_training_record_is_visited_exactly_once(self):
        with tempfile.TemporaryDirectory() as td:
            corpus = self.corpus(Path(td), self.rows(40))
            seen = [corpus.order['train'][i] for i in range(40)]
            self.assertEqual(sorted(int(v) for v in seen), list(range(40)))

    def test_record_matches_direct_encoding(self):
        with tempfile.TemporaryDirectory() as td:
            corpus = self.corpus(Path(td), self.rows(12))
            rows = self.rows(12)
            for position in range(12):
                index = int(corpus.order['train'][position])
                prefix, answer = corpus.record('train', position)
                bos = corpus.tokenizer.token_to_id('[BOS]')
                eos = corpus.tokenizer.token_to_id('[EOS]')
                self.assertEqual(prefix, [bos])
                self.assertEqual(answer, rows[index]['token_ids'] + [eos])

    def test_position_wraps_within_the_split(self):
        with tempfile.TemporaryDirectory() as td:
            corpus = self.corpus(Path(td), self.rows(6))
            self.assertEqual(corpus.record('train', 0), corpus.record('train', 6))
            self.assertEqual(corpus.record('train', 3), corpus.record('train', 9))


if __name__ == '__main__':
    unittest.main()
