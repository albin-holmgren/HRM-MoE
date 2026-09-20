"""Deterministic streaming access to JSONL pretraining corpora.

The training runner used to read train.jsonl and valid.jsonl into Python lists and encode
every record before the first step. That is fine for a fixture of a few hundred rows and
impossible for a real corpus: train.jsonl is tens of gigabytes and holds millions of
chunks, so the process exhausts memory before training starts. This indexes line offsets
in one pass, keeps a seeded permutation of the training order, and decodes a record only
when a batch asks for it. Resident memory stays flat and the batch sequence stays
reproducible, because the order comes from the seed rather than from file order.
"""
import hashlib
import json
from array import array
from pathlib import Path

import numpy as np

from nord_pilot.data_tools.records import encode_record


def index_lines_and_digest(path, block=1 << 22):
    """Byte offsets of each line, plus the file's sha256, in a single read.

    The digest is needed anyway for the resume fingerprint, so it is computed during the
    same pass that finds the line starts instead of re-reading the file.
    """
    offsets = array('q', [0])
    accumulator = hashlib.sha256()
    position = 0
    with Path(path).open('rb') as handle:
        while True:
            chunk = handle.read(block)
            if not chunk:
                break
            accumulator.update(chunk)
            start = 0
            while True:
                found = chunk.find(b'\n', start)
                if found < 0:
                    break
                offsets.append(position + found + 1)
                start = found + 1
            position += len(chunk)
    # A file ending in a newline leaves an offset equal to its size, which would be a
    # record with no bytes. A final line without a newline is a real record and stays.
    if len(offsets) > 1 and offsets[-1] == Path(path).stat().st_size:
        offsets.pop()
    return np.array(offsets, dtype=np.int64), accumulator.hexdigest()


class JsonlCorpus:
    """Line-indexed, lazily decoded view of one or more JSONL splits."""

    def __init__(self, paths, tokenizer, max_seq_len, vocab_size, seed, shuffled=('train',)):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.paths = {name: Path(path) for name, path in paths.items()}
        self.offsets, self.digests, self.handles, self.order = {}, {}, {}, {}
        for name, path in self.paths.items():
            offsets, digest = index_lines_and_digest(path)
            self.offsets[name] = offsets
            self.digests[name] = digest
            # Training order is seeded, so a run and its resume see the same batches. Other
            # splits stay in file order so evaluation is comparable across steps.
            count = len(offsets)
            generator = np.random.default_rng(seed)
            self.order[name] = (generator.permutation(count) if name in shuffled
                                else np.arange(count)).astype(np.int64)
        self.handles = {name: path.open('rb') for name, path in self.paths.items()}

    def __len__(self):
        return len(self.paths)

    def count(self, name):
        return int(len(self.offsets[name]))

    def record(self, name, position):
        """Encoded (prefix, answer) for one record, reading only that line."""
        count = self.count(name)
        index = int(self.order[name][position % count])
        handle = self.handles[name]
        handle.seek(int(self.offsets[name][index]))
        row = json.loads(handle.readline())
        return encode_record(row, self.tokenizer, self.max_seq_len, self.vocab_size)

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.handles = {}
