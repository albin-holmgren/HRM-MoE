"""Tests for the scaled corpus builder's dedup estimator and filter admission.

These run offline. The near-duplicate estimator is the part most likely to be silently
wrong, because a broken filter still produces a plausible-looking corpus: it just stops
removing the copies it was written to remove. So the estimator is checked against true
Jaccard on synthetic edits, and the dedup pass is checked against injected duplicates.
"""
import json
import random
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nord_pilot.data_tools import prepare_scale as S


def words(count, seed=0):
    """Distinctive filler text. Repetitive filler would collide on 5-word windows and make
    unrelated documents look similar, which is not what these tests are meant to measure.
    Tokens stay alphabetic so the low-text filter does not reject them for another reason."""
    rnd = random.Random(seed)
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return " ".join("w" + "".join(rnd.choice(alphabet) for _ in range(9)) for _ in range(count))


def sentences(count, seed=0, marker="s"):
    rnd = random.Random(seed + 991)
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    def token():
        return "".join(rnd.choice(alphabet) for _ in range(10))
    return [f"{marker} sentence {i} carries distinct filler content {token()} about topic {token()}"
            for i in range(count)]


class SketchTest(unittest.TestCase):
    def test_identical_text_has_identical_sketch(self):
        text = words(400, seed=1)
        self.assertEqual(S.sketch(S.normalize(text).lower()),
                         S.sketch(S.normalize(text).lower()))

    def test_unrelated_text_scores_near_zero(self):
        a = S.sketch(S.normalize(words(400, seed=2)).lower())
        b = S.sketch(S.normalize(words(400, seed=3)).lower())
        self.assertIsNotNone(a)
        overlap = sum(1 for x, y in zip(a, b) if x == y) / S.SKETCH
        self.assertLess(overlap, 0.15)

    def test_short_text_is_unsketchable(self):
        self.assertIsNone(S.sketch(S.normalize(words(S.SHINGLE + S.SKETCH - 1)).lower()))
        self.assertIsNotNone(S.sketch(S.normalize(words(S.SHINGLE + S.SKETCH)).lower()))

    def test_estimate_tracks_true_jaccard(self):
        """Deleting sentences must move the estimate the same way true Jaccard moves."""
        base = sentences(120)
        text = " ".join(base)
        truth, estimate = [], []
        for keep in (1.0, 0.85, 0.6):
            rnd = random.Random(11)
            edited = " ".join(s for s in base if rnd.random() < keep)
            a = S.shingles(text)
            b = S.shingles(edited)
            truth.append(len(a & b) / len(a | b))
            sa = S.sketch(S.normalize(text).lower())
            sb = S.sketch(S.normalize(edited).lower())
            estimate.append(sum(1 for x, y in zip(sa, sb) if x == y) / S.SKETCH)
        for t, e in zip(truth, estimate):
            self.assertLess(abs(t - e), 0.12, msg=f"true={truth} est={estimate}")
        self.assertGreater(truth[0], truth[1])
        self.assertGreater(truth[1], truth[2])

    def test_windows_are_local_to_the_document(self):
        """A window's id must not depend on earlier text, or deletions invalidate every id."""
        body = words(300, seed=21)
        prefix = words(37, seed=22)
        a = S.shingles(body)
        b = S.shingles(prefix + " " + body)
        self.assertEqual(len(a & b) / len(a), 1.0)

    def test_bands_are_aligned_with_documents(self):
        """band_keys emits BANDS keys per document, so owners must repeat to match."""
        rows = np.array([S.sketch(S.normalize(words(200, seed=s)).lower()) for s in range(5)])
        keys = S.band_keys(rows)
        self.assertEqual(keys.shape, (5 * S.BANDS,))
        owners = np.repeat(np.arange(5), S.BANDS)
        self.assertEqual(owners.shape, keys.shape)

    def test_shared_document_shares_a_band(self):
        text = words(500, seed=9)
        sketch = S.sketch(S.normalize(text).lower())
        keys = set(S.band_keys(np.array([sketch])))
        copy_keys = set(S.band_keys(np.array([S.sketch(S.normalize(text).lower())])))
        self.assertTrue(keys & copy_keys)


class DedupTest(unittest.TestCase):
    def setUp(self):
        self.out = Path(tempfile.mkdtemp(prefix="nord-dedup-test-"))
        self.corpus = S.Corpus(self.out, S.SOURCES)
        self.template = sentences(90)

    def tearDown(self):
        for handle in self.corpus.handles.values():
            handle.close()
        shutil.rmtree(self.out, ignore_errors=True)

    def add(self, text, doc_id, group="example.test"):
        self.corpus.consider({"text": text, "language_score": 1.0, "source": "edu",
                              "doc_id": doc_id, "dataset": "test", "revision": "r1",
                              "license": "odc-by", "url": "https://" + group + "/p",
                              "language": "en", "group": group})
        return self.corpus.accepted

    def run_dedup(self):
        sketches, owners, files = self.corpus.close()
        keep, oversize = S.deduplicate(self.corpus, sketches, owners, self.corpus.counts,
                                       self.corpus.per_source, self.corpus.source_of_candidate)
        return keep, oversize

    def test_near_duplicate_copy_is_removed(self):
        original = " ".join(self.template)
        edited = " ".join(line for i, line in enumerate(self.template) if i % 20)
        self.add(original, "keep-me")
        self.add(edited, "drop-me")
        self.assertEqual(self.corpus.accepted, 2)
        kept, _ = self.run_dedup()
        self.assertEqual(kept, 1)
        self.assertEqual(self.corpus.counts["near_duplicate"], 1)

    def test_exact_duplicate_is_removed_before_sketching(self):
        text = " ".join(self.template)
        self.add(text, "first")
        self.add(text, "second")
        kept, _ = self.run_dedup()
        self.assertEqual(kept, 1)
        self.assertEqual(self.corpus.counts["exact_duplicate"], 1)

    def test_distinct_document_survives(self):
        self.add(" ".join(self.template), "a")
        self.add(words(300, seed=42), "b", group="other.test")
        kept, _ = self.run_dedup()
        self.assertEqual(kept, 2)

    def test_three_way_duplicate_keeps_one(self):
        text = " ".join(self.template)
        edited = " ".join(line for i, line in enumerate(self.template) if i % 25)
        self.add(text, "one")
        self.add(edited, "two")
        self.add(edited, "three")
        kept, _ = self.run_dedup()
        self.assertEqual(kept, 1)

    def test_kept_record_keeps_required_keys(self):
        self.add(" ".join(self.template), "a")
        self.run_dedup()
        staged = [self.out / ("staged-edu.jsonl")]
        self.assertTrue(staged[0].exists())
        row = json.loads(staged[0].read_text().splitlines()[0])
        for key in ("doc_id", "text", "normalized_sha256", "source", "group", "split"):
            self.assertIn(key, row)
        self.assertIn(row["split"], {"train", "valid", "test"})


if __name__ == "__main__":
    unittest.main()
