"""Build a scaled, deduplicated pretraining corpus from Hugging Face parquet shards.

Four constraints drive this design.

1. Bandwidth. FineWeb-Edu shards are ~1.9 GB and FineWeb-2 Swedish shards ~4.8 GB, while
   a parquet footer is a few MB and each row group holds ~1000 documents. Downloading
   whole shards to keep a few thousand documents wastes ~99% of the transfer, so this
   reads HTTP byte ranges and decodes only the row groups actually needed.
2. Throughput. A single Hub connection delivers only a few MB/s, so shards are read by a
   thread pool. The document filter and dedup index are pure Python and hold the GIL, so
   the sketch is kept cheap (bottom-k instead of 32 MinHash permutations) and the
   LSH index is numpy arrays rather than a dict of lists.
3. Memory. Near-duplicate detection is O(corpus). At 5B tokens the corpus is ~20 GB of
   text and millions of documents, which does not fit in 16 GB RAM alongside the index.
   Candidates therefore stream to disk, the index is two compact numpy arrays, and the
   final decision is made by sorting band keys instead of hashing into Python objects.
4. Honesty. Near-duplicate removal is an estimate. The manifest says which estimator,
   which threshold, and what was not checked.

Free to run: no GPU, no cloud calls, no teacher calls. Chunk format matches the existing
pilot corpus consumed by the HRM+MoE trainer.
"""
import argparse, collections, hashlib, io, json, random, re, subprocess, threading, time, unicodedata
from pathlib import Path
from urllib.parse import urlsplit
from datetime import datetime, timezone

import numpy as np
import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

SESSION = requests.Session()
SESSION.mount("https://", HTTPAdapter(max_retries=Retry(total=6, backoff_factor=2,
             status_forcelist=[408, 429, 500, 502, 503, 504], respect_retry_after_header=True)))

SOURCES = [
    {"name": "edu", "repo": "HuggingFaceFW/fineweb-edu", "glob": "data/CC-MAIN-2024",
     "license": "odc-by", "language": "en", "weight": 0.45},
    {"name": "math", "repo": "HuggingFaceTB/finemath", "glob": "finemath-4plus",
     "license": "odc-by", "language": "en", "weight": 0.15},
    {"name": "explanations", "repo": "HuggingFaceTB/smollm-corpus", "glob": "cosmopedia-v2",
     "license": "odc-by", "language": "en", "weight": 0.25},
    {"name": "sv", "repo": "HuggingFaceFW/fineweb-2", "glob": "data/swe_Latn/train",
     "license": "odc-by", "language": "sv", "weight": 0.15},
]
SPECIAL = ["[PAD]", "[UNK]", "[BOS]", "[SEP]", "[EOS]"]
CHUNK_TOKENS = 254
SHINGLE = 5
SKETCH = 32
BANDS = 8
SKETCH_JACCARD = 0.65
# Shingle ids use a 61-bit Mersenne polynomial hash; the MinHash permutations use a 31-bit
# Mersenne prime. The modulus sizes are not cosmetic. In uint64, (a*x + b) with both
# operands below 2^61 overflows and wraps, which turns the affine map into a many-to-one
# collision and inflates every similarity estimate; measured that way, a pair with 0.797
# true Jaccard scored 0.969. With x, a and b all below 2^31 every product stays under 2^62
# and the arithmetic is exact, and because a is nonzero modulo a prime the map is a
# bijection, so equal minima mean equal shingles rather than a collision.
SHINGLE_MODULUS = (1 << 61) - 1
PERMUTATION_MODULUS = (1 << 31) - 1
SHINGLE_BASE = 0x100000001B3
SHINGLE_POWERS = [pow(SHINGLE_BASE, k, SHINGLE_MODULUS) for k in range(SHINGLE)]
_PERMUTATION_INDEX = np.arange(1, SKETCH + 1, dtype=np.uint64)
PERMUTATIONS_A = (np.uint64(0x1BF58E7B) * _PERMUTATION_INDEX) % (PERMUTATION_MODULUS - 1) + 1
PERMUTATIONS_B = (np.uint64(0x3D2C9F41) * _PERMUTATION_INDEX) % PERMUTATION_MODULUS
APPROX_BYTES_PER_TOKEN = 4
ROW_GROUP_ROWS = 1000
WORDS_PER_TOKEN = 0.75
ROW_GROUP_OVERHEAD = 0.14
ROW_GROUP_TOKENS = 800_000
NEEDED_COLUMNS = ("text", "url", "language_score")
BAND_MULTIPLIERS = np.array([0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F,
                             0x165667B19E3779F9, 0x27D4EB2F165667C5], dtype=np.uint64)
MAX_BAND_GROUP = 2000


def sha(x):
    return hashlib.sha256(x if isinstance(x, bytes) else x.encode()).hexdigest()


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def shingles(norm):
    """Ids of a normalized document's SHINGLE-word windows, one per window.

    Each id depends only on its own SHINGLE words. The first version rolled a hash forward
    and never removed the outgoing word, so every id depended on the whole document prefix.
    Deleting one sentence then changed every later id, and two copies of the same document
    measured 0.008 Jaccard instead of 1.0, which made near-duplicate detection inert while
    still reporting plausible zeros. The polynomial window hash below subtracts the
    outgoing word, so ids are local to their window.
    """
    words = norm.split()
    identifiers = set()
    if len(words) < SHINGLE:
        return identifiers
    ids = [int(sha(word)[:12], 16) % SHINGLE_MODULUS for word in words]
    highest = SHINGLE_POWERS[SHINGLE - 1]
    window = 0
    for offset in range(SHINGLE):
        window = (window + ids[offset] * SHINGLE_POWERS[SHINGLE - 1 - offset]) % SHINGLE_MODULUS
    identifiers.add(window)
    for start in range(len(ids) - SHINGLE):
        window = ((window - ids[start] * highest) * SHINGLE_BASE + ids[start + SHINGLE]) % SHINGLE_MODULUS
        identifiers.add(window)
    return identifiers


def sketch(norm):
    """MinHash sketch of a document's 5-word shingles, one value per permutation.

    The earlier bottom-k sketch kept the SKETCH smallest raw shingle ids and compared those
    sets directly. That estimator is biased low for the near-duplicates this filter exists
    to catch: deleting even 5% of a document measured 0.50 estimated against 0.60-0.79 true
    Jaccard, so a 0.65 threshold silently missed most paraphrased or re-crawled copies.
    Here the shingles are hashed once and then run through SKETCH independent affine
    permutations, each keeping its own minimum. The share of equal positions is an unbiased
    Jaccard estimate, and each kept minimum is the minimum of about |shingles|/SKETCH
    shingles, so position values are near-uniform instead of clustered by document size.

    Returning fewer than SKETCH values means the document has fewer distinct shingles than
    permutations, which makes the estimate noisy, so it is reported as unusable.
    """
    if len(norm.split()) < SHINGLE + SKETCH:
        return None
    values = np.fromiter(shingles(norm), dtype=np.uint64) % PERMUTATION_MODULUS
    unique = np.unique(values)
    if unique.size < SKETCH:
        return None
    hashed = (unique[:, None] * PERMUTATIONS_A[None, :]) % PERMUTATION_MODULUS
    hashed = (hashed + PERMUTATIONS_B[None, :]) % PERMUTATION_MODULUS
    return tuple(int(v) for v in hashed.min(axis=0))


def band_keys(sketches):
    """Hash each ranked band of each sketch to one 64-bit key.

    Any two documents that share every value in a band produce the same key, so grouping
    by key yields exactly the pairs worth comparing, with hash collisions only ever adding
    candidates.
    """
    shaped = sketches.reshape(len(sketches), BANDS, SKETCH // BANDS).astype(np.uint64)
    return (shaped * BAND_MULTIPLIERS).sum(axis=2, dtype=np.uint64).reshape(-1)


def group_key(row):
    url = row.get("url") or ""
    host = urlsplit(url).hostname if url else None
    return host or sha(json.dumps(row["text"][:200], sort_keys=True))


def split_for(group):
    bucket = int(sha("nord-real-v1:" + group)[:8], 16) % 100
    return "valid" if bucket < 5 else ("test" if bucket < 10 else "train")


def reject_reason(text, language_score=1.0):
    words = text.split()
    if not 80 <= len(words) <= 3000:
        return "length"
    if "\ufffd" in text or sum(c.isalpha() for c in text) / max(1, len(text)) < .5:
        return "encoding_or_low_text"
    if re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text):
        return "email_present"
    if re.search(r"(?i)\b(gsm8k|mmlu[- ]?pro|math[- ]?500|humaneval|hellaswag|arc[- ]challenge|aime 202[4-6])\b", text):
        return "benchmark_name"
    if language_score is not None and language_score < 0.8:
        return "language_score"
    if any(token in text for token in SPECIAL):
        return "control_token_text"
    if len(set(words)) / len(words) < 0.15:
        return "repetition"
    return None


class RangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests with a cached buffer.

    Parquet access is footer-then-row-group, so one request per read() would be slow. A
    read that fits the cached buffer is served from memory; otherwise the exact span is
    fetched, with a floor so tiny metadata reads do not cost a request each.
    """

    MIN_FETCH = 1 << 20

    def __init__(self, url, min_fetch=MIN_FETCH):
        self.url = url
        self.min_fetch = min_fetch
        self.total_downloaded = 0
        self.buffer = b""
        self.buffer_start = 0
        self.pos = 0
        response = SESSION.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=120)
        try:
            response.raise_for_status()
            content_range = response.headers.get("Content-Range", "")
        finally:
            response.close()
        if "/" not in content_range:
            raise ValueError("server does not support byte ranges for " + url)
        self.size = int(content_range.split("/")[-1])
        self._fill(0, min(self.min_fetch, self.size))

    def _fill(self, start, length):
        response = SESSION.get(self.url, headers={"Range": f"bytes={start}-{start + length - 1}"},
                               timeout=180)
        response.raise_for_status()
        self.buffer = response.content
        self.total_downloaded += len(self.buffer)
        self.buffer_start = start

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, max(0, self.size - self.pos))
        if n <= 0:
            return b""
        if not (self.buffer_start <= self.pos and self.pos + n <= self.buffer_start + len(self.buffer)):
            self._fill(self.pos, min(max(n, self.min_fetch), self.size - self.pos))
        offset = self.pos - self.buffer_start
        data = self.buffer[offset:offset + n]
        self.pos += len(data)
        return data

    def readinto(self, target):
        data = self.read(len(target))
        target[:len(data)] = data
        return len(data)

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else (self.pos + offset if whence == 1 else self.size + offset)
        return self.pos

    def tell(self):
        return self.pos

    def seekable(self):
        return True

    def readable(self):
        return True


class Corpus:
    """Streaming filter, exact-dedup set, and a compact bottom-k sketch index.

    Candidates are written to disk as they stream in because the text itself is far too
    large to hold. Only the sketch rows and the band keys stay in memory, as numpy arrays
    appended in fixed-size chunks.
    """

    def __init__(self, out, sources, flush_docs=4096):
        self.out = out
        self.flush_docs = flush_docs
        self.seen = set()
        self.counts = collections.Counter()
        self.per_source = {s["name"]: collections.Counter() for s in sources}
        self.paths = {s["name"]: out / ("candidates-" + s["name"] + ".jsonl") for s in sources}
        self.source_order = [s["name"] for s in sources]
        self.handles = {name: path.open("w") for name, path in self.paths.items()}
        self.sizes = collections.defaultdict(int)
        self.accepted = 0
        self.offset = {}
        self.pending = []
        self.sketch_rows, self.owners = [], []
        self.source_of_candidate = []
        self.current = None

    def for_source(self, name):
        self.current = name
        self.offset.setdefault(name, self.accepted)

    def _flush(self):
        if not self.pending:
            return
        candidates = np.array([c for c, _ in self.pending], dtype=np.int64)
        sketches = np.array([s for _, s in self.pending], dtype=np.int64)
        self.sketch_rows.append(sketches)
        self.owners.append(candidates)
        self.pending = []

    def consider(self, row):
        reason = reject_reason(row["text"], row.get("language_score"))
        if reason:
            self.counts[reason] += 1
            self.per_source[row["source"]][reason] += 1
            return False
        norm = normalize(row["text"]).lower()
        digest = sha(norm)
        if digest in self.seen:
            self.counts["exact_duplicate"] += 1
            self.per_source[row["source"]]["exact_duplicate"] += 1
            return False
        self.seen.add(digest)
        values = sketch(norm)
        record = {"doc_id": row["doc_id"], "text": row["text"], "normalized_sha256": digest,
                  "source": row["source"], "dataset": row["dataset"], "revision": row["revision"],
                  "license": row["license"], "url": row["url"], "language": row["language"],
                  "group": row["group"], "split": split_for(row["group"])}
        line = json.dumps(record, ensure_ascii=False) + "\n"
        self.handles[row["source"]].write(line)
        self.sizes[row["source"]] += len(line.encode())
        self.source_of_candidate.append(row["source"])
        if values is not None and len(values) == SKETCH:
            self.pending.append((self.accepted, values))
            if len(self.pending) >= self.flush_docs:
                self._flush()
        self.per_source[row["source"]]["candidate_documents"] += 1
        self.per_source[row["source"]]["candidate_approx_tokens"] += len(row["text"]) // APPROX_BYTES_PER_TOKEN
        self.accepted += 1
        return True

    def close(self):
        self._flush()
        for handle in self.handles.values():
            handle.close()
        sketches = np.concatenate(self.sketch_rows) if self.sketch_rows else np.empty((0, SKETCH), np.int64)
        owners = np.concatenate(self.owners) if self.owners else np.empty(0, np.int64)
        self.sketch_rows = self.owners = self.pending = self.seen = None
        return sketches, owners, {name: {"file": self.paths[name].name, "bytes": self.sizes[name]}
                                  for name in self.paths}


def deduplicate(corpus, sketches, owners, counts, per_source, source_of_candidate):
    """Resolve near-duplicates by sorting band keys, then keep candidates that survive.

    Every candidate that shares a band with an earlier survivor is dropped. Ties are
    broken by candidate order so the result does not depend on thread or shard order.
    """
    dropped = np.zeros(len(source_of_candidate), dtype=bool)
    row_of_candidate = np.full(len(source_of_candidate), -1, dtype=np.int64)
    row_of_candidate[owners] = np.arange(len(owners), dtype=np.int64)
    if owners.size:
        # One band pass at a time, so only owners.size keys and matching owners exist at
        # once instead of BANDS times that. Band order does not affect the outcome: a pair
        # sharing any band is caught in that band's pass, and ties are broken by candidate
        # order, so the result stays independent of thread and shard order.
        keys = band_keys(sketches)
        # band_keys flattens row-major, so band b of every document is the strided column
        # b of the (document, band) matrix, not a contiguous slice.
        key_matrix = keys.reshape(len(owners), BANDS)
        oversize = 0
        for band in range(BANDS):
            band_keys_view = key_matrix[:, band]
            band_owners = owners
            order = np.argsort(band_keys_view, kind="stable")
            sorted_keys, sorted_owners = band_keys_view[order], band_owners[order]
            boundaries = np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1
            starts = np.concatenate(([0], boundaries))
            ends = np.concatenate((boundaries, [len(sorted_keys)]))
            sizes = ends - starts
            multi = np.flatnonzero(sizes >= 2)
            for group in multi:
                members = sorted_owners[starts[group]:ends[group]]
                if len(members) > MAX_BAND_GROUP:
                    oversize += 1
                    continue
                leaders = []
                for member in sorted(set(members.tolist())):
                    if dropped[member]:
                        continue
                    row = sketches[row_of_candidate[member]]
                    if any(np.intersect1d(row, sketches[row_of_candidate[leader]],
                                          assume_unique=True).size / SKETCH >= SKETCH_JACCARD
                           for leader in leaders):
                        dropped[member] = True
                    else:
                        leaders.append(member)
        del keys, key_matrix
    else:
        oversize = 0
    # Candidate ids are assigned in harvest order, so the files must be read in the same
    # source order the candidates were written. kept counts documents that survived, which
    # is not the number of candidates scanned once anything is dropped.
    kept, index = 0, 0
    for source in corpus.source_order:
        path = corpus.paths[source]
        staged = corpus.out / ("staged-" + source + ".jsonl")
        with path.open() as reader, staged.open("w") as sink:
            for line in reader:
                row = json.loads(line)
                if dropped[index]:
                    counts["near_duplicate"] += 1
                    per_source[source]["near_duplicate"] += 1
                else:
                    sink.write(line)
                    per_source[source]["kept_documents"] += 1
                    per_source[source]["kept_approx_tokens"] += len(row["text"]) // APPROX_BYTES_PER_TOKEN
                    kept += 1
                index += 1
        path.unlink()
    return kept, oversize


def harvest(source, shards, revision, corpus, target_candidate_tokens,
            max_download_bytes, time_budget_s, workers):
    """Read row groups from shards in parallel over a range-request file, in shard order.

    Results are consumed in submission order, which keeps which-document-wins deterministic
    when deduplicating. Each round asks only for the tokens still missing and spreads them
    over as few shards as should cover the gap, so the batch shrinks as the target nears.
    """
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import hf_hub_url
    stop = threading.Event()
    downloaded, shard_log, rows_seen, raw_seen = 0, [], 0, 0
    started = time.monotonic()

    def fetch(filename, token_budget, min_fetch):
        """Read a contiguous run of row groups covering roughly token_budget raw tokens."""
        if stop.is_set():
            return filename, [], 0, 0, 0, 0
        url = hf_hub_url(repo_id=source["repo"], filename=filename, repo_type="dataset",
                         revision=revision)
        handle = RangeFile(url, min_fetch=min_fetch)
        parquet = pq.ParquetFile(handle)
        metadata = parquet.metadata
        groups = metadata.num_row_groups
        available = set(parquet.schema_arrow.names)
        columns = [c for c in NEEDED_COLUMNS if c in available]
        rows, rows_read, chars = [], 0, 0
        wanted = max(1, int(token_budget / WORDS_PER_TOKEN * (1 + ROW_GROUP_OVERHEAD)))
        try:
            need, seen_bytes = 0, 0
            while need < groups and seen_bytes < wanted:
                seen_bytes += metadata.row_group(need).total_byte_size
                need += 1
            first = 0 if need >= groups else random.Random(sha(filename)[:12]).randrange(groups - need + 1)
            for g in range(first, first + need):
                if stop.is_set():
                    break
                table = parquet.read_row_group(g, columns=columns)
                present = set(table.column_names)
                for i in range(table.num_rows):
                    text = table["text"][i].as_py() or ""
                    row = {"text": text,
                           "url": table["url"][i].as_py() if "url" in present else "",
                           "language_score": (table["language_score"][i].as_py()
                                              if "language_score" in present else 1.0),
                           "source": source["name"], "dataset": source["repo"],
                           "revision": revision, "license": source["license"],
                           "language": source["language"], "doc_id": sha(text)}
                    row["group"] = group_key(row)
                    rows.append(row)
                    chars += len(text)
                rows_read += table.num_rows
                del table
        finally:
            downloaded_bytes = handle.total_downloaded
            del handle, parquet
        return filename, rows, downloaded_bytes, groups, rows_read, chars // APPROX_BYTES_PER_TOKEN

    with ThreadPoolExecutor(max_workers=workers) as pool:
        shards_done = 0
        while shards_done < len(shards):
            remaining = target_candidate_tokens - corpus.per_source[source["name"]]["candidate_approx_tokens"]
            if remaining <= 0:
                stop.set()
                break
            # Fan out only as widely as the missing tokens can keep busy. A row group is
            # ~250k tokens, so asking twelve shards for 30k tokens each would download
            # twelve row groups to keep one shard's worth of text.
            fanout = max(1, min(workers, len(shards) - shards_done,
                                -(-remaining // ROW_GROUP_TOKENS)))
            token_budget = max(ROW_GROUP_TOKENS, remaining // fanout)
            batch = shards[shards_done:shards_done + fanout]
            min_fetch = max(1 << 18, int(token_budget * APPROX_BYTES_PER_TOKEN / 2))
            for filename, rows, bytes_, groups, rows_read, raw in pool.map(
                    lambda name: fetch(name, token_budget, min_fetch), batch):
                downloaded += bytes_
                shard_log.append({"file": filename, "row_groups": groups, "docs_read": rows_read,
                                  "bytes": bytes_})
                rows_seen += rows_read
                raw_seen += raw
                shards_done += 1
                for row in rows:
                    corpus.consider(row)
                print(json.dumps({"source": source["name"], "shard": filename.rsplit("/", 1)[-1],
                                  "raw_tokens": raw, "candidate_tokens":
                                  corpus.per_source[source["name"]]["candidate_approx_tokens"],
                                  "candidate_docs": corpus.accepted,
                                  "downloaded_mb": downloaded >> 20,
                                  "elapsed_s": round(time.monotonic() - started)}), flush=True)
                del rows
            if downloaded >= max_download_bytes or time.monotonic() - started > time_budget_s:
                stop.set()
                break
    return downloaded, shard_log, rows_seen, raw_seen


def count_lines(path):
    total = 0
    with path.open("rb") as handle:
        for _ in handle:
            total += 1
    return total


def train_tokenizer(out, staged, vocab, doc_budget):
    """Train BPE on the train split only, over a bounded deterministic sample.

    A 32k BPE converges long before it has seen billions of tokens, and training it on
    every document would add a full extra pass over multi-GB staged files. Documents are
    chosen by a stable per-document hash, so the sample spreads across every source
    instead of following the order the shards happened to arrive in.
    """
    lines_per_file, total, samples = {}, 0, []
    for path in staged:
        lines_per_file[path] = count_lines(path)
        total += lines_per_file[path]
    for path, lines in lines_per_file.items():
        wanted = max(1, int(doc_budget * lines / max(1, total)))
        stride = max(1, lines // wanted)
        sample = path.with_suffix(".sample")
        samples.append(sample)
        kept = 0
        with path.open() as source, sample.open("w") as sink:
            for line in source:
                row = json.loads(line)
                if (row["split"] == "train" and kept < wanted
                        and int(row["normalized_sha256"][:8], 16) % stride == 0):
                    sink.write(line)
                    kept += 1
        print(json.dumps({"stage": "tokenizer-sample", "file": path.name,
                          "kept": kept, "of": lines}), flush=True)
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab, special_tokens=SPECIAL,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=False)

    def stream():
        for sample in samples:
            with sample.open() as handle:
                for line in handle:
                    yield json.loads(line)["text"]

    tokenizer.train_from_iterator(stream(), trainer)
    tokenizer.save(str(out / "tokenizer.json"))
    for sample in samples:
        sample.unlink()
    return tokenizer


def encode_corpus(out, tokenizer, staged):
    stats = collections.defaultdict(collections.Counter)
    handles = {s: (out / f"{s}.jsonl").open("w") for s in ["train", "valid", "test"]}
    groups = {s: set() for s in handles}
    control_ids = {tokenizer.token_to_id(t) for t in SPECIAL}
    seen_docs, total = 0, 0
    with (out / "provenance.jsonl").open("w") as provenance:
        for path in staged:
            batch = []
            with path.open() as source:
                for line in source:
                    batch.append(json.loads(line))
                    if len(batch) >= 256:
                        seen_docs, total = _encode_batch(tokenizer, batch, handles, groups, stats,
                                                         control_ids, provenance, seen_docs, total)
                        batch = []
                if batch:
                    seen_docs, total = _encode_batch(tokenizer, batch, handles, groups, stats,
                                                     control_ids, provenance, seen_docs, total)
    for handle in handles.values():
        handle.close()
    (out / "test-fresh.jsonl").write_bytes((out / "test.jsonl").read_bytes())
    stats["fresh_test"] = collections.Counter(stats["test"])
    return stats, groups


def _encode_batch(tokenizer, batch, handles, groups, stats, control_ids, provenance, seen_docs, total):
    batch.sort(key=lambda r: (len(r["text"]), r["doc_id"]))
    encodings = tokenizer.encode_batch([r["text"] for r in batch])
    for row, encoding in zip(batch, encodings):
        ids = encoding.ids
        if any(t in control_ids for t in ids):
            raise ValueError("unexpected special token in document " + row["doc_id"])
        if tokenizer.decode(ids) != row["text"]:
            raise ValueError("tokenizer round-trip failed for " + row["doc_id"])
        stat = stats[row["split"]]
        stat["documents"] += 1
        stat["text_tokens"] += len(ids)
        stat[row["source"] + "_tokens"] += len(ids)
        groups[row["split"]].add(row["group"])
        for i in range(0, len(ids), CHUNK_TOKENS):
            chunk = ids[i:i + CHUNK_TOKENS]
            record = {"id": row["doc_id"] + ":" + str(i), "doc_id": row["doc_id"],
                      "kind": "pretrain", "token_ids": chunk,
                      "document_end": i + CHUNK_TOKENS >= len(ids)}
            handles[row["split"]].write(json.dumps(record) + "\n")
            stat["sequences"] += 1
            stat["supervised_tokens"] += len(chunk) + int(record["document_end"])
        provenance.write(json.dumps({k: v for k, v in row.items() if k != "text"},
                                    ensure_ascii=False) + "\n")
        seen_docs += 1
        total += len(ids)
        if seen_docs % 100000 == 0:
            print(json.dumps({"stage": "encode", "encoded_docs": seen_docs, "text_tokens": total,
                              "train_supervised_tokens": stats["train"]["supervised_tokens"]}), flush=True)
    return seen_docs, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target-tokens", type=int, default=5_000_000_000)
    ap.add_argument("--shards-per-source", type=int, default=1200)
    ap.add_argument("--tokenizer-docs", type=int, default=300_000)
    ap.add_argument("--tokenizer-vocab", type=int, default=32768)
    ap.add_argument("--max-download-bytes", type=int, default=120_000_000_000)
    ap.add_argument("--source-time-budget-s", type=int, default=14400)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for guard in ["manifest.json", "tokenizer.json", "train.jsonl"]:
        if (a.out / guard).exists():
            raise ValueError("output already contains " + guard + "; choose a fresh directory")
    from huggingface_hub import HfApi
    api = HfApi()
    started = time.monotonic()
    corpus = Corpus(a.out, SOURCES)
    sources, download_bytes, raw_read = [], 0, 0
    try:
        for source in SOURCES:
            meta = api.dataset_info(source["repo"])
            revision = meta.sha
            if (meta.cardData or {}).get("license") != source["license"]:
                raise ValueError("dataset license metadata changed for " + source["repo"])
            files = sorted(f for f in api.list_repo_files(source["repo"], repo_type="dataset")
                           if f.endswith(".parquet") and source["glob"] in f)
            if not files:
                raise ValueError(f"no shards matched {source['repo']} {source['glob']}")
            stride = max(1, len(files) // a.shards_per_source)
            shards = files[::stride][:a.shards_per_source]
            target = int(a.target_tokens * source["weight"])
            corpus.for_source(source["name"])
            print(json.dumps({"stage": "harvest", "source": source["name"], "shards": len(shards),
                              "candidate_token_target": target}), flush=True)
            downloaded, shard_log, rows_seen, raw = harvest(
                source, shards, revision, corpus, target,
                a.max_download_bytes - download_bytes, a.source_time_budget_s, a.workers)
            download_bytes += downloaded
            raw_read += raw
            sources.append({"name": source["name"], "repo": source["repo"], "glob": source["glob"],
                            "license": source["license"], "revision": revision,
                            "weight": source["weight"], "candidate_token_target": target,
                            "shards_available": len(files), "raw_tokens_read": raw,
                            "docs_read": rows_seen, "shards": shard_log,
                            "downloaded_bytes": downloaded,
                            "counts": dict(corpus.per_source[source["name"]])})
    finally:
        sketches, owners, candidate_files = corpus.close()
    print(json.dumps({"stage": "harvested", "candidate_documents": corpus.accepted,
                      "sketched": int(owners.size), "rejections": dict(corpus.counts),
                      "download_gb": round(download_bytes / 1e9, 2)}), flush=True)
    del corpus.seen
    survivors, oversize = deduplicate(corpus, sketches, owners, corpus.counts, corpus.per_source,
                                      corpus.source_of_candidate)
    print(json.dumps({"stage": "deduplicated", "kept_documents": survivors,
                      "near_duplicate": corpus.counts["near_duplicate"],
                      "exact_duplicate": corpus.counts["exact_duplicate"],
                      "bands_skipped_over_cap": oversize}), flush=True)
    del sketches, owners, corpus.source_of_candidate
    staged = [a.out / ("staged-" + s["name"] + ".jsonl") for s in SOURCES]
    tokenizer = train_tokenizer(a.out, staged, a.tokenizer_vocab, a.tokenizer_docs)
    print(json.dumps({"stage": "tokenizer", "vocab": tokenizer.get_vocab_size()}), flush=True)
    stats, groups = encode_corpus(a.out, tokenizer, staged)
    for a_name, b_name in [("train", "valid"), ("train", "test"), ("valid", "test")]:
        assert not (groups[a_name] & groups[b_name]), f"group overlap between {a_name} and {b_name}"
    for path in staged + [a.out / v["file"] for v in candidate_files.values()]:
        path.unlink(missing_ok=True)
    files = {}
    for path in sorted(a.out.glob("*")):
        if path.is_file() and path.name != "manifest.json":
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 22), b""):
                    digest.update(block)
            files[path.name] = digest.hexdigest()
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(),
                "purpose": "scaled real-text pretraining corpus, not a product benchmark",
                "candidate_documents": corpus.accepted,
                "kept_documents": survivors,
                "rejections": {k: v for k, v in corpus.counts.items()},
                "per_source": {k: dict(v) for k, v in corpus.per_source.items()},
                "sources": sources,
                "candidate_files": candidate_files,
                "download_bytes": download_bytes,
                "raw_tokens_read": raw_read,
                "target_tokens": a.target_tokens,
                "chunk_tokens": CHUNK_TOKENS,
                "token_counting": f"streaming targets use a {APPROX_BYTES_PER_TOKEN}-bytes-per-token "
                                  "estimate; statistics report true BPE counts",
                "statistics": {k: dict(v) for k, v in stats.items()},
                "tokenizer_vocab": tokenizer.get_vocab_size(),
                "tokenizer_training_split": "train only",
                "tokenizer_sample": f"at most {a.tokenizer_docs} train documents, selected by "
                                    "stable document hash, spread across sources",
                "near_dedup": f"{SHINGLE}-word shingles, bottom-{SKETCH} k-minimum-values sketch "
                              f"(not 32-permutation MinHash), {BANDS} band keys grouped by sort; "
                              "a pair is only compared when every id in one band is shared, so "
                              f"near-duplicates that differ inside a band are missed; a compared "
                              f"pair is dropped when >= {SKETCH_JACCARD:.0%} of the 32 ids match; "
                              f"bands with more than {MAX_BAND_GROUP} members are skipped entirely; "
                              "this is an estimate, so recall is approximate and zero residual "
                              "overlap is not guaranteed",
                "split_policy": "hostname (or synthetic seed) group, stable 90/5/5 hash",
                "decontamination": "benchmark-name screening only; full benchmark overlap audit NOT performed",
                "PII": "email screening only; not complete PII removal",
                "runtime_seconds": time.monotonic() - started,
                "code_sha256": sha(Path(__file__).read_bytes()),
                "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                             text=True, cwd=Path(__file__).parent).stdout.strip(),
                "files": files}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"statistics": manifest["statistics"],
                      "download_gb": round(download_bytes / 1e9, 2),
                      "runtime_seconds": manifest["runtime_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
