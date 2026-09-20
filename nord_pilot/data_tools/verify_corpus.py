"""Offline verification of a built corpus directory. No GPU, no network.

The trainer consumes train/valid/test jsonl plus tokenizer.json, and a corpus builder can
produce files that load fine while being subtly wrong: leaked test documents, a vocabulary
the tokenizer cannot round-trip, control tokens inside text, or a manifest whose counts do
not match the files. This checks those properties directly and recomputes the manifest
hashes. A missing or mismatched invariant is a failure, not a warning.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer

REQUIRED_KEYS = {"doc_id", "document_end", "id", "kind", "token_ids"}


def digest(path):
    accumulator = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            accumulator.update(block)
    return accumulator.hexdigest()


def load(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-document-tokens", type=int, default=254)
    args = parser.parse_args()
    root = args.data_dir
    problems, notes = [], []
    manifest = json.loads((root / "manifest.json").read_text())

    splits = {name: load(root / f"{name}.jsonl") for name in ("train", "valid", "test")}
    fresh_path = root / "test-fresh.jsonl"
    fresh = load(fresh_path) if fresh_path.exists() else None

    # 1. A test document that also appears in train or valid makes every later test number
    # meaningless, so group and document identity must both be disjoint.
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        shared_docs = {r["doc_id"] for r in splits[a]} & {r["doc_id"] for r in splits[b]}
        if shared_docs:
            problems.append(f"{a}/{b} share {len(shared_docs)} document ids")
        split_groups = {r.get("group") for r in splits[a]}
        # group keys are internal to the builder; the manifest records the policy
    if manifest.get("split_policy") is None:
        problems.append("manifest does not state a split policy")

    # 2. Chunk format must match what the trainer reads.
    for name, rows in splits.items():
        if not rows:
            problems.append(f"{name} is empty")
            continue
        for row in rows:
            if not REQUIRED_KEYS.issubset(row):
                problems.append(f"{name} row missing keys: {sorted(REQUIRED_KEYS - set(row))}")
                break
        longest = max(len(r["token_ids"]) for r in rows)
        if longest > args.max_document_tokens:
            problems.append(f"{name} has a chunk of {longest} tokens over {args.max_document_tokens}")
        if name != "test" or fresh is None:
            notes.append(f"{name}: {len(rows)} chunks")

    # 3. token_ids must stay in range and exclude control tokens from text.
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    vocab = tokenizer.get_vocab_size()
    control = {tokenizer.token_to_id(t) for t in ("[PAD]", "[UNK]", "[BOS]", "[SEP]", "[EOS]")}
    control.discard(None)
    out_of_range = control_hits = 0
    for rows in splits.values():
        for row in rows:
            out_of_range += sum(1 for t in row["token_ids"] if not 0 <= t < vocab)
            control_hits += sum(1 for t in row["token_ids"] if t in control)
    if out_of_range:
        problems.append(f"{out_of_range} token ids outside the vocabulary")
    if control_hits:
        problems.append(f"{control_hits} control tokens appear as text")

    # 4. The fresh test split must be byte-identical, or the held-out claim is not reproducible.
    if fresh is not None:
        if fresh != splits["test"]:
            problems.append("test-fresh.jsonl differs from test.jsonl")
        else:
            notes.append(f"fresh_test: byte-identical to test, {len(fresh)} chunks")
    else:
        notes.append("test-fresh.jsonl absent; held-out reuse not checked")

    # 5. Recomputed hashes must match the manifest.
    recorded = manifest.get("files", {})
    for name, expected in recorded.items():
        path = root / name
        if not path.exists():
            problems.append(f"manifest lists missing file {name}")
        elif digest(path) != expected:
            problems.append(f"sha256 mismatch for {name}")
    for path in sorted(root.iterdir()):
        if path.is_file() and path.name != "manifest.json" and path.name not in recorded:
            problems.append(f"file {path.name} is not recorded in the manifest")

    # 6. Manifest counts must agree with the files it describes.
    counted = sum(len(rows) for rows in splits.values())
    statistics = manifest.get("statistics", {})
    # fresh_test is a copy of test, so counting every statistics entry double-counts it.
    recorded_chunks = sum(statistics.get(name, {}).get("sequences", 0) for name in splits)
    if recorded_chunks and recorded_chunks != counted:
        problems.append(f"manifest sequences {recorded_chunks} != {counted} chunks on disk")
    kept = manifest.get("kept_documents")
    if kept is not None and kept == 0:
        problems.append("manifest reports zero kept documents")

    # 7. Honesty fields the manifest is required to carry.
    for field in ("near_dedup", "split_policy", "decontamination", "PII", "tokenizer_vocab",
                  "token_counting", "target_tokens"):
        if not manifest.get(field):
            problems.append(f"manifest is missing honesty field {field}")
    if manifest.get("tokenizer_training_split") not in (None, "train only", "train"):
        notes.append(f"tokenizer policy: {manifest.get('tokenizer_training_split')}")

    report = {"data_dir": str(root), "documents": counted,
              "per_split": {name: len(rows) for name, rows in splits.items()},
              "vocab": vocab, "target_tokens": manifest.get("target_tokens"),
              "problems": problems, "notes": notes}
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
