# Real-data rehearsal

This bounded preparation path replaces the shop fixture for subsequent ingestion tests. It does not download or process the full 18.5T-token FineWeb corpus. The previous Nord plan's module 43 explicitly describes that supply as public data, not an already prepared local corpus.

Run on the local CPU with `requests` and `tokenizers` installed:

```bash
python nord_pilot/data_tools/prepare_real.py --out /path/to/new-data
python -m unittest nord_pilot.test_local nord_pilot.test_guard nord_pilot.test_followup nord_pilot.data_tools.test_data
python nord_pilot/run.py --device cpu --config /path/to/cpu-real.json --data-dir /path/to/new-data --out /path/to/new-run --steps 12 --schedule-steps 12 --batch-tokens 1024 --eval-every 6 --minutes 5
```

Use a copy of `nord_pilot/configs/cpu.json` with `vocab_size=8192` for the local check. A future CUDA config must use `grouped_triton`, the new vocabulary and the original native kernel gates. This new tokenizer requires fresh weights; do not load the shop checkpoint into a changed vocabulary.

The downloader samples 5,500 documents in dispersed deterministic windows from FineWeb-Edu sample-10BT, FineMath finemath-4plus, FineWeb-2 swe_Latn and SmolLM-Corpus cosmopedia-v2. It records upstream revision headers, per-page hashes, source metadata and local code hash. A cached page whose revision differs from current source metadata fails closed. These windows are not a representative random sample of the entire corpora. The API reader is for a small rehearsal; use pinned Parquet shards/streaming for larger runs.

Filters reject overly short/long documents, poor language confidence, replacement characters, low alphabetic fraction, email addresses, benchmark names and heavy repetition. Normalized exact dedup and approximate MinHash candidate matching run before splitting. Website hostname or synthetic prompt hash determines a stable 90/5/5 train/valid/test assignment. It is not a complete registered-domain/semantic-seed leakage guarantee. The pipeline does not certify complete PII removal or benchmark decontamination. Add full evaluation-text overlap checks before publishing benchmark scores.

An 8,192-entry byte-level BPE tokenizer trains exclusively on training documents, including all 256 byte symbols. Tokenizer round trips are checked. Documents become causal 254-token chunks with source IDs; EOS appears only at actual document ends. Packed attention keeps sequences separate. All tokens in causal text are prediction targets; raw passages are never placed into a bidirectional prefix and then scored as if they were unseen.

`prepare_teacher.py` reuses the previous Nord pilot24 exports only after checking the recorded DeepSeek V4.1 Flash model, verification status, task separation and new-tokenizer context length. It exports prompt + code solution, not long teacher traces. Prior execution verification is recorded as historical and is not silently presented as a fresh test.

It also prepares 50 new high-reasoning direct-DeepSeek requests plus independently calculated final-answer checks. It sends no requests. The narrow generated task family is suitable for client plumbing, not a reasoning benchmark. New output must be independently checked, and short explanations reviewed, before admission to training. A direct-provider cost estimate is not a gateway quote or an enforced spending cap. API billing is separate from Verda credit.

Sources: https://huggingface.co/datasets/HuggingFaceFW/fineweb ; https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu ; https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 ; https://huggingface.co/datasets/HuggingFaceTB/finemath ; https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus ; https://api-docs.deepseek.com/quick_start/pricing/ ; https://api-docs.deepseek.com/guides/thinking_mode/ ; https://cdn.deepseek.com/policies/en-US/deepseek-open-platform-terms-of-service.html

Dataset metadata lists ODC-By for these repositories. Keep attribution/provenance and applicable Common Crawl/source terms; a dataset license is not a blanket clearance of every underlying web document. DeepSeek's Open Platform terms section 4.2 explicitly permits model training/distillation subject to its terms and applicable law.
