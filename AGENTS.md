# HRM-Text Agent Guide

## Real-data rehearsal lessons (2026-09-20)

- Continue a completed bounded run through `--init-export`: it verifies exact config/tokenizer compatibility, records the parent export SHA/step, loads weights into a fresh optimizer, and keeps subsequent exact resumes bound to that lineage. Pass the same trusted export on every resume; do not relabel weights-only continuation as optimizer-state continuation.
- Expanded corpora must reuse the original tokenizer byte-for-byte when continuing weights. `test-fresh.jsonl` contains only test documents absent from the base provenance and must remain excluded from training and checkpoint selection. Dataset-server 429 responses require paced requests and Retry-After handling; partial page caches are reusable.
- The 69,238,784-parameter real-data H100 run at `133a6de` passed 20 vs 10+10 exact recovery (256 tensors, zero difference) and 600-step training. Full validation improved 9.476005 -> 5.834325; untouched-test loss improved 9.468887 -> 5.811927. Six fixed completions remained repetitive/incorrect: this is a technical learning result, not usable chat or a GPT-level result.
- Preserve execution source, tokenizer and corpus alongside full checkpoints. This runner fingerprints `schedule_steps` and rejects steps beyond it; the completed 600-step checkpoint cannot simply be resumed with a larger schedule. Plan any longer run or explicit weights-only continuation before renting the GPU, and never bypass the recovery fingerprint.
- The real-data run used chunks of at most 255 input tokens packed into a 2,048-token batch. Its approximately 16.1k input tokens/s training-only rate is not a long-context, 0.6B-model or end-to-end throughput claim. The final lineage consumed 1,163,066 tokens, much less than the prepared corpus.

- The 2026-09-20 scaled build (`prepare_scale.py`, revision `0b4ff72`, 2h26m, 53.24 GB downloaded, $0 cloud) landed at **2.81B train tokens, not the 5B target**: edu exhausted its 500-shard supply while math/explanations/sv hit the `--shards-per-source` and source-time caps. Report 2,810,517,880 train tokens (12,637,289 chunks) when planning compute; do not quote 5B. Extending to >=5B requires raising the shard cap and re-running a multi-hour free build.
- `verify_corpus.py` must stream splits. Loading the 18 GB `train.jsonl` as Python lists caused swap thrash and never completed; the streaming rewrite holds only doc-id sets and hashes `test-fresh` incrementally, with RSS flat at 0.37 GB. Keep it that way when splits grow.
- `nord_pilot/data_tools/streaming.py` (`JsonlCorpus`) is the only supported way to read a corpus of this size in the trainer. It indexes line offsets and hashes the file in one pass (supplying the resume fingerprint digests without a second 18 GB read), seeds a training permutation from `--seed`, and decodes a record only when a batch needs it. Bound evaluation with `--valid-records`. `nord_pilot/run.py` previously built full Python lists of every record and cannot be reverted to that shape.
- Local streaming measurements, to be re-measured on GPU rather than assumed: 12.6M records indexed in 8.5 s at 0.37 GB RSS, 14,140 records/s decode. All 35 local tests pass and the exact-recovery gate still reports `max_abs_difference: 0.0` over 75 tensors.
- Exact parameter counts from instantiating the real configs (vocab 32,768): 69.2M at hidden 512/L8; 149.5M at 768/L8; 260.1M at 1024/L8; 320.9M at 1024/L10; 381.8M at 1024/L12; 591.3M at 1280/L12. Quote these instead of approximate size labels.

- `gpu_real_test.sh` uses the packaged real corpus, 8,192-token vocabulary, full validation, 20 versus 10+10 exact recovery and a bounded 600-step continuation. Keep it separate from the historical fixture test.
- Verda guard `--max-hours` and `--stop-usd` may only tighten the original two-hour/$20 limits. The real-data test uses one hour/$5, with independent local and remote guards, then verified artifact download before manual cleanup.

- Pin dataset viewer x-revision against HF metadata and save page hashes. Cosmopedia seed_data can be merely the string fineweb; group by prompt hash instead. Hostname grouping is not full semantic or registered-domain decontamination.
- Do not call Tokenizer.get_vocab_size() in a per-token loop: this installed tokenizers build copies the vocabulary. Read it once and pass the size to the encoder.
- Raw pretraining is causal with a BOS-only prefix; EOS belongs only at actual document ends. Train the tokenizer on training documents only. Changing token IDs requires fresh model weights.
- Public corpus availability is not a prepared local corpus. Tokenizer compression and a tiny CPU ingestion check are not capability evidence. DeepSeek API billing is separate from Verda credit; distinguish historical verifier records from checks rerun now.

## Project Positioning

This repository is the working checkout for HRM-Text training on the rjob
cluster. It is not a generic language-model training template.

Use these files as the primary source of truth:

- `README.md` for the HRM-Text workflow and supported training modes
- `docker/Dockerfile` for the tested CUDA/PyTorch/FlashAttention environment
- `config/cfg_pretrain.yaml` for default pretraining settings
- `config/cfg_sft.yaml` for full-parameter SFT settings
- `pretrain.py` for the actual Hydra and FSDP2 training entrypoint

## Version Control

All agent code changes in this repository must be managed with git:

- Work on a named branch instead of leaving substantial changes only in the
  worktree.
- Run `git status --short --branch` before and after edits.
- Stage only intentional code, script, and documentation changes; never stage
  data, checkpoints, W&B outputs, or rjob logs.
- Commit completed changes with a concise message after validation. If a change
  is intentionally left uncommitted, record why in the user update.
- After completing cluster, training, data-prep, or evaluation work, update this
  AGENTS.md with durable lessons learned, reusable commands, and new pitfalls
  before committing. Treat this as part of the done criteria, not optional
  cleanup.
- Record all future HRM experiment results and operational outcomes in
  `docs/hrm_eval_results.md`. Training launches, eval jobs, benchmark
  summaries, failed attempts, root-cause notes, cleanup actions, resource
  settings, follow-up AIME/MMLU-Pro results, and notable reruns should be
  appended there rather than scattered across separate ad hoc notes. Write new
  experiment records in Chinese; keep benchmark names, metric keys, paths, and
  command fragments in their original spelling when that is clearer.
- Do not edit entrypoint scripts while an rjob is still reading them from shared
  storage. Prefer committing fixes, then launching a fresh rjob from that commit.

## Environment Model

Do not treat the local `.venv` as the training environment. HRM training depends
on Hopper-class GPUs, CUDA 12.8-era PyTorch, and `flash_attn_3`, so meaningful
training and import validation should run through rjob unless the task is
explicitly CPU-only.

The source-of-truth environment is the HRM image from the README:

```bash
sapientai/hrm-text:latest
```

The current rjob default uses the internal XTuner image plus a lightweight HRM
Python overlay because this cluster cannot currently pull the public Docker Hub
image:

```bash
registry.h.pjlab.org.cn/ailab-puyu-puyu_gpu/xtuner:pt28_20250911_6652194
```

If an internal HRM image is available, override `image=... bootstrap=0` when
launching rjob.

Build an internal image on a machine with Docker daemon access:

```bash
IMAGE_NAME=registry.h.pjlab.org.cn/ailab-moe-moe_gpu/hrm-text \
PUSH=1 bash scripts/build_hrm_image.sh
```

## Rjob Workflow

Use the HRM rjob wrappers under `scripts/`:

- `scripts/rjob_hrm_env_check.sh` checks the container imports and can request
  either CPU-only (`num_gpus=0`) or one 8-GPU node (`num_gpus=8`)
- `scripts/rjob_hrm_pretrain.sh` launches `pretrain.py` with pretraining config
- `scripts/rjob_hrm_sft.sh` launches `pretrain.py --config-name cfg_sft`
- `scripts/rjob_hrm_eval.sh` launches evaluation; with the default entrypoint it
  runs `python -m evaluation.main` on one GPU, and with
  `entrypoint=scripts/hrm_eval_fanout_entrypoint.sh num_gpus=8` it runs sharded
  8-card evaluation
- `scripts/rjob_hrm_eval_after_epoch.sh` watches a checkpoint directory until a
  target epoch is complete and stable, then submits an eval rjob
- `scripts/rjob_hrm_prepare_data.sh` runs HRM pretraining data preparation
- `scripts/rjob_hrm_common.sh` owns rjob resources, image, mounts, and RDMA flags
- `scripts/hrm_entrypoint.sh` owns container-side env vars and `torchrun`
- `scripts/hrm_eval_entrypoint.sh` owns container-side evaluation setup
- `scripts/hrm_eval_fanout_entrypoint.sh` shards each selected benchmark's
  prompts across available GPUs and aggregates metrics after generation
- `scripts/hrm_prepare_data_entrypoint.sh` owns `data_io` download, tokenization,
  and stratified sampling
- `scripts/build_hrm_image.sh` builds and optionally pushes an internal HRM image
- `docker/requirements/runtime_overlay.txt` is the bootstrap overlay for the
  internal fallback image; it intentionally excludes `torch`, `flash_attn_3`,
  and `vllm`
- `scripts/tokenize_data_io_python.py` is a no-Cargo fallback that writes the
  same tokenized layout as `data_io/tokenizer`
- `scripts/compare_tokenized_outputs.py` compares Rust and Python tokenized
  outputs for small-sample parity before trusting the Python fallback
- Set `tokenizer_workers=<N>` on `stage=tokenize` when using the Python
  fallback; the container may report a low `os.cpu_count()` despite a larger
  rjob CPU request.
- Prefer the official Rust `data_io/tokenizer` for full data prep. A shared
  Rust toolchain is installed at `/mnt/shared-storage-user/quxiaoye/.hrm-rust`,
  and the wrapper passes `CARGO_HOME`, `RUSTUP_HOME`, and mirror URLs into
  rjob. `tokenizer_impl=auto` uses Rust only when `cargo` is already present;
  it falls back to Python instead of auto-installing Rust. Use
  `tokenizer_impl=rust bootstrap=0 cpu=5` for tokenization so the
  Rust tokenizer sees only four worker threads; the official tokenizer holds a
  file's tokenized output in memory before writing, so exposing dozens of CPUs
  can over-parallelize large parquet files.

These wrappers follow the same pattern as the reference XTuner project:
thin experiment wrappers set `job_name`, `num_gpus`, and training overrides,
then delegate to a common rjob launcher and one container entrypoint.

Common overrides:

```bash
dry_run=true bash scripts/rjob_hrm_env_check.sh
job_name=hrm-env-check-cpu bash scripts/rjob_hrm_env_check.sh
job_name=hrm-env-check-gpu num_gpus=8 bash scripts/rjob_hrm_env_check.sh
stage=check bash scripts/rjob_hrm_prepare_data.sh
stage=download_cleaned bash scripts/rjob_hrm_prepare_data.sh
stage=tokenize bash scripts/rjob_hrm_prepare_data.sh
stage=tokenize tokenizer_impl=rust bootstrap=0 cpu=5 bash scripts/rjob_hrm_prepare_data.sh
stage=sample epochs=4 bash scripts/rjob_hrm_prepare_data.sh
python scripts/compare_tokenized_outputs.py /path/to/rust_tokenized /path/to/python_tokenized
num_gpus=8 arch_size=L global_batch_size=172032 extra_args='lr=2.5e-4' bash scripts/rjob_hrm_pretrain.sh
num_gpus=16 arch_size=XL bash scripts/rjob_hrm_pretrain.sh
resume_from=/path/to/pretrain data_path=/path/to/sft_data checkpoint_path=/path/to/out bash scripts/rjob_hrm_sft.sh
ckpt_path=/path/to/checkpoints/run_dir run_only='[GSM8k,MATH]' bash scripts/rjob_hrm_eval.sh
ckpt_path=/path/to/checkpoints/run_dir ckpt_epoch=1 batch_size=16 bash scripts/rjob_hrm_eval.sh
ckpt_path=/path/to/checkpoints/run_dir ckpt_epoch=1 num_gpus=8 batch_size=16 entrypoint=scripts/hrm_eval_fanout_entrypoint.sh bash scripts/rjob_hrm_eval.sh
ckpt_path=/path/to/checkpoints/run_dir ckpt_epoch=1 num_gpus=8 batch_size=16 bash scripts/rjob_hrm_eval_after_epoch.sh
ckpt_path=/path/to/checkpoints/run_dir ckpt_epoch=1 num_gpus=8 batch_size=16 entrypoint=scripts/hrm_eval_fanout_entrypoint.sh eval_config=/mnt/shared-storage-user/quxiaoye/HRM-Text/evaluation/config/hrm_mmlu_pro_benchmarking.yaml eval_data_dir=/mnt/shared-storage-user/quxiaoye/HRM-Text/eval_data_hf_parquet hf_datasets_offline=1 bash scripts/rjob_hrm_eval.sh
ckpt_path=/path/to/checkpoints/run_dir ckpt_epoch=1 num_gpus=8 entrypoint=scripts/hrm_eval_fanout_entrypoint.sh eval_config=/mnt/shared-storage-user/quxiaoye/HRM-Text/evaluation/config/hrm_maj_vote_benchmarking.yaml eval_data_dir=/mnt/shared-storage-user/quxiaoye/HRM-Text/eval_data_hf_parquet hf_datasets_offline=1 bash scripts/rjob_hrm_eval.sh
```

Default rjob settings are intentionally close to the reference project:

- `gpu_group=moe_gpu`
- `namespace=ailab-moe`
- `gpu=8` per replica for training
- one replica per 8 GPUs for training
- env-check wrapper lowers CPU/memory and disables RDMA because it only
  validates imports
- eval wrapper defaults to one GPU, no RDMA, no gang start, and a separate
  entrypoint so it does not touch a running training entrypoint
- 8-card eval uses one eval replica with eight GPUs; the fanout entrypoint keeps
  all selected GPUs busy by splitting each benchmark's prompt list across GPUs
  instead of assigning one benchmark per GPU
- host network, gang start, RDMA resources
- mounts for `quxiaoye`, `moegroup`, `moegroup2`, and `intern7shared`
- data-prep wrapper defaults `bootstrap=0` because it installs `data_io`
  requirements itself and does not need the training runtime overlay

Preserve these hardcoded cluster paths unless the task is specifically about
changing cluster environment setup.

## Data Expectations

Pretraining expects sampled tokenized data at `data.path`, defaulting to
`/dev/shm/sampled` in `config/data/hlm.yaml`. The directory must contain:

- `metadata.json`
- `tokens.npy`
- `epoch_<n>/inst_start.npy`
- `epoch_<n>/inst_len.npy`
- `epoch_<n>/resp_start.npy`
- `epoch_<n>/resp_len.npy`

The companion `data_io` checkout lives at:

```bash
/mnt/shared-storage-user/quxiaoye/data_io
```

Keep HRM's tokenized pretraining output under this repository:

```bash
/mnt/shared-storage-user/quxiaoye/HRM-Text/data_tokenized_bpe_65k
```

Keep HRM's default sampled pretraining output under this repository too:

```bash
/mnt/shared-storage-user/quxiaoye/HRM-Text/data_sampled_bpe_65k_e4_ctx4097
```

The default data-prep path uses the official cleaned dataset from Hugging Face,
then tokenizes it with `data_io/tokenizer` when Rust/Cargo is available or the
Python fallback otherwise, then samples it to the persistent HRM-Text path:

```bash
stage=all bash scripts/rjob_hrm_prepare_data.sh
```

The data-prep wrapper intentionally defaults `HF_ENDPOINT` to
`https://huggingface.co`; the login environment's `hf-mirror.com` setting can
list repos but fails on actual file downloads with the installed Hub client.
Override with `hf_endpoint=...` only after confirming `hf download` works.
Data-prep rjobs also pass the cluster HTTP(S) proxy into the container because
compute nodes otherwise cannot reach Hugging Face file downloads. Keep PJLab
internal domains in `NO_PROXY` so pip can still use the internal mirror. CPU
rjobs cannot use host networking on this platform; if a download must run under
rjob with host networking, explicitly request an 8-GPU task with
`num_gpus=8 host_network=true`.
If CPU rjobs cannot use host networking, use
`scripts/local_hrm_download_cleaned.sh` from the login machine to download the
cleaned dataset onto shared storage, then resume the rjob flow at
`stage=tokenize`. `scripts/local_hrm_after_download.sh` can watch the download
tmux session and submit tokenization automatically after a successful download.

Do not default to `/dev/shm/sampled` for reusable experiments. `/dev/shm` is
node-local and disappears when the rjob exits, so a separate training rjob may
land on another node and miss the sampled data. Prefer the persistent sampled
path above and pass it as `data_path` for training. Only use `/dev/shm/sampled`
for short same-container debugging when the user explicitly asks.

For multi-node training, generate the persistent sampled dataset once with the
same `epochs` and `context_size` as training, then mount/read that shared path
from all nodes. Re-sampling per node is deterministic but wastes startup time
and can hide node-local path mistakes.

SFT data is prepared with:

```bash
python scripts/prepare_sft_data.py \
  --train input.jsonl \
  --tokenizer /path/to/tokenizer.json \
  --output /path/to/sft_data \
  --epochs 5
```

`--epochs` must match the training config's `epochs`.

## Training Guardrails

- Do not submit a long pretraining or SFT job unless the user explicitly asks.
- Use `dry_run=true` on a wrapper when changing launcher options.
- Run CPU-only `scripts/rjob_hrm_env_check.sh` before using a new image, then
  run `num_gpus=8 bash scripts/rjob_hrm_env_check.sh` before the first real
  training job when GPU quota is available.
- rjob stdout/stderr is also written to `rjob_logs/` by `hrm_entrypoint.sh`;
  inspect that directory if `rjob logs` is empty or flaky.
- Keep checkpoints on shared storage when training with more than one node.
- Keep cleaned `data_io` outputs and HRM tokenized outputs on shared storage;
  keep reusable sampled data on shared storage under HRM-Text, not in
  `/dev/shm`.
- Pass Hydra overrides directly through `extra_args` or the wrapper-specific
  env vars instead of editing default configs for one-off experiments.
- For SFT, require `resume_from`, `data_path`, and `checkpoint_path`.
- For evaluation, require `ckpt_path`; pass benchmark subsets with
  `run_only='[GSM8k,MATH]'` and lower memory use with `batch_size=16`.
  The HRM evaluation entrypoint defaults `eval_workdir` to
  `/mnt/shared-storage-user/quxiaoye/data_io/tokenizer` because current
  checkpoints store the BPE tokenizer path as `../trained_tokenizers/...`.
- For all-benchmark 8-card evaluation, prefer
  `entrypoint=scripts/hrm_eval_fanout_entrypoint.sh num_gpus=8`; it data-shards
  every benchmark across GPUs and avoids the low-utilization pattern of one GPU
  per benchmark.
- When waiting for a training epoch before evaluation, use
  `scripts/rjob_hrm_eval_after_epoch.sh` or equivalent logic. Checkpoint
  readiness means `fsdp2_epoch_<N>/.metadata` exists, all expected
  `carry_epoch_<N>.<rank>.pt` files exist, and file sizes/mtimes have stayed
  stable across repeated checks.
- W&B is used by `pretrain.py`; make sure credentials are available in the job
  environment before long runs.

## Cluster Lessons Learned

- Tokenization and sampling are different stages. The expensive one-time
  tokenizer output is `data_tokenized_bpe_65k`; `sample_tokenized.py` only
  builds `tokens.npy`, `metadata.json`, and `epoch_*` arrays for training.
- The Python tokenizer fallback must be treated as an emergency path until it
  passes a small-sample parity check against the Rust tokenizer with
  `scripts/compare_tokenized_outputs.py`. The persistent data used here was
  produced by the official Rust tokenizer, not the fallback.
- The official Rust tokenizer should run with `tokenizer_impl=rust bootstrap=0
  cpu=5`; exposing many CPUs can over-parallelize parquet files and blow up
  memory because the tokenizer holds each tokenized file before writing.
- Rust/Cargo is installed in `/mnt/shared-storage-user/quxiaoye/.hrm-rust` with
  a Cargo mirror. Keep `CARGO_HOME`, `RUSTUP_HOME`, and mirror env vars in rjob.
- Tokenized outputs created by rjob may be root-owned. If local `mv` fails with
  permission denied, use a short CPU rjob to move/verify the directory instead
  of copying hundreds of GB.
- Do not edit an entrypoint script while an rjob is still executing it from
  shared storage. Bash can keep reading the mounted script as it runs; a data
  prep job can finish `sampled_output_ok` and still fail at script EOF if the
  file changed underneath it. Treat the data as usable only after validating the
  sampled files, and use a fresh rjob for the edited script.
- The login node may fail to `np.load(..., mmap_mode="r")` very large GPFS
  `.npy` files with `OSError: [Errno 19] No such device`, even though the data
  prep rjob successfully used `open_memmap` on a compute node. For local
  validation, parse `.npy` headers or check files structurally; verify compute
  mmap behavior through rjob.
- In bash helpers that use `local -n`, do not name the nameref variable the
  same as the caller's variable. `local -n target_array=target_array` creates a
  circular name reference and silently drops Hydra overrides such as
  `data.path`, causing training to fall back to `/dev/shm/sampled`.
- The internal XTuner fallback image has a `flash_attn_interface`
  `_flash_attn_backward` wrapper that uses the older keyword `causal` instead
  of `is_causal` and requires `softmax_scale`; the observed signature also
  accepts `window_size`, `softcap`, `deterministic`, and `sm_margin`. Keep
  `models/flash_attention_prefixlm_v2.py` compatible with both APIs and pass an
  explicit `q.shape[-1] ** -0.5` scale through forward and backward so the two
  API paths cannot diverge silently.
- CPU rjobs cannot rely on host networking here. Downloads that need host
  network should either run locally on the login node or request an 8-GPU rjob
  with `host_network=true`; sampling/tokenization should not need host network.
- `hf-mirror.com` can list repos but has failed for actual cleaned-data
  downloads with the installed Hub client. Use `https://huggingface.co` unless
  a download test proves otherwise.
- `rjob logs` requires a target type, e.g. `rjob logs job <name>` or
  `rjob logs replica <replica>`. In STARTING it can return empty-log errors;
  also inspect `rjob_logs/` and `data_prep_logs/`.
- For first 8-card bring-up, use README's L-size settings:
  `arch_size=L global_batch_size=172032 extra_args='lr=2.5e-4'`. XL is the
  two-node reference default.
- Use `WANDB_MODE=offline` for bring-up if credentials are uncertain. Switch to
  online only when the run is meant to be recorded.
- The persistent sample build for
  `/mnt/shared-storage-user/quxiaoye/HRM-Text/data_sampled_bpe_65k_e4_ctx4097`
  produced 18 files and about 673G: `tokens.npy`, `metadata.json`, and four
  epoch directories. Token writing took about 47 minutes, epoch index generation
  about 43 seconds, and report generation about 6 seconds on the observed CPU
  rjob.
- `num_gpus=8` only allocates eight GPUs for an eval rjob; it does not make a
  single `evaluation.main` process use all eight. Avoid the naive strategy of
  one benchmark per GPU for full eval because benchmark sizes vary and the job
  leaves GPUs idle after short benchmarks finish. Use prompt-level sharding
  within each benchmark and aggregate generations before computing metrics.
- Epoch-gated eval watchers should write idempotency markers under `rjob_logs/`
  rather than under checkpoint directories. Checkpoint directories are often
  root-owned because training rjobs create them.
- For long-lived local watchers, run them in tmux and tee logs into
  `rjob_logs/`. If you change watcher submission logic, restart the watcher;
  if you only change the future eval entrypoint path it will be picked up by the
  rjob when the watcher eventually submits.
- Always `dry_run=true` a new rjob eval launch shape before leaving a watcher to
  submit it automatically.
- Local watchdogs that run under `set -u` should use `bash -c`, not `bash -lc`.
  On the login node, the login-shell profile has referenced `ZSH_VERSION`
  without a default and can make a watcher exit before it writes its own log.
- Keep auto-submitted eval job names short, e.g. `hrm-eval-e1-0602`. The rjob
  client combines the job name with `generated-task-0` for a Kubernetes label,
  so descriptive checkpoint-derived names can exceed the 63-character label
  limit even before the job is submitted.
- Full benchmark eval rjobs should not depend on live Hugging Face Hub dataset
  resolution. Compute-node `datasets.load_dataset(...)` has failed here with
  proxy errors even when direct file downloads from the login node worked.
  Download the required benchmark parquet files to shared storage with
  `scripts/download_eval_data.py`, then launch eval with
  `eval_data_dir=/mnt/shared-storage-user/quxiaoye/HRM-Text/eval_data_hf_parquet`
  so `HRM_EVAL_DATA_DIR` is passed into the rjob.
- When using local parquet eval data, setting `HF_DATASETS_OFFLINE=1` is useful
  because it proves benchmark construction is not secretly falling back to Hub
  access. Missing local mappings should fail fast before model inference.
- Python 3.12 evaluates stricter `typing` generics at import time. Use
  `Generator[yield_type, send_type, return_type]`; a two-argument
  `Generator[...]` annotation can kill all eval shards before inference starts.
- Current HRM checkpoints may store tokenizer paths like
  `../trained_tokenizers/bpe/tokenizer.json`. Eval runs from
  `/mnt/shared-storage-user/quxiaoye/data_io/tokenizer` so this resolves to the
  shared `data_io/trained_tokenizers` directory. Resolve the path before calling
  `AutoTokenizer.from_pretrained`, and pass the tokenizer directory rather than
  the raw `tokenizer.json` file path.
- `torch.compile` can fail on eval prefill/decode with FA3 custom ops because
  Dynamo fake tensors do not support `flash_attn_3.fwd.default` in this image.
  Keep inference compile optional with `HRM_EVAL_TORCH_COMPILE=1`; default to
  eager execution for bring-up and correctness runs.
- Local `py_compile` can fail because rjobs create root-owned `__pycache__`
  files under the repo. Use a writable temporary prefix, e.g.
  `PYTHONPYCACHEPREFIX=/tmp/hrm_pycache_check python -m py_compile ...`, rather
  than changing ownership of generated cache files during debugging.
- MMLU-Pro and AIME are not part of `evaluation/config/hrm_benchmarking.yaml`.
  Use `evaluation/config/hrm_mmlu_pro_benchmarking.yaml` for MMLU-Pro and
  `evaluation/config/hrm_maj_vote_benchmarking.yaml` for AIME majority voting.
  MMLU-Pro local data is `TIGER-Lab/MMLU-Pro/data/test-00000-of-00001.parquet`;
  AIME25 local data is `math-ai/aime25/test.jsonl`, so the local eval loader
  must use the `json` dataset builder for AIME instead of assuming every cached
  benchmark file is parquet.
- For 64x8 MoE speed work, precision is a hard gate. Run local equivalence and
  the CUDA/bfloat16 rjob gate before treating a kernel, dispatch, or FSDP/EP
  change as usable. Do not trade away router fp32 softmax, top-k=8, routing
  weight normalization, or index-add aggregation semantics for speed.
- The current fastest safe 64x8 MoE path is grouped Triton expert compute with
  `HRM_MOE_TRITON_AUTOTUNE=1` and `HRM_MOE_TRITON_SM_MARGIN=8` or `16`.
  `HRM_MOE_TRITON_BLOCK_M=64` and `SM_MARGIN=32` were slower in same-shape
  smoke runs. Keep the wrapper env parsing tolerant of empty string values
  because rjob launchers pass unset env vars explicitly.
- CUTLASS `grouped_gemm` passed the CUDA/bfloat16 equivalence gate in the
  XTuner fallback image, but the training smoke was much slower than Triton
  because backward/wgrad dominated. Do not switch the default MoE backend to
  CUTLASS without a new precision gate and a faster same-shape smoke.
- Do not use `torch.compile` for MoE `train_batch` by default. The force-compile
  smoke caused graph breaks/recompilation and very large optimizer-step times.
  Keep MoE compile disabled unless a fresh gate proves both correctness and
  speed.
- Expert Parallel all-to-all equivalence is not sufficient. The custom
  autograd all-to-all path passed distributed correctness, but multi-step
  training still stalled on the second backward. EP changes need a multi-step
  training gate, not just a forward/backward equivalence script.
- Keep the MoE dispatch `torch.sort(..., stable=True)` unless a new same-shape
  smoke proves otherwise. A default-sort experiment passed numerical
  equivalence but slowed the Triton path, so it was reverted.
- Use `HRM_MOE_PROFILE=1` only for diagnosis. It inserts CUDA events and
  synchronizes phase boundaries, so the absolute step time is slower than a
  normal smoke; use it for breakdown, not headline speed. The current Triton
  64x8 profile shows forward dispatch/combine and forward grouped GEMMs are
  more promising than backward grouped GEMM. Backward `grad_input_gemm` and
  `grad_weight_gemm` were small in the measured steady step.
- Do not replace `flat_token_idx = arange(...).repeat_interleave(top_k)` with
  `sort_idx // top_k` just because it is algebraically equivalent. That
  token-index experiment passed local and CUDA/bfloat16 equivalence but slowed
  the same-shape Triton smoke, so it was reverted.
- Do not move the routing weight multiply before the expert down projection in
  HRM MoE just because it is algebraically equivalent. The pre-weight attempt
  passed local CPU/fp32 equivalence but failed the CUDA/bfloat16 gate on router
  gradients (`hrm-moe-preweight-eq-06050027`), with max absolute difference
  0.1640625 against `atol=5e-2`. MoE precision is sensitive to bf16 rounding
  placement; keep routing weights applied after the down projection unless a
  new fused forward/backward kernel passes the full CUDA/bfloat16 gradient
  gate.
- GitHub directions for future MoE speed work: MegaBlocks supports continuing
  grouped-GEMM/block-sparse work; Tutel and DeepEP are more relevant for
  expert-parallel communication and parameter/communication isolation; use
  TorchTitan mainly as a reference for FSDP2 organization and observability.
  Any imported idea still needs HRM local equivalence, CUDA/bfloat16 gate, and
  same-shape training smoke before adoption.
- Additional GitHub/kernel directions for the 1.2x dense MoE target: PyTorch's
  persistent cache-aware grouped GEMM work supports further Triton scheduler,
  grouped launch, and TMA-style tuning; DeepGEMM is a possible future backend
  only if the rjob image/toolchain matches and BF16 training gradients pass the
  HRM gate; fused gate/up or fused dispatch-combine kernels must include
  training backward validation, not just faster forward timing.
- Do not directly set `HRM_MOE_TRITON_BLOCK_M=256` with the current
  `num_stages=3` Triton grouped GEMM configs. The CUDA/bfloat16 gate job
  `hrm-moe-bm256-eq-06050045` failed at Triton compile time with shared memory
  `Required: 327704, Hardware limit: 232448`. Larger block-M exploration needs
  lower-stage configs or OOR-tolerant autotune first, then the normal
  CUDA/bfloat16 equivalence gate.
- `HRM_MOE_TRITON_BLOCK_M=256` with `num_stages=2` still OORed in
  `hrm-moe-bm256s2-eq-06050051` (`Required: 262160, Hardware limit: 232448`).
  `num_stages=1` passed the CUDA/bfloat16 gate
  (`hrm-moe-bm256s1-eq-06050054`) but the same-shape smoke
  `hrm-moe64x8-bm256s1-06050056` was slower than the current best
  (second-step core about 0.469s vs 0.389s). Do not adopt BM256 stage-1 as a
  default; prefer tuning the BM128 path or narrower grouped-GEMM configs next.
- When a long rjob is reading one worktree, use a separate git worktree for
  MoE speed code experiments. The 32-card run `hrm-moe32g-gt06050041` reads
  `/mnt/shared-storage-user/quxiaoye/HRM-Text-moe64x8`; speed tuning moved to
  `/mnt/shared-storage-user/quxiaoye/HRM-Text-moe64x8-tune` so code edits do
  not perturb the running training job.
- Evaluate MoE checkpoints from the same MoE worktree that produced them; the
  dense HRM-Text checkout may not have the experimental MoE architecture and
  config classes needed to load the checkpoint. The shared monitor script can
  be reused from the main checkout with
  `HRM_MONITOR_REPO_DIR=/mnt/shared-storage-user/quxiaoye/HRM-Text-moe64x8`
  so rjob evals launch from this worktree and write summaries/docs here.
- When launching SFT from this MoE worktree, pass
  `repo_dir=/mnt/shared-storage-user/quxiaoye/HRM-Text-moe64x8` explicitly.
  The copied `scripts/rjob_hrm_common.sh` still defaults to the dense checkout
  path, and forgetting `repo_dir` can start SFT from code that does not match
  the MoE checkpoint. For UltraData SFT, use
  `scripts/local_ultradata_moe64_sft_after_prepare.sh` so the prepared HRM SFT
  layout is header-validated before launch, `resume_epoch` is pinned to the
  requested stable checkpoint with all 32 carry files, and SFT starts from EMA
  weights with a fresh optimizer via `weights_only_resume_from_ema=true`.
  For the UltraData MoE64 SFT run, do not start from the currently latest
  checkpoint if it is only epoch 3; set `SFT_RESUME_EPOCH=4` and wait for
  `hrm-moe32g-sm16-06050339/fsdp2_epoch_4` to become stable.
- Keep UltraData MoE SFT monitors alive after rjob submission. The watcher must
  inspect rjob state and final checkpoint artifacts until `fsdp2_epoch_5` plus
  all 32 `carry_epoch_5.*.pt` files are stable; a successful submit command is
  not enough. On failure, retry with suffixed job/checkpoint paths such as
  `hrm-moe64-sft-ultra0607-r2` and
  `checkpoints/hrm-moe64x8-ultradata-sft-0607-r2` so partial failed attempts do
  not obscure the next run.
- This worktree's `rjob_logs/` may be root-owned because many MoE rjobs wrote
  logs from containers. If a local monitor cannot create its state/log there,
  put the local monitor state and tee log in a writable checkout such as
  `/mnt/shared-storage-user/quxiaoye/HRM-Text/rjob_logs/`, while keeping
  `fanout.output_dir` under the MoE worktree for container-written summaries.
- Large checkpoint directories created by rjob containers may also be
  root-owned. After MoE speed measurements are documented, clear disposable
  speed-test checkpoints with a short CPU rjob that mounts `quxiaoye` and runs
  exact-path `rm -rf`; do not rely on login-node deletion. Preserve active
  long-run checkpoint paths such as `hrm-moe32g-sm16-06050339`.

## Local Validation

CPU-only checks can run locally, for example:

```bash
python scripts/test_prepare_sft_filter.py
```

GPU, distributed, FA3, and FSDP2 behavior should be validated through rjob.

## Nord Verda single-GPU pilot (2026-09-20)

- `nord_pilot/` is a separate Verda-only technical test; do not use the PJLab rjob
  defaults for it. User explicitly selected HRM + MoE and a bounded first GPU test.
- `pretrain.py` max_steps exits without saving and its existing resume helper does
  not restore loop position. The separate pilot saves complete state and fingerprints.
- CPU tests opt into mathematical PrefixLM attention using NORD_REFERENCE_ATTENTION=1;
  CUDA pilot forbids that switch and gates native FA3/Triton gradients first.
- Verda shutdown does not stop billing. Release only the dedicated test instance,
  with volume_ids=[] so OS/checkpoint volumes remain. Never put API credentials in
  the mounted repository. Watchdog tests are mocked, not a live provider validation.
- Fixture data and tokenizer are generated by `nord_pilot/prepare.py`, ignored by git
  and bundled separately in the handoff archive. They are not a production dataset.

- Verda creation endpoints may return an unquoted UUID despite application/json schema; parse only valid UUID strings as the non-JSON fallback. After ambiguous mutation responses, list resources before retrying.

- Packed FA3 PrefixLM requires prefix_lens with numseqs+1 entries and a terminal zero; the shifted cumulative lengths use this sentinel. CPU references must enforce the same metadata contract.

- Single-device autocast does not cast the custom Triton expert input. Pilot runner explicitly casts MoE boundary activations to BF16 in a pre-hook while retaining FP32 master weights/residuals; apply the same setup on export reload.

- The first H100 pilot at revision b8aa07d passed native kernel parity, exact 213-tensor recovery and export, but 200 steps overfit the tiny fixture (validation 1.7249 -> 2.1436). Preserve the strict failure; technical learning is not product quality. Retain an earlier export and use best-checkpoint selection/early stopping with real validation data for subsequent work.

- Follow-up pilot validation must cover every held-out example once, weighted by target tokens. Persist best weights, patience counters and last evaluation step through resume; a stopped checkpoint must remain stopped. Selection uses fixed global evaluation steps to preserve split-run equivalence.
- Audit changed familiar facts, unseen fact values and changed source IDs separately. Fixture-specific answer validation can repair citations or abstain but is not evidence of general reasoning or a production RAG verifier.

- Long continuations should start from a verified best export with `--init-export`
  (weights-only, fresh optimizer, strict config/tokenizer/parent-SHA validation) so
  later checkpoints still carry the full exact-resume fingerprint. A 9,000-step run
  over ~17M tokens stayed well inside the 1-hour/$5 guard, so the binding constraint
  on longer schedules is budget, not single-H100 throughput.
- Treat falling validation and held-out loss as a technical signal only. Across both
  the 600-step and 9,000-step runs, raw completions stayed repetitive and wrong; never
  present loss improvement as capability, and never call the 69.2M pilot a 0.6B model.
- Verify a downloaded results archive against its remote SHA-256 before deleting the
  instance, and re-load every exported checkpoint locally to confirm all floating
  tensors are finite. The console Credentials page keeps rendering a stale grid after
  a reload; only trust the rendered empty-state row when revoking temporary credentials.
- Do not assume the browser debugger attaches to an existing console tab. Open a fresh
  agent tab for Verda console work, and treat the temporary Cloud API key as revoked
  only once the page shows "You currently have no cloud API keys".

- Bounded held-out scoring is now mandatory for paid runs. `evaluate_corpus.py` gained
  `--test-records` and `--disjoint-limit`: scoring the whole built corpus means 151M
  target tokens, roughly 2.4 hours of forward passes per model on one H100, and the
  unbounded leak check reads all 18 GB of train.jsonl. When either bound is active the
  report says it scored a prefix and that leakage inspection was a spot check, so a
  capped run never reads as a full audit.
- The scaled paid run is `work/verda-scale-private/run-scale.sh` (free steps 1-3, then
  provision, run, verify, delete). `nord_pilot/gpu_scale_test.sh` is the on-GPU plan:
  kernel gate, 20-vs-10+10 recovery equivalence, training from a step-0 export, then
  bounded held-out scoring against that same untrained initialization. Its PASS gate is
  exercised without a GPU by `work/verda-scale-private/test-pass-gate.py`, which runs the
  shipped heredoc against synthetic results; 5/5 cases behave as intended.
- Ship a bounded corpus prefix, not the full 21 GB. `stage-corpus.py` takes the first N
  records in file order, which is the same order the streaming reader consumes, so a
  prefix is equivalent for a short run. Measured: 2M train records = 447M tokens = 2.9 GB,
  gzipped to 1.04 GB at `gzip -1` (about 212 MB/s here), i.e. roughly two minutes of upload
  onto the H100. That is why no CPU staging instance is required.
- `preflight.py` indexes and decodes the staged corpus with the trainer's own `JsonlCorpus`
  on CPU for free; 2M train / 40k valid records index in 1.2 s at 0.13 GB peak RSS. Only
  trust its memory number after the macOS/linux `ru_maxrss` unit fix (bytes vs kilobytes).
- Budget reality, corrected by measurement (2026-09-21). The earlier $152-304 figure for a
  full 2.81B-token pass came from the 69M run's 17,168 tok/s, which was launch-bound: one
  maximum-length record per step. The batch probe on the paid H100 measured 110,318 tok/s at
  batch 32,768 (12.63% MFU) versus 6,058 tok/s at batch 2,048, so `cost_model.py --plan` now
  puts a full 2.81B-token pass at about $24 flat/$24 sqrt at every size in the family. Batch
  size, not parameter count, was setting the price. Treat $24-ish per full pass as the working
  number and re-measure before trusting it at a new size.
- Paid steps run unattended through `work/verda-scale-private/`, which holds the Verda
  `credentials.json` (mode 600, never bundled or printed). The guard rejects any credential
  file that is group- or world-readable, and the packaging step excludes it from the upload.

- The second paid scaled run (revision `100bd18`, 60-minute container budget, `1H100.80S.32V`
  in FIN-02, local and remote guards at 2h/$8) completed its full 10,000-step schedule:
  187,219,968 parameters, valid loss 10.97629937151646 -> best 3.5718350700864128 at step
  9750, held-out fresh test 10.97513985581203 -> 3.6312330527437875 over 12,000 bounded
  records, 326,461,487 input tokens at 115,219 tok/s, wall 3,265 s, peak 46.8 GB. Spend
  $5.08364. `capability: not established` and all six fixed completions were still repetitive
  and wrong (for example `2 + 3 =` -> `3` then a repeated `- n = 3 = 3`), so report the loss
  number as a technical result only.
- A run that ends because its cosine finished is different from one the clock cut: `stop_reason:
  completed` means the step cap was reached. At equal steps the second run trailed the first
  (step 4500: 3.7404 run1 vs 3.8139 run2) because the learning-rate warmup, cosine decay and
  `bp_steps` warmup now actually affect training, and it finished lower only by running 10,000
  steps instead of 4,505. Compare like for like before crediting a schedule change.
- Verda reuses IP addresses across instances, so a `known_hosts` entry pinned by an earlier
  run makes `StrictHostKeyChecking=accept-new` abort instead of re-learning the key, and the
  bootstrap retry loop spins while the GPU bills. Clear the file at the start of every run.
  Observed cost of the stale entry: roughly $0.8 of H100 time across ~28 attempts.
- macOS has no `setsid` binary, so a `nohup` job that is merely backgrounded dies with the
  shell that started it. `work/verda-scale-private/launch-detached.py` double-forks into a new
  session; use it for anything that must outlive the invoking shell.
- Ship `trained/export.pt`, not `trained/latest.pt`. `latest.pt` carries the optimizer state
  (3.7 GB against 749 MB of weights) and a wall-clock-bounded run never resumes from it, so
  that extra weight is downloaded while the H100 is still billing and then never read.
  Dropping it took the archive from 5.25 GB to 2.25 GB: 52 files, 2,252,554,748 bytes, every
  SHA-256 matching, and `trained/best-export.pt` reloaded on CPU here and reproduced the same
  step-9750 text the GPU produced.
- A push to the upstream `XiaoYee/HRM-MoE` is denied (403) for `albin-holmgren`, so the branch
  is published to the fork `albin-holmgren/HRM-MoE` instead.
- The third paid scaled run (revision `99c0dc4`, 75-minute container budget, weights-only
  continuation of run 2) is the counterexample that the other two should be read against. Its
  learning was real: valid loss fell from 3.5677182 (the parent itself) to 3.4613595 at step
  3250, monotonically, with all eight experts routed and no NaN. Its weights did not survive.
  The batch probe had measured 65536 at a shallow backprop depth (49.77 GB at 8 steps) and the
  training run reached the deep end of the warmup at step 3400, where `F.cross_entropy` tried
  to allocate 7.99 GiB against 73.96/79.18 GiB in use and the process exited with a traceback.
  `best-export.pt` was then written only on the post-loop path, so the measured improvement
  was downloaded as numbers with no weights behind them and `technical_pass` is null. Spend
  $3.39595. Treat a probe that does not measure the depth the run reaches as no measurement.
- A memory wall must not be allowed to cost the session's weights. Three fixes now hold that
  line and each has a unit test: `batch_probe.py` pins `MEASUREMENT_BP_STEPS=5` and rejects any
  candidate above `MAX_PEAK_FRACTION=0.72` of the card even when it is the fastest; `run.py`
  writes `best-export.pt` the moment a better validation loss is found and adds a weights-only
  `latest-weights.pt` beside the optimizer-carrying `latest.pt`; and an out-of-memory stop is
  caught and recorded as `status='out_of_memory'`, still writing the exports, the summary and
  the held-out score. What a paid run is worth is decided by what is on disk after a crash, not
  by the loss it had reached.
