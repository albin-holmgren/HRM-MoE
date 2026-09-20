# Nord HRM + MoE: first Verda technical test

**Executed on Verda H100 on 2026-09-20. Overall validation gate FAILED due to fixture overfitting.**

Native FA3/Triton checks, 200 training steps, exact 213-tensor resume, exported-model
reload and the 2,048-token stress test passed. Held-out fixture loss improved from
11.1183 to 1.7249 at step 40, then worsened to 2.1436 at step 200. Preserve this
failure; it is not a sellable model or a full pass. The step-40 export and final
checkpoint are saved separately. The runtime was PyTorch 2.11.0+cu128 / CUDA 12.8
on one NVIDIA H100 80GB. See the accompanying first-test results report for costs
and cleanup confirmation.

This package uses the actual upstream HRM recurrent modules and MoE implementation,
with a separate single-GPU runner. It does not use pretrained weights. It is a
technical test, not the final language-model training run. Mamba/diffusion are excluded.

## Exact first-test shape

- 94,404,608 total parameters, 56,655,872 unique active parameters per token.
- H and L each have four Transformer blocks, hidden width 512, four attention heads.
- Eight experts, top two selected, expert intermediate width 512; normalized routing,
  FP32 router softmax and auxiliary balancing coefficient 0.001.
- Two H cycles and three L cycles. Reuse means compute exceeds the unique active count.
- 32,768 output vocabulary slots; technical fixture tokenizer uses 487 actual tokens.
- 2,048-token packed batches; a separate stress pass fills one sequence to 2,048.
- FP32 master parameters, BF16 CUDA autocast, upstream AdamATan2 and EMA. Constant
  2e-4 learning rate and fixed five backpropagation steps for this technical test;
  this does NOT reproduce the published pretraining warmup/training recipe.
- About 1.76 GiB for FP32 weights, gradients, two optimizer moments and EMA alone.
  Activations, temporary kernels, logits and allocator overhead are additional and
  must be measured on H100. Do not use that state-only number as peak VRAM.

## What is tested

1. Native FlashAttention-3 PrefixLM forward/backward versus a mathematical reference.
2. Triton expert forward/backward versus upstream grouped PyTorch computation.
3. Forty uninterrupted steps versus twenty + fresh-process resume to forty, comparing
   model, optimizer, RNG and data position; GPU tolerance 1e-5. A failure stops the pilot.
4. Fresh-process export reload and greedy generation; generated text has no quality claim.
5. A ten-step 2,048-token sequence stress pass.
6. Continued learning to step 200, reporting loss, expert usage, training throughput
   and peak allocated memory. Tiny fixture validation loss must decrease.

The synthetic fixture contains 128 English/Swedish training records and 32 held-out
records about fictional shop opening times. The tokenizer is trained only on the
training split. This is our own generated fixture, not licensed external text.
Neither these examples nor the resulting tokenizer/checkpoint are a production corpus
or product evaluation. Throughput is workload-specific: measure representative real
training data before pricing the main run. No benchmark leaderboard score is claimed.

## Why there is a separate runner

At upstream revision `d33a41e5d11ce163cee3ed99efb3060e5ce35633`, the default trainer's
`max_steps` exit skips saving, and its resume helper loads weights/optimizer without
restoring the loop position. The pilot saves full state plus RNG, data cursor, source
hashes, config and dataset fingerprints atomically, retaining the previous checkpoint.
It rejects changed configurations/data/code on resume. It uses one device without
FSDP; this test does not establish distributed/FSDP recovery.

The only upstream model edit is explicit `NORD_REFERENCE_ATTENTION=1` selection for
local CPU mathematical tests. GPU runs refuse that mode and use native FA3/Triton.
CPU tests alone are not evidence that those GPU kernels work. They were subsequently
checked on the H100 as reported above. The single-device runner explicitly casts
MoE boundary activations to BF16, preserving FP32 master weights and residuals.
Packed prefix lengths include the terminal zero required by native FA3.

## Local verification (already performed)

Python 3.12.13, PyTorch 2.8.0, numpy 2.2.6, pydantic 2.11.7, einops 0.8.1,
tokenizers 0.21.4. Install these in a disposable environment, then from package root:

```bash
python -m unittest nord_pilot.test_local nord_pilot.test_guard -v
bash -n nord_pilot/launch_verda.sh nord_pilot/gpu_test.sh
bash nord_pilot/launch_verda.sh
```

The local test uses a tiny 4,334,592-parameter HRM-MoE with the same recurrence/routing
pattern. It verifies exact fresh-process recovery and export, prefix-mask isolation,
learning, configuration mismatch rejection and overwrite protection. Provider requests
are mocked in watchdog unit tests; real deployment and cleanup are tracked in the
first-test results report.

## Verda deployment prerequisites

Choose a **new dedicated one-H100 80GB on-demand VM**, PAY AS YOU GO, with hostname
`nord-hrm-pilot-<unique-suffix>`. Use a CUDA-driver + Docker image with NVIDIA Container
Toolkit and a driver compatible with the pinned CUDA 12.8 runtime. Use approximately
150 GiB OS storage initially, sufficient for the environment and pilot checkpoints;
verify actual free space. Retain all volumes on eviction/deletion. No 8-GPU node needed.

Check the final price is at most $3.50/hour for the single GPU, and obtain actual
storage hourly charges and tax fraction. The recent public GPU price is $3.35/hour.
150 GiB at $0.20/GiB-month is $30/month; approximate hourly cost on a 730-hour basis is
$0.0411. Use the console's actual rate in the launcher rather than assuming this.

A **$25 first-test allocation** is the target, not a promise of uninterrupted provider
availability. The independent watchdog stops by two hours since VM creation or $20
estimated consumed charges, whichever is earlier. At $3.35, two hours GPU rent is
$6.70 before storage/tax. Setup and image pull time count. This tighter first-test
window supersedes the earlier proposed four-hour maximum. If the window is insufficient,
stop, preserve logs and inspect the cause; do not automatically extend it.

Use a project-scoped Verda Cloud API credential permitted to read and delete this
instance. On the test host, keep it outside this package at a root-readable 0600 file:

```json
{"client_id":"YOUR_ID","client_secret":"YOUR_SECRET"}
```

Do not paste secrets into chat, put them in the repository, or mount them into the
GPU container. `verda_guard.py` reads them only in its separate host service.

Copy/extract the prepared archive to `/opt/nord-hrm-pilot` on the new VM. The root
volume is retained, so logs and checkpoints survive deletion of compute. Check
`nvidia-smi`, Docker and available disk space before launching. The launcher deliberately
does not create an instance or buy credits.

Dry-run first:

```bash
cd /opt/nord-hrm-pilot
bash nord_pilot/launch_verda.sh
```

When the test is actually authorized and the dedicated VM exists, run as root:

```bash
bash nord_pilot/launch_verda.sh --execute INSTANCE_UUID nord-hrm-pilot-UNIQUE \
  /root/verda-pilot-credentials.json ACTUAL_STORAGE_DOLLARS_PER_HOUR ACTUAL_TAX_FRACTION
```

Use e.g. `0` for the tax fraction only if the invoice has no additional tax. The
launcher arms a separate systemd API watchdog before pulling the image. It checks
instance UUID/hostname and single-H100 shape. It mounts only the source/output tree,
uses the pinned AMD64 container digest in `image.txt`, disables container networking,
and runs no package installation during the test. Missing runtime dependencies fail
at the kernel gate rather than starting training. The public image manifest/digest
was verified; its CUDA runtime was executed on the Verda H100, not locally on this Mac.

## Billing stop and recovery

Verda explicitly says **shutdown continues billing**. The watchdog therefore calls
PUT `/v1/instances` with `action=delete`, the exact test UUID, `volume_ids=[]`, and
`delete_permanently=false`. This releases the GPU while retaining the OS/data volumes.
The host guard is outside the Docker timeout and is also triggered on launcher exit.
It starts counting from provider `created_at`, including setup before the launcher.
It checks provider confirmation; errors are retried and logged without credential data.

Monitor this first test from the console. API/network/service failure can defeat an
automatic spending limit; the watchdog is not a provider-enforced financial guarantee.
If cleanup is unconfirmed, delete only the dedicated test VM in the console with **no
volumes selected**. Never rely on balance exhaustion: Verda can delete volumes at zero
balance. Retained volumes continue to cost money after GPU deletion.

Logs: `/var/lib/nord-pilot/INSTANCE_UUID/container.log`, `status.json` and systemd journal
`journalctl -u nord-hrm-guard-INSTANCE_UUID`. Results: package-root `pilot_runs/<timestamp>`.
After completion, retrieve results from the retained volume using another existing host
or a separately budgeted temporary CPU instance. Do not restart an H100 merely to copy
files without counting that cost. Credentials remain on the retained OS disk until
removed; protect access and revoke the test credential after recovery.

## What a pass means

A pass means this particular implementation can train, recover and generate on H100.
It does not mean the model is useful, that the $500 full run will succeed, or that an
HRM-MoE checkpoint matches published 0.6B/1B benchmark results. Review loss, expert use,
short/long-sequence speed and memory before setting the actual training token budget.
The later cumulative $125 product-quality gate still requires real data and task tests.

Sources: https://github.com/XiaoYee/HRM-MoE ; https://api.verda.com/v1/docs ;
https://docs.verda.com/cpu-and-gpu-instances/shutdown-hibernate-and-delete/
