"""Autoresearch-style fixed-budget search for the Nord HRM-MoE pilot.

Ported from karpathy/autoresearch's method (one script, a fixed wall-clock
budget, one metric, keep-if-better / discard-if-worse, a results log) onto the
pilot runner that already exists here. The point is to spend GPU money only on
the variants that survived a free local screen.

Not a capability benchmark. Every row is a technical learning result.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = os.environ.get("NORD_PYTHON", sys.executable)

# Metric: best validation loss in nats/token over a FIXED bounded slice of the
# held-out split, at a fixed record count. Lower is better. Fixing the slice and
# the record count is what makes two variants comparable; the vocab is fixed at
# 32768 across the family so this is also like-for-like. Train loss is NOT a
# cross-variant metric here because a deeper model consumes different batches,
# so the held-out number is the one that decides keep/discard.
OVERLAY_KEYS = {
    "n_layers", "half_layers", "hidden_size", "num_heads", "expansion",
    "max_seq_len", "vocab_size", "H_cycles", "L_cycles", "bp_min_steps",
    "bp_max_steps", "bp_warmup_ratio", "norm_type", "norm_eps", "rope_theta",
    "pos_emb_type", "init_type", "attn_type", "moe_num_experts", "moe_top_k",
    "moe_intermediate_size", "moe_norm_topk_prob", "moe_router_aux_loss_coef",
    "moe_implementation",
}


def load_json(path):
    return json.loads(Path(path).read_text())


def write_variant_config(base_config, overlay, out_path):
    unknown = set(overlay) - OVERLAY_KEYS
    if unknown:
        raise SystemExit(f"unknown config keys in overlay: {sorted(unknown)}")
    cfg = dict(base_config)
    cfg.update(overlay)
    Path(out_path).write_text(json.dumps(cfg))
    return cfg


def parse_run_output(stdout, metrics_path):
    """Pull the pre-training parameter line and per-variant expert routing."""
    parameters = None
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and "parameters" in line and "valid_loss" in line:
            try:
                parameters = json.loads(line).get("parameters")
            except json.JSONDecodeError:
                pass
    last_metric = None
    expert_totals = None
    if metrics_path.is_file():
        rows = [json.loads(l) for l in metrics_path.read_text().splitlines() if l.strip()]
        if rows:
            last_metric = rows[-1]
            totals = None
            for r in rows:
                counts = r.get("expert_counts")
                if counts:
                    totals = counts if totals is None else [a + b for a, b in zip(totals, counts)]
            expert_totals = totals
    return parameters, last_metric, expert_totals


def run_variant(config_path, out_dir, args, lr):
    cmd = [
        PYTHON, str(ROOT / "run.py"),
        "--device", "cpu",
        "--config", str(config_path),
        "--out", str(out_dir),
        "--steps", str(args.step_cap),
        "--schedule-steps", str(args.schedule_steps),
        "--lr", str(lr),
        "--batch-tokens", str(args.batch_tokens),
        "--eval-every", str(args.eval_every),
        "--valid-records", str(args.valid_records),
        "--minutes", str(args.minutes),
        "--bp-steps", str(args.bp_steps),
        "--checkpoint-every", str(args.step_cap + 1),
    ]
    env = dict(os.environ)
    env["NORD_REFERENCE_ATTENTION"] = "1"
    started = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT.parent), env=env,
                          capture_output=True, text=True, timeout=args.timeout)
    return proc, time.time() - started


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=Path, required=True,
                    help="JSON list of {name, overlay, lr?} entries; overlay patches the base config")
    ap.add_argument("--base-config", type=Path, default=ROOT / "configs" / "cpu.json")
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True,
                    help="JSONL log of every attempt, kept or discarded")
    ap.add_argument("--minutes", type=float, default=0.5, help="fixed wall-clock budget per variant")
    ap.add_argument("--step-cap", type=int, default=10_000)
    ap.add_argument("--schedule-steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-tokens", type=int, default=256)
    ap.add_argument("--bp-steps", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--valid-records", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=3600.0)
    args = ap.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    base_config = load_json(args.base_config)
    variants = load_json(args.variants)

    best = None
    rows = []
    for entry in variants:
        name = entry["name"]
        overlay = entry.get("overlay", {})
        lr = entry.get("lr", args.lr)
        var_dir = args.work_dir / name
        var_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = var_dir / "config.json"
        write_variant_config(base_config, overlay, cfg_path)
        summary_path = var_dir / "summary.json"
        if summary_path.is_file():
            summary_path.unlink()
        proc, wall = run_variant(cfg_path, var_dir, args, lr)
        summary = load_json(summary_path) if summary_path.is_file() else None
        parameters, _last_metric, expert_totals = parse_run_output(
            proc.stdout, var_dir / "metrics.jsonl")
        if summary is None:
            row = dict(name=name, verdict="error", metric=None, lr=lr,
                       wall_seconds=round(wall, 3), returncode=proc.returncode,
                       stderr_tail=proc.stderr.strip().splitlines()[-5:])
        else:
            metric = summary.get("best_valid_loss")
            if best is None or (metric is not None and metric < best["metric"]):
                verdict = "keep"
                best = dict(name=name, metric=metric)
            else:
                verdict = "discard"
            row = dict(name=name, verdict=verdict, metric=metric, lr=lr,
                       best_step=summary.get("best_step"),
                       initial_valid_loss=summary.get("initial_valid_loss"),
                       tokens_per_training_second=summary.get("tokens_per_training_second"),
                       input_tokens=summary.get("input_tokens"),
                       parameters=parameters,
                       expert_counts=expert_totals,
                       wall_seconds=round(wall, 3))
        rows.append(row)
        print(json.dumps(row), flush=True)

    with args.results.open("a") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(json.dumps(dict(best=best, attempts=len(rows), results=str(args.results))))


if __name__ == "__main__":
    main()

