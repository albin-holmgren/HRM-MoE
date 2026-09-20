"""Reproducible cost model for scaling the HRM-MoE pilot on one H100.

Every number here is derived from a measurement or from the model code, and the
assumptions are printed so they can be challenged. It exists because the earlier
planning estimates quoted parameter counts and dollar figures without connecting
them to the recurrence the model actually executes.

Key measured fact (nord-long H100 run, 2026-09-20): 69,238,784 parameters, batch of
2,048 tokens, bp_steps=5, 17,167.84 input tokens/s compute-only. That is 0.87% of the
H100's dense bf16 peak, so wall clock at that batch size is dominated by 32 sequential
small operations and kernel launch overhead, not by arithmetic. Both the flat and the
linear extrapolations below are printed: the truth lies between them, and only a live
run with a realistic batch size can settle it.
"""
import argparse
import json

MEASURED_TOKENS_PER_SECOND = 17167.84
MEASURED_CONFIG = dict(n_layers=8, half_layers=True, hidden_size=512, num_heads=4,
                       moe_num_experts=8, moe_top_k=2, moe_intermediate_size=512,
                       H_cycles=2, L_cycles=3, vocab_size=32768)
H100_BF16_TFLOPS = 989.0
VERDA_H100_USD_PER_HOUR = 3.348

FAMILY = dict(half_layers=True, moe_num_experts=8, moe_top_k=2,
              H_cycles=2, L_cycles=3, vocab_size=32768)


def layer_passes(cfg):
    """Sequential block applications per token.

    The L level runs L_cycles times inside each of H_cycles outer iterations; the H
    level runs once per outer iteration. With half_layers the configured depth is split
    evenly between the two levels.
    """
    per_level = cfg['n_layers'] // 2 if cfg['half_layers'] else cfg['n_layers']
    return (cfg['L_cycles'] * cfg['H_cycles'] + cfg['H_cycles']) * per_level


def train_flops_per_token(cfg):
    h = cfg['hidden_size']
    m = cfg['moe_intermediate_size']
    macs = (4 * h * h + cfg['moe_top_k'] * 3 * h * m) * layer_passes(cfg)
    return 6 * macs  # forward + backward, two FLOPs per multiply-accumulate


def parameter_count(cfg):
    h = cfg['hidden_size']
    m = cfg['moe_intermediate_size']
    per_level = cfg['n_layers'] // 2 if cfg['half_layers'] else cfg['n_layers']
    return (4 * h * h + cfg['moe_num_experts'] * 3 * h * m) * per_level * 2 + 2 * cfg['vocab_size'] * h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus-tokens', type=float, default=2_810_517_880,
                        help='train tokens in the built corpus')
    parser.add_argument('--rate', type=float, default=VERDA_H100_USD_PER_HOUR)
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--plan', action='store_true',
                        help='print how many full-corpus runs of each size fit a budget')
    parser.add_argument('--budget', type=float, default=500.0, help='budget in USD for --plan')
    parser.add_argument('--overhead-per-run-usd', type=float, default=2.5,
                        help='one gate + recovery + held-out scoring pass, roughly a few minutes of H100')
    args = parser.parse_args()

    reference_flops = train_flops_per_token(MEASURED_CONFIG)
    reference_mfu = 100 * MEASURED_TOKENS_PER_SECOND * reference_flops / 1e12 / H100_BF16_TFLOPS
    rows = []
    candidates = [
        ('69M h512 L8', dict(n_layers=8, hidden_size=512, num_heads=4, moe_intermediate_size=512)),
        ('150M h768 L8', dict(n_layers=8, hidden_size=768, num_heads=6, moe_intermediate_size=768)),
        ('260M h1024 L8', dict(n_layers=8, hidden_size=1024, num_heads=8, moe_intermediate_size=1024)),
        ('321M h1024 L10', dict(n_layers=10, hidden_size=1024, num_heads=8, moe_intermediate_size=1024)),
        ('382M h1024 L12', dict(n_layers=12, hidden_size=1024, num_heads=8, moe_intermediate_size=1024)),
    ]
    for name, override in candidates:
        cfg = dict(FAMILY)
        cfg.update(override)
        params = parameter_count(cfg)
        flops = train_flops_per_token(cfg)
        ratio = flops / reference_flops
        # Flat: throughput unchanged, because the measured batch was latency-bound.
        # Linear: throughput falls as 1/FLOPs, i.e. the run is fully compute-bound.
        # Sqrt: a middle case that assumes per-step time grows with sqrt(FLOPs).
        def usd(tokens_per_second):
            return args.rate * args.corpus_tokens / tokens_per_second / 3600
        rows.append(dict(
            name=name, parameters=params, layer_passes=layer_passes(cfg),
            train_flops_per_token=flops, tokens_per_parameter=args.corpus_tokens / params,
            chinchilla_tokens_20x=20 * params,
            usd_flat=usd(MEASURED_TOKENS_PER_SECOND),
            usd_linear=usd(MEASURED_TOKENS_PER_SECOND / ratio),
            usd_sqrt=usd(MEASURED_TOKENS_PER_SECOND / ratio ** 0.5),
        ))
    payload = dict(measured_tokens_per_second=MEASURED_TOKENS_PER_SECOND,
                   measured_mfu_percent=reference_mfu, rate=args.rate,
                   corpus_tokens=args.corpus_tokens, candidates=rows)
    if args.json:
        print(json.dumps(payload, indent=2))
        return
    if args.plan:
        dollar = '$'
        print(f'budget {args.budget:.0f} USD; one full pass over {args.corpus_tokens/1e9:.2f}B tokens per run')
        print(f"  fixed overhead beside training: {args.overhead_per_run_usd:.2f} USD per run")
        print(f"{'config':16s}{'pass_usd':>9s}{'flat':>6s}{'sqrt':>6s}{'lin':>5s}"
              f"{'flat':>7s}{'sqrt':>7s}")
        for row in rows:
            counts = {label: max(0, int(args.budget // (row[label] + args.overhead_per_run_usd)))
                      for label in ('usd_flat', 'usd_sqrt', 'usd_linear')}
            # Tokens actually seen for the affordable flat and sqrt counts: this is what
            # decides how well trained the result is, not the parameter count.
            seen_flat = counts['usd_flat'] * args.corpus_tokens / row['parameters']
            seen_sqrt = counts['usd_sqrt'] * args.corpus_tokens / row['parameters']
            print(f"{row['name']:16s}{row['usd_flat']:9.0f}{counts['usd_flat']:6d}{counts['usd_sqrt']:6d}"
                  f"{counts['usd_linear']:5d}{seen_flat:7.1f}{seen_sqrt:7.1f}")
        print('  flat/sqrt/lin: full-corpus runs affordable under the budget at each extrapolation;'
              ' the last two columns are tokens-per-parameter at that many runs')
        return
    print(f'measured reference: {layer_passes(MEASURED_CONFIG)} sequential layer-passes/token, '
          f'{reference_flops/1e6:.0f} MFLOP/token train')
    print(f'  {MEASURED_TOKENS_PER_SECOND:,.0f} tok/s -> {reference_mfu:.2f}% of H100 bf16 peak'
          f'  => cost grows slower than FLOPs when the batch stays small')
    print(f'  corpus {args.corpus_tokens/1e9:.2f}B train tokens at ${args.rate}/h\n'
          f"{'config':16s}{'params':>8s}{'passes':>8s}{'tok/param':>11s}{'$flat':>9s}{'$sqrt':>9s}{'$linear':>9s}")
    for row in rows:
        print(f"{row['name']:16s}{row['parameters']/1e6:7.0f}M{row['layer_passes']:8d}"
              f"{row['tokens_per_parameter']:11.1f}{row['usd_flat']:8.0f}${row['usd_sqrt']:8.0f}$"
              f"{row['usd_linear']:8.0f}$")


if __name__ == '__main__':
    main()
