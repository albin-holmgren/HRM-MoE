"""Decode a saved export with sampling vs greedy on CPU reference kernels.

No cloud calls, no training. Tests whether the repetitive pilot completions are
partly a greedy-decoding artifact. Not a capability benchmark.
"""
import argparse, hashlib, json, os, sys
from pathlib import Path
os.environ['NORD_REFERENCE_ATTENTION'] = '1'
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from models.baselines.hrm_nocarry_bp_warmup import HierarchicalReasoningModel
from models.lm_head import LMHead

PROMPTS = ['Stockholm is the capital of', 'Sverige är ett land i',
           'To calculate the average of three numbers,', 'Water freezes when',
           '2 + 3 =', 'Question: What is the capital of Sweden? Answer:']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pack(seqs, prefixes, device='cpu'):
    cu, inputs, positions = [0], [], []
    for seq in seqs:
        inputs += seq
        positions += list(range(len(seq)))
        cu.append(len(inputs))
    causal = [len(s) - p for s, p in zip(seqs, prefixes)]
    t = lambda x: torch.tensor(x, dtype=torch.int32, device=device)
    batch = dict(inputs=t(inputs), position_ids=t(positions), prefix_lens=t(prefixes + [0]),
                 causal_lens=t(causal), cu_seqlens=t(cu))
    for k, v in dict(total_seqlen=len(inputs), numseqs=len(seqs), max_seqlen_prefix=max(prefixes),
                     max_seqlen_causal=max(causal), max_seqlen_all=max(map(len, seqs))).items():
        batch[k] = torch.tensor(v, dtype=torch.int32)
    return batch, cu


def pick(logits, seen, temperature, top_p, penalty):
    scores = logits.float().clone()
    if penalty != 1.0:
        for tok in set(seen):
            scores[tok] = scores[tok] / penalty if scores[tok] > 0 else scores[tok] * penalty
    if temperature == 0.0:
        return int(scores.argmax())
    scores = scores / temperature
    order = torch.argsort(scores, descending=True)
    probs = F.softmax(scores[order], dim=-1)
    keep = torch.cumsum(probs, dim=-1) - probs < top_p
    keep[0] = True
    filtered = torch.full_like(scores, float('-inf'))
    filtered[order[keep]] = scores[order[keep]]
    return int(torch.multinomial(F.softmax(filtered, dim=-1), 1))


def generate(model, tok, sequences, steps, mode, seed=0):
    torch.manual_seed(seed)
    eos = tok.token_to_id('[EOS]')
    seqs = [list(s) for s in sequences]
    out = [[] for _ in seqs]
    done = [False] * len(seqs)
    temperature, top_p, penalty = mode
    for _ in range(steps):
        batch, cu = pack(seqs, [1] * len(seqs))
        _, logits = model(None, batch, bp_steps=5)
        last = logits[torch.tensor(cu[1:]) - 1]
        for i in range(len(seqs)):
            if done[i]:
                continue
            tok_id = pick(last[i, :tok.get_vocab_size()], out[i], temperature, top_p, penalty)
            seqs[i].append(tok_id)
            out[i].append(tok_id)
            if tok_id == eos:
                done[i] = True
        if all(done):
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--export', type=Path, required=True)
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--tokens', type=int, default=48)
    ap.add_argument('--seeds', type=int, default=3)
    a = ap.parse_args()
    torch.set_num_threads(4)
    tok = Tokenizer.from_file(str(a.data_dir / 'tokenizer.json'))
    ck = torch.load(a.export, map_location='cpu', weights_only=False)
    if ck['tokenizer_sha256'] != sha(a.data_dir / 'tokenizer.json'):
        raise RuntimeError('Export tokenizer mismatch')
    model = LMHead(HierarchicalReasoningModel(ck['config']), ck['config'])
    model.load_state_dict(ck['model'])
    model.eval()
    bos = tok.token_to_id('[BOS]')
    prefixes = [[bos] + tok.encode(p).ids for p in PROMPTS]
    modes = {'greedy': (0.0, 1.0, 1.0), 'sample_t0.8_pp0.95_rp1.15': (0.8, 0.95, 1.15)}
    result = {'export': str(a.export), 'step': ck['step'], 'parameters': sum(p.numel() for p in model.parameters()),
              'tokens': a.tokens, 'modes': {}}
    with torch.inference_mode():
        for name, mode in modes.items():
            seeds = range(1) if name == 'greedy' else range(a.seeds)
            runs = []
            for seed in seeds:
                gen = generate(model, tok, prefixes, a.tokens, mode, seed)
                runs.append([{'prompt': p, 'completion': tok.decode(g), 'tokens': len(g)}
                             for p, g in zip(PROMPTS, gen)])
            result['modes'][name] = runs
        # Held-out loss on a bounded slice to confirm the saved-weights number reproduces.
        rows = [json.loads(l) for l in (a.data_dir / 'test-fresh.jsonl').read_text().splitlines()][:16]
        eos = tok.token_to_id('[EOS]')
        nll = targets = 0
        for off in range(0, len(rows), 4):
            chunk = rows[off:off + 4]
            ys = [r['token_ids'] + ([eos] if r.get('document_end', True) else []) for r in chunk]
            batch, _ = pack([[bos] + y[:-1] for y in ys], [1] * len(ys))
            _, logits = model(None, batch, bp_steps=5)
            labels = torch.tensor(sum(ys, []))
            nll += float(F.cross_entropy(logits.float(), labels, reduction='sum'))
            targets += len(labels)
        result['fresh_test_slice'] = {'documents': len(rows), 'target_tokens': targets, 'loss': nll / targets}
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'modes'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
