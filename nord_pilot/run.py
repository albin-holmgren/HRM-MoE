"""Single-device HRM-MoE technical pilot using upstream model and optimizer.
No pretrained weights, FSDP, external APIs, or automatic provisioning.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def configure_cuda_mixed_precision(model):
    # Autocast handles Linear/FA3 inputs, but not custom Triton expert dispatch.
    # Keep FP32 master weights and residuals; explicitly cast at each MoE boundary.
    import torch
    from models.layers import SparseMoESwiGLU
    def cast_input(module, args):
        if not torch.is_autocast_enabled('cuda'):
            raise RuntimeError('CUDA pilot MoE requires an explicit autocast context')
        return (args[0].to(torch.get_autocast_dtype('cuda')), *args[1:])
    for module in model.modules():
        if isinstance(module, SparseMoESwiGLU):module.register_forward_pre_hook(cast_input)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', choices=['cpu','cuda'], required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--schedule-steps', type=int, default=200)
    ap.add_argument('--resume', type=Path)
    ap.add_argument('--minutes', type=float, default=30)
    ap.add_argument('--checkpoint-every', type=int, default=10)
    ap.add_argument('--batch-tokens', type=int, default=2048)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--stress-context', action='store_true', help='Fill each prefix to the context limit for a kernel/memory stress test, not quality training')
    return ap.parse_args()

def main():
    a = parse_args()
    if not (0 < a.steps <= a.schedule_steps and a.minutes > 0 and a.checkpoint_every > 0):
        raise ValueError('Invalid step/time bounds')
    if a.device == 'cpu':
        os.environ['NORD_REFERENCE_ATTENTION'] = '1'
    elif os.environ.get('NORD_REFERENCE_ATTENTION') == '1':
        raise RuntimeError('GPU pilot must exercise native FA3, not reference attention')
    import torch
    import torch.nn.functional as F
    from tokenizers import Tokenizer
    from models.baselines.hrm_nocarry_bp_warmup import HierarchicalReasoningModel
    from models.lm_head import LMHead
    from models.adam_atan2 import AdamATan2
    torch.set_num_threads(4)
    if a.device == 'cuda':
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
            raise RuntimeError('This pilot requires a Hopper GPU (H100/H200)')
        if torch.cuda.device_count() != 1:
            raise RuntimeError('Expose exactly one GPU for this pilot')
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    cfg = json.loads(a.config.read_text())
    tok = Tokenizer.from_file(str(ROOT/'data/tokenizer.json'))
    if tok.get_vocab_size() > cfg['vocab_size']:
        raise ValueError('Tokenizer does not fit model vocabulary')
    if a.batch_tokens < cfg['max_seq_len']:
        # Actual samples still checked below. CPU tests use a short config.
        raise ValueError('batch-tokens must be at least max_seq_len')
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out/'latest.pt').exists() and not a.resume:
        raise RuntimeError('Existing run: use --resume or choose a new output directory')
    model = LMHead(HierarchicalReasoningModel(cfg), cfg).to(a.device)
    if a.device == 'cuda':configure_cuda_mixed_precision(model)
    optim = AdamATan2(model.parameters(), lr=2e-4, ema=0.999)
    parameter_count = sum(p.numel() for p in model.parameters())
    fingerprint = {'config':cfg, 'tokenizer_sha256':digest(ROOT/'data/tokenizer.json'),
       'train_sha256':digest(ROOT/'data/train.jsonl'), 'valid_sha256':digest(ROOT/'data/valid.jsonl'),
       'batch_tokens':a.batch_tokens,'schedule_steps':a.schedule_steps,'seed':a.seed,'device':a.device,
       'source_sha256':digest(__file__), 'reference_sha256':digest(ROOT/'reference_attention.py'), 'torch_version':str(torch.__version__), 'stress_context':a.stress_context}
    # Bind recovery to the source tree, not just the runner.
    fingerprint['model_sources'] = {str(p.relative_to(ROOT.parent)):digest(p) for p in sorted((ROOT.parent/'models').rglob('*.py'))}
    step, cursor = 0, 0
    if a.resume:
        ck = torch.load(a.resume, map_location='cpu', weights_only=False)  # Only trusted own checkpoints.
        if ck['fingerprint'] != fingerprint:
            raise RuntimeError('Resume config/data/code fingerprint mismatch')
        model.load_state_dict(ck['model'])
        optim.load_state_dict(ck['optim'])
        step, cursor = ck['step'], ck['cursor']
        torch.set_rng_state(ck['rng'])
        random.setstate(ck['python_rng'])
        if a.device == 'cuda': torch.cuda.set_rng_state_all(ck['cuda_rng'])
    records = {}
    for name in ('train','valid'):
        records[name] = [json.loads(x) for x in (ROOT/f'data/{name}.jsonl').read_text().splitlines()]
    random.Random(a.seed).shuffle(records['train'])
    def encode(r):
        prefix = [tok.token_to_id('[BOS]')] + tok.encode(r['instruction']).ids + [tok.token_to_id('[SEP]')]
        answer = tok.encode(r['response']).ids + [tok.token_to_id('[EOS]')]
        if len(prefix)+len(answer)-1 > cfg['max_seq_len']:
            raise ValueError('Sample exceeds max_seq_len; do not silently truncate evidence')
        return prefix, answer
    encoded = {k:[encode(r) for r in rs] for k,rs in records.items()}
    if a.stress_context:
        def extend(pair):
            p,r=pair
            target=cfg['max_seq_len']-len(r)+1
            middle=p[1:-1]
            p=[p[0]]+(middle*((target+len(middle))//len(middle)))[:target-2]+[p[-1]]
            return p,r
        encoded={k:[extend(pair) for pair in pairs] for k,pairs in encoded.items()}
    def batch_at(name, start, budget):
        inputs, labels, positions, pl, cl, cu = [], [], [], [], [], [0]
        count = 0
        while count < len(encoded[name]):
            p, r = encoded[name][(start+count)%len(encoded[name])]
            n = len(p)+len(r)-1
            if len(inputs)+n > budget: break
            inputs += p+r[:-1]
            labels += [-100]*(len(p)-1)+r
            positions += list(range(n))
            pl.append(len(p)); cl.append(len(r)-1); cu.append(cu[-1]+n)
            count += 1
        if count == 0: raise ValueError('No sample fits batch budget')
        to = lambda x: torch.tensor(x,dtype=torch.int32,device=a.device)
        batch = dict(inputs=to(inputs),position_ids=to(positions),prefix_lens=to(pl+[0]),causal_lens=to(cl),cu_seqlens=to(cu))
        for k,v in dict(total_seqlen=len(inputs),numseqs=count,max_seqlen_prefix=max(pl),max_seqlen_causal=max(cl),max_seqlen_all=max(x+y for x,y in zip(pl,cl))).items():
            batch[k] = torch.tensor(v,dtype=torch.int32,device='cpu')
        return batch, to(labels).long(), start+count
    def autocast():
        return torch.autocast('cuda',dtype=torch.bfloat16) if a.device=='cuda' else contextlib.nullcontext()
    def loss_at(batch, labels, training):
        ctx={'aux_losses':[],'expert_counts':[]}
        with autocast():
            _, logits=model(None,batch,moe_context=ctx,bp_steps=5)
            ce=F.cross_entropy(logits.float(),labels,ignore_index=-100)
            aux=torch.stack(ctx['aux_losses']).mean() if ctx['aux_losses'] else ce.new_zeros(())
            loss=ce+cfg['moe_router_aux_loss_coef']*aux
        return loss,ce,ctx
    def evaluate():
        model.eval()
        with torch.no_grad():
            b,y,_=batch_at('valid',0,a.batch_tokens)
            _,ce,_=loss_at(b,y,False)
        model.train()
        return float(ce)
    def atomic_save(path, obj):
        temp=path.with_suffix('.tmp')
        torch.save(obj,temp)
        with temp.open('rb') as f: os.fsync(f.fileno())
        os.replace(temp,path)
    def save():
        # Preserve previous completed checkpoint until the new one is durable.
        if (a.out/'latest.pt').exists():
            import shutil
            shutil.copy2(a.out/'latest.pt',a.out/'previous.pt')
        atomic_save(a.out/'latest.pt',dict(model=model.state_dict(),optim=optim.state_dict(),step=step,cursor=cursor,
            rng=torch.get_rng_state(),python_rng=random.getstate(),cuda_rng=torch.cuda.get_rng_state_all() if a.device=='cuda' else [],fingerprint=fingerprint))
    stop=[False]
    for sig in (signal.SIGTERM, signal.SIGINT): signal.signal(sig,lambda *args:stop.__setitem__(0,True))
    start=time.monotonic(); times=[]; losses=[]; tokens=0
    initial_valid=evaluate()
    manifest={'parameters':parameter_count,'torch':torch.__version__,'cuda':torch.version.cuda,
       'device':torch.cuda.get_device_name() if a.device=='cuda' else 'CPU reference only',
       'initial_step':step,'fingerprint':fingerprint,'initial_valid_loss':initial_valid}
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'parameters':parameter_count,'start_step':step,'valid_loss':initial_valid}),flush=True)
    if a.device=='cuda': torch.cuda.reset_peak_memory_stats()
    while step < a.steps and not stop[0] and time.monotonic()-start < a.minutes*60:
        b,y,next_cursor=batch_at('train',cursor,a.batch_tokens)
        if a.device=='cuda': torch.cuda.synchronize()
        t=time.monotonic()
        model.train(); optim.zero_grad(set_to_none=True)
        loss,ce,ctx=loss_at(b,y,True)
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss; retaining last good checkpoint')
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
        optim.step()
        step+=1; cursor=next_cursor
        if a.device=='cuda': torch.cuda.synchronize()
        dt=time.monotonic()-t
        times.append(dt); losses.append(float(ce.detach())); tokens+=len(y)
        counts=torch.stack(ctx['expert_counts']).sum(0).tolist()
        metric={'step':step,'loss':losses[-1],'seconds':dt,'input_tokens':len(y),
             'supervised_tokens':int((y!=-100).sum()),'grad_norm':float(norm),'expert_counts':counts}
        with (a.out/'metrics.jsonl').open('a') as f:f.write(json.dumps(metric)+'\n')
        if step % 10==0:print(json.dumps(metric),flush=True)
        if step % a.checkpoint_every==0:save()
    save()
    final_valid=evaluate()
    atomic_save(a.out/'export.pt',{'model':model.state_dict(),'config':cfg,'tokenizer_sha256':fingerprint['tokenizer_sha256'],'step':step})
    summary={'status':'completed' if step==a.steps else 'bounded_stop','step':step,'initial_valid_loss':initial_valid,
       'final_valid_loss':final_valid,'first_train_loss':losses[0] if losses else None,'last_train_loss':losses[-1] if losses else None,
       'input_tokens':tokens,'training_seconds':sum(times),'tokens_per_training_second':tokens/sum(times) if times else None,
       'peak_memory_bytes':torch.cuda.max_memory_allocated() if a.device=='cuda' else None,
       'wall_seconds':time.monotonic()-start,'capability_claim':'none; synthetic technical fixture'}
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
    if step!=a.steps:sys.exit(3)
if __name__=='__main__':main()
