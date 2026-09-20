"""Reload our exported model in a fresh process; uncached greedy generation."""
import argparse, json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))

def main():
    p=argparse.ArgumentParser();p.add_argument('--export',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],required=True);p.add_argument('--tokens',type=int,default=16);a=p.parse_args()
    if a.device=='cpu':os.environ['NORD_REFERENCE_ATTENTION']='1'
    import torch
    import contextlib
    from tokenizers import Tokenizer
    from models.baselines.hrm_nocarry_bp_warmup import HierarchicalReasoningModel
    from models.lm_head import LMHead
    from nord_pilot.run import digest
    torch.set_num_threads(4)
    ck=torch.load(a.export,map_location='cpu',weights_only=False)
    assert digest(ROOT/'data/tokenizer.json')==ck['tokenizer_sha256']
    cfg=ck['config']; model=LMHead(HierarchicalReasoningModel(cfg),cfg).to(a.device)
    model.load_state_dict(ck['model']);model.eval()
    tok=Tokenizer.from_file(str(ROOT/'data/tokenizer.json'))
    row=json.loads((ROOT/'data/valid.jsonl').read_text().splitlines()[0])
    ids=[tok.token_to_id('[BOS]')]+tok.encode(row['instruction']).ids+[tok.token_to_id('[SEP]')]
    prefix=len(ids);generated=[]
    with torch.inference_mode():
        for _ in range(a.tokens):
            if len(ids)>=cfg['max_seq_len']:break
            n=len(ids)
            to=lambda x:torch.tensor(x,dtype=torch.int32,device=a.device)
            batch={'inputs':to(ids),'position_ids':to(list(range(n))),'prefix_lens':to([prefix,0]),'causal_lens':to([n-prefix]),'cu_seqlens':to([0,n])}
            for k,v in dict(total_seqlen=n,numseqs=1,max_seqlen_prefix=prefix,max_seqlen_causal=n-prefix,max_seqlen_all=n).items():batch[k]=torch.tensor(v,dtype=torch.int32)
            with torch.autocast('cuda',dtype=torch.bfloat16) if a.device=='cuda' else contextlib.nullcontext():
                _,logits=model(None,batch)
            if not torch.isfinite(logits).all():raise RuntimeError('Nonfinite inference logits')
            # Unused padded vocabulary entries cannot be decoded by the fixture tokenizer.
            token=int(logits[-1,:tok.get_vocab_size()].argmax());generated.append(token);ids.append(token)
            if token==tok.token_to_id('[EOS]'):break
    print(json.dumps({'status':'reload_and_generation_pass','step':ck['step'],'token_ids':generated,'text':tok.decode(generated),'capability_claim':'none'},ensure_ascii=False))
if __name__=='__main__':main()
