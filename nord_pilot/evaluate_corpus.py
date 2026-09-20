"""Compare two trusted exports on a held-out pretokenized corpus and fixed prompts."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from models.baselines.hrm_nocarry_bp_warmup import HierarchicalReasoningModel
from models.lm_head import LMHead
from nord_pilot.run import configure_cuda_mixed_precision

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def parse_args():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--test-file',default='test-fresh.jsonl');p.add_argument('--initial-export',type=Path,required=True)
    p.add_argument('--trained-export',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    return p.parse_args()
def pack(seqs,prefixes):
    cu=[0];inputs=[];positions=[]
    for seq in seqs:inputs+=seq;positions+=list(range(len(seq)));cu.append(len(inputs))
    causal=[len(seq)-prefix for seq,prefix in zip(seqs,prefixes)];tensor=lambda x:torch.tensor(x,dtype=torch.int32,device='cuda')
    batch=dict(inputs=tensor(inputs),position_ids=tensor(positions),prefix_lens=tensor(prefixes+[0]),causal_lens=tensor(causal),cu_seqlens=tensor(cu))
    for key,value in dict(total_seqlen=len(inputs),numseqs=len(seqs),max_seqlen_prefix=max(prefixes),max_seqlen_causal=max(causal),max_seqlen_all=max(map(len,seqs))).items():batch[key]=torch.tensor(value,dtype=torch.int32)
    return batch,cu
def ids(path):return {json.loads(line)['doc_id'] for line in path.read_text().splitlines()}
def main():
    a=parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0]!=9:raise RuntimeError('Evaluation requires one Hopper GPU')
    test_path=a.data_dir/a.test_file;rows=[json.loads(line) for line in test_path.read_text().splitlines()]
    if not rows:raise ValueError('Held-out test is empty')
    test_ids={row['doc_id'] for row in rows}
    if test_ids&ids(a.data_dir/'train.jsonl') or test_ids&ids(a.data_dir/'valid.jsonl'):raise ValueError('Test document leaked into train/valid')
    tok=Tokenizer.from_file(str(a.data_dir/'tokenizer.json'));token_hash=sha(a.data_dir/'tokenizer.json')
    bos,eos=tok.token_to_id('[BOS]'),tok.token_to_id('[EOS]')
    prompts=['Stockholm is the capital of','Sverige är ett land i','To calculate the average of three numbers,','Water freezes when','2 + 3 =','Question: What is the capital of Sweden? Answer:']
    results=[]
    for name,path in [('initial',a.initial_export),('trained_best',a.trained_export)]:
        started=time.monotonic();ck=torch.load(path,map_location='cpu',weights_only=False)
        if ck['tokenizer_sha256']!=token_hash:raise RuntimeError('Export tokenizer mismatch')
        model=LMHead(HierarchicalReasoningModel(ck['config']),ck['config']).cuda();model.load_state_dict(ck['model']);configure_cuda_mixed_precision(model);model.eval()
        nll=0.;targets=0
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for offset in range(0,len(rows),8):
                chunk=rows[offset:offset+8];ys=[r['token_ids']+([eos] if r.get('document_end',True) else []) for r in chunk]
                batch,_=pack([[bos]+y[:-1] for y in ys],[1]*len(ys));_,logits=model(None,batch,bp_steps=5)
                labels=torch.tensor(sum(ys,[]),device='cuda');nll+=float(F.cross_entropy(logits.float(),labels,reduction='sum'));targets+=len(labels)
            seqs=[[bos]+tok.encode(prompt).ids for prompt in prompts];generated=[[] for _ in prompts];done=[False]*len(prompts)
            for _ in range(48):
                batch,cu=pack(seqs,[1]*len(seqs));_,logits=model(None,batch,bp_steps=5)
                choices=logits[torch.tensor(cu[1:],device='cuda')-1].argmax(-1).tolist()
                for i,choice in enumerate(choices):
                    if not done[i]:seqs[i].append(choice);generated[i].append(choice);done[i]=choice==eos
                if all(done):break
        result={'name':name,'step':ck['step'],'parameters':sum(p.numel() for p in model.parameters()),'test_file':a.test_file,'test_sha256':sha(test_path),'test_sequences':len(rows),'test_documents':len(test_ids),'test_target_tokens':targets,'test_loss':nll/targets,'seconds':time.monotonic()-started,'completions':[{'prompt':p,'completion':tok.decode(g),'tokens':len(g)} for p,g in zip(prompts,generated)],'claim':'fixed raw text completions; not instruction-tuned or a capability benchmark'}
        results.append(result);print(json.dumps(result,ensure_ascii=False),flush=True);del model,ck;torch.cuda.empty_cache()
    if results[1]['test_loss']>=results[0]['test_loss']:raise RuntimeError('Fresh-test loss did not improve')
    a.out.write_text(json.dumps(results,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
