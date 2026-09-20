"""Compare two trusted exports on a held-out pretokenized corpus and fixed prompts."""
import argparse,hashlib,json,sys,time
from itertools import islice
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
    p.add_argument('--test-records',type=int,default=0,help='0 scores the whole held-out split; a positive value scores only that many records in file order, which keeps a paid run bounded')
    p.add_argument('--disjoint-limit',type=int,default=0,help='0 inspects every training and validation record for leakage; a positive value inspects only that many per split, which is a bounded spot check rather than a full audit')
    return p.parse_args()
def pack(seqs,prefixes):
    cu=[0];inputs=[];positions=[]
    for seq in seqs:inputs+=seq;positions+=list(range(len(seq)));cu.append(len(inputs))
    causal=[len(seq)-prefix for seq,prefix in zip(seqs,prefixes)];tensor=lambda x:torch.tensor(x,dtype=torch.int32,device='cuda')
    batch=dict(inputs=tensor(inputs),position_ids=tensor(positions),prefix_lens=tensor(prefixes+[0]),causal_lens=tensor(causal),cu_seqlens=tensor(cu))
    for key,value in dict(total_seqlen=len(inputs),numseqs=len(seqs),max_seqlen_prefix=max(prefixes),max_seqlen_causal=max(causal),max_seqlen_all=max(map(len,seqs))).items():batch[key]=torch.tensor(value,dtype=torch.int32)
    return batch,cu
def stream(path):
    """Yield parsed rows one at a time.

    The previous version used path.read_text().splitlines() for the held-out split and
    for the train/valid leak check. That worked for the ~1 MB rehearsal corpus but not
    for the built one, where train.jsonl is 18 GB and test-fresh.jsonl is about 1 GB; the
    process would exhaust memory before evaluating anything on a paid GPU. Rows are read
    and released one at a time and only document ids are retained.
    """
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)

def assert_disjoint(test_ids,path,limit=0):
    """Fail if any held-out document also appears in a training split.

    Membership is tested against the held-out id set while streaming, so the millions of
    training document ids never need to be resident at once. A positive limit inspects
    only that many records per split; the caller then reports a bounded spot check instead
    of presenting it as a full audit of the training data.
    """
    rows=0
    for row in stream(path):
        if limit and rows>=limit:break
        rows+=1
        if row.get('doc_id') in test_ids:
            raise ValueError(f'Test document leaked into {path.name}')
    return {'records_inspected':rows,'complete':limit<=0}
def main():
    a=parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0]!=9:raise RuntimeError('Evaluation requires one Hopper GPU')
    test_path=a.data_dir/a.test_file
    # Count and collect ids in one streaming pass, then re-read for evaluation so the whole
    # test split is never resident. The file is read twice; that is cheaper than holding it.
    test_ids=set();test_sequences=0
    for row in stream(test_path):
        test_sequences+=1
        if row.get('doc_id'):test_ids.add(row['doc_id'])
    if not test_sequences:raise ValueError('Held-out test is empty')
    # Scoring the whole split means 151M target tokens, which is hours of forward passes per
    # model on one GPU. A bounded cap keeps a paid run inside its budget, and the reported
    # scope says so rather than implying the full split was scored.
    scored_records=test_sequences if a.test_records<=0 else min(a.test_records,test_sequences)
    leakage={name:assert_disjoint(test_ids,a.data_dir/f'{name}.jsonl',a.disjoint_limit) for name in ('train','valid')}
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
            batch_rows=[]
            for row in islice(stream(test_path),scored_records):
                batch_rows.append(row)
                if len(batch_rows)<8:continue
                chunk=batch_rows;batch_rows=[]
                ys=[r['token_ids']+([eos] if r.get('document_end',True) else []) for r in chunk]
                batch,_=pack([[bos]+y[:-1] for y in ys],[1]*len(ys));_,logits=model(None,batch,bp_steps=5)
                labels=torch.tensor(sum(ys,[]),device='cuda');nll+=float(F.cross_entropy(logits.float(),labels,reduction='sum'));targets+=len(labels)
            if batch_rows:
                ys=[r['token_ids']+([eos] if r.get('document_end',True) else []) for r in batch_rows]
                batch,_=pack([[bos]+y[:-1] for y in ys],[1]*len(ys));_,logits=model(None,batch,bp_steps=5)
                labels=torch.tensor(sum(ys,[]),device='cuda');nll+=float(F.cross_entropy(logits.float(),labels,reduction='sum'));targets+=len(labels)
            seqs=[[bos]+tok.encode(prompt).ids for prompt in prompts];generated=[[] for _ in prompts];done=[False]*len(prompts)
            for _ in range(48):
                batch,cu=pack(seqs,[1]*len(seqs));_,logits=model(None,batch,bp_steps=5)
                choices=logits[torch.tensor(cu[1:],device='cuda')-1].argmax(-1).tolist()
                for i,choice in enumerate(choices):
                    if not done[i]:seqs[i].append(choice);generated[i].append(choice);done[i]=choice==eos
                if all(done):break
        scope='a bounded prefix of the held-out split' if scored_records<test_sequences else 'the whole held-out split'
        audit='full' if all(item['complete'] for item in leakage.values()) else 'bounded spot check, not a full training-data audit'
        result={'name':name,'step':ck['step'],'parameters':sum(p.numel() for p in model.parameters()),'test_file':a.test_file,'test_sha256':sha(test_path),'test_sequences':test_sequences,'test_documents':len(test_ids),'test_records_scored':scored_records,'test_target_tokens':targets,'test_loss':nll/targets,'score_scope':scope,'leakage_check':{'scope':audit,**leakage},'seconds':time.monotonic()-started,'completions':[{'prompt':p,'completion':tok.decode(g),'tokens':len(g)} for p,g in zip(prompts,generated)],'claim':f'fixed raw text completions scored on {scope}; leakage inspection was a {audit}; not instruction-tuned or a capability benchmark'}
        results.append(result);print(json.dumps(result,ensure_ascii=False),flush=True);del model,ck;torch.cuda.empty_cache()
    if results[1]['test_loss']>=results[0]['test_loss']:raise RuntimeError('Fresh-test loss did not improve')
    a.out.write_text(json.dumps(results,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
