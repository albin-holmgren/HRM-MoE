"""Full-fixture and counterfactual audit of saved pilot weights on CPU reference kernels.
This is not a product benchmark. No cloud calls, training, or pretrained weights.
"""
import argparse,hashlib,json,os,re,sys,time
from pathlib import Path
os.environ['NORD_REFERENCE_ATTENTION']='1'
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from models.baselines.hrm_nocarry_bp_warmup import HierarchicalReasoningModel
from models.lm_head import LMHead

def make_counterfactual(row, novel=False):
    r=dict(row);r['id']='changed-'+str(row['id'])
    pattern=r'(opens at |öppnar klockan )(\d+)'
    old=int(re.search(pattern,row['instruction']).group(2));new=12+(old%4) if novel else 8+(old-8+2)%4
    r['instruction']=re.sub(pattern,lambda m:m.group(1)+str(new),row['instruction'],count=1)
    r['response']=re.sub(pattern,lambda m:m.group(1)+str(new),row['response'],count=1)
    return r

def scores(pred,expected):
    fact=lambda s:re.search(r'(?:opens at |öppnar klockan )(\d+)',s)
    p,e=fact(pred),fact(expected)
    return {'exact':pred.strip()==expected.strip(),'fact_correct':bool(p and e and p.group(1)==e.group(1)),
      'citation_correct':re.findall(r'\[S\d+\]',pred)==re.findall(r'\[S\d+\]',expected)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--exports',nargs='+',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--batch',type=int,default=4);ap.add_argument('--samples',type=int,default=16);a=ap.parse_args()
    if a.out.exists():raise ValueError('Choose a fresh report directory')
    a.out.mkdir(parents=True);torch.set_num_threads(4)
    tok=Tokenizer.from_file(str(ROOT/'data/tokenizer.json'));bos,sep,eos=[tok.token_to_id(s) for s in ('[BOS]','[SEP]','[EOS]')]
    train=[json.loads(x) for x in (ROOT/'data/train.jsonl').read_text().splitlines()]
    valid=[json.loads(x) for x in (ROOT/'data/valid.jsonl').read_text().splitlines()]
    def change_source(r):
        new=dict(r);new['id']='source-'+str(r['id']);old=f'[S{r["id"]}]';replacement=f'[S{(r["id"]+64)%128}]'
        for key in ('instruction','response'):new[key]=r[key].replace(old,replacement)
        return new
    groups={'train':train,'held_out':valid,'changed_known_hours':[make_counterfactual(r) for r in train[:a.samples]],
      'changed_novel_hours':[make_counterfactual(r,novel=True) for r in train[:a.samples]],
      'changed_source_ids':[change_source(r) for r in train[:a.samples]]}
    (a.out/'evaluation_cases.json').write_text(json.dumps(groups,ensure_ascii=False,indent=2))
    def prefix(r):return [bos]+tok.encode(r['instruction']).ids+[sep]
    def pack(sequences,prefixes):
        cu=[0];pos=[];ids=[]
        for seq in sequences:ids+=seq;pos+=list(range(len(seq)));cu.append(len(ids))
        causal=[len(s)-p for s,p in zip(sequences,prefixes)]
        tensor=lambda x:torch.tensor(x,dtype=torch.int32)
        b={'inputs':tensor(ids),'position_ids':tensor(pos),'prefix_lens':tensor(prefixes+[0]),'causal_lens':tensor(causal),'cu_seqlens':tensor(cu)}
        for k,v in dict(total_seqlen=len(ids),numseqs=len(sequences),max_seqlen_prefix=max(prefixes),max_seqlen_causal=max(causal),max_seqlen_all=max(map(len,sequences))).items():b[k]=tensor(v)
        return b,cu
    results=[]
    for path in a.exports:
        start=time.monotonic();ck=torch.load(path,map_location='cpu',weights_only=False)
        assert hashlib.sha256((ROOT/'data/tokenizer.json').read_bytes()).hexdigest()==ck['tokenizer_sha256']
        config=dict(ck['config']);config['moe_implementation']='grouped'
        model=LMHead(HierarchicalReasoningModel(config),config);model.load_state_dict(ck['model']);del ck;model.eval()
        checkpoint={'export':str(path),'backend':'CPU float32 reference attention + grouped PyTorch experts; not identical CUDA numerics','groups':{}}
        with torch.inference_mode():
            for group,rows in groups.items():
                total_nll=0;total_targets=0
                for offset in range(0,len(rows),a.batch):
                    chunk=rows[offset:offset+a.batch];ps=[prefix(r) for r in chunk];answers=[tok.encode(r['response']).ids+[eos] for r in chunk]
                    b,cu=pack([p+r[:-1] for p,r in zip(ps,answers)],[len(p) for p in ps]);_,logits=model(None,b)
                    labels=torch.tensor(sum(([-100]*(len(p)-1)+r for p,r in zip(ps,answers)),[]))
                    total_nll+=float(F.cross_entropy(logits.float(),labels,ignore_index=-100,reduction='sum'));total_targets+=sum(map(len,answers))
                predictions=[]
                for offset in range(0,min(a.samples,len(rows)),a.batch):
                    chunk=rows[offset:min(offset+a.batch,a.samples)];ps=[prefix(r) for r in chunk];generated=[[] for _ in chunk];done=[False]*len(chunk)
                    for _ in range(24):
                        b,cu=pack([p+g for p,g in zip(ps,generated)],[len(p) for p in ps]);_,logits=model(None,b)
                        choices=logits[torch.tensor(cu[1:])-1,:tok.get_vocab_size()].argmax(-1).tolist()
                        for i,choice in enumerate(choices):
                            if not done[i]:
                                generated[i].append(choice);done[i]=choice==eos
                        if all(done):break
                    for r,g in zip(chunk,generated):
                        pred=tok.decode(g);predictions.append({'id':r['id'],'instruction':r['instruction'],'expected':r['response'],'generated':pred,**scores(pred,r['response'])})
                counts={k:sum(p[k] for p in predictions) for k in ('exact','fact_correct','citation_correct')}
                result={'teacher_forced_examples':len(rows),'target_tokens':total_targets,'full_split_loss':total_nll/total_targets,'generated_examples':len(predictions),'correct_counts':counts,'predictions':predictions}
                checkpoint['groups'][group]=result
                print(json.dumps({'checkpoint':str(path),'group':group,'loss':result['full_split_loss'],'n':len(predictions),'correct':counts,'elapsed_seconds':time.monotonic()-start}),flush=True)
                (a.out/f'checkpoint-{len(results)}.json').write_text(json.dumps(checkpoint,ensure_ascii=False,indent=2))
        checkpoint['elapsed_seconds']=time.monotonic()-start;results.append(checkpoint);del model
    (a.out/'summary.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
