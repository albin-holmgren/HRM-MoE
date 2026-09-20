"""Bounded, auditable real-text rehearsal. No teacher calls or cloud provisioning.
Viewer rows are revision-checked; cached pages and hashes make the sample replayable.
This is not a full-corpus download or a benchmark-contamination certification.
"""
import argparse, collections, hashlib, json, random, re, unicodedata
from pathlib import Path
from urllib.parse import urlsplit
from datetime import datetime, timezone
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
SESSION=requests.Session()
SESSION.mount("https://",HTTPAdapter(max_retries=Retry(total=2,backoff_factor=.5,status_forcelist=[429,500,502,503,504])))
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

SOURCES = [
    {'name':'edu','repo':'HuggingFaceFW/fineweb-edu','config':'sample-10BT','pages':30,'license':'odc-by'},
    {'name':'math','repo':'HuggingFaceTB/finemath','config':'finemath-4plus','pages':5,'license':'odc-by'},
    {'name':'sv','repo':'HuggingFaceFW/fineweb-2','config':'swe_Latn','pages':10,'license':'odc-by'},
    {'name':'explanations','repo':'HuggingFaceTB/smollm-corpus','config':'cosmopedia-v2','pages':10,'license':'odc-by'},
]
SPECIAL=['[PAD]','[UNK]','[BOS]','[SEP]','[EOS]']
def sha(x):return hashlib.sha256(x if isinstance(x,bytes) else x.encode()).hexdigest()
def normalize(text):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',text)).strip()
def shingles(text):
    words=normalize(text).lower().split()
    return {sha(' '.join(words[i:i+5]))[:16] for i in range(len(words)-4)}
def signature(words):
    return tuple(min(int(sha(str(i)+w)[:16],16) for w in words) for i in range(32))
def split_for(group):
    bucket=int(sha('nord-real-v1:'+group)[:8],16)%100
    return 'valid' if bucket<5 else ('test' if bucket<10 else 'train')
def reject_reason(row):
    text=row.get('text','');words=text.split()
    if not 80<=len(words)<=3000:return 'length'
    if '\ufffd' in text or sum(c.isalpha() for c in text)/max(1,len(text))<.5:return 'encoding_or_low_text'
    if re.search(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}',text):return 'email_present'
    if re.search(r'(?i)\b(gsm8k|mmlu[- ]?pro|math[- ]?500|humaneval|hellaswag|arc[- ]challenge|aime 202[4-6])\b',text):return 'benchmark_name'
    if row.get('language_score',1)<.8:return 'language_score'
    if any(token in text for token in SPECIAL):return 'control_token_text'
    if len(set(words))/len(words)<.15:return 'repetition'
    return None

def fetch(out):
    cache=out/'raw_pages';cache.mkdir(parents=True,exist_ok=True)
    all_rows=[];sources=[];download_bytes=0
    for source in SOURCES:
        meta=SESSION.get('https://huggingface.co/api/datasets/'+source['repo'],timeout=40);meta.raise_for_status();meta=meta.json()
        if meta.get('cardData',{}).get('license')!=source['license']:raise ValueError('Dataset license metadata changed')
        rev=meta['sha'];entry={**source,'revision':rev,'pages_downloaded':[]}
        for i in range(source['pages']):
            # Dispersed deterministic windows, not a representative random corpus sample.
            offset=100*(i*997+17)
            path=cache/f'{source["name"]}-{offset}.json'
            if path.exists():wrapped=json.loads(path.read_text())
            else:
                response=SESSION.get('https://datasets-server.huggingface.co/rows',params={'dataset':source['repo'],'config':source['config'],'split':'train','offset':offset,'length':100},timeout=45)
                response.raise_for_status()
                if len(response.content)>20_000_000:raise ValueError('Page too large')
                wrapped={'revision':response.headers.get('x-revision'),'payload':response.json()}
                path.write_text(json.dumps(wrapped,ensure_ascii=False))
            if wrapped['revision']!=rev:raise ValueError('Viewer revision differs from pinned dataset revision')
            payload=wrapped['payload']
            if any(r.get('truncated_cells') for r in payload['rows']):raise ValueError('Viewer returned truncated document')
            download_bytes+=path.stat().st_size
            if download_bytes>100_000_000:raise ValueError('Rehearsal download cap exceeded')
            entry['pages_downloaded'].append({'offset':offset,'rows':len(payload['rows']),'sha256':sha(path.read_bytes())})
            for wrapped_row in payload['rows']:
                row=wrapped_row['row'];text=row['text'];url=row.get('url') or ''
                group=urlsplit(url).hostname if url else None
                if not group:group=sha(json.dumps(row.get('prompt',text),sort_keys=True,ensure_ascii=False))
                all_rows.append({'doc_id':sha(text),'text':text,'source':source['name'],'dataset':source['repo'],'revision':rev,'license':source['license'],'url':url,'group':group,'language':'sv' if source['name']=='sv' else 'en','language_score':row.get('language_score',1)})
            if (i+1)%10==0:print(json.dumps({'source':source['name'],'pages':i+1,'download_bytes':download_bytes}),flush=True)
        sources.append(entry)
    return all_rows,sources,download_bytes

def curate(rows):
    counts=collections.Counter();seen=set();kept=[];bands=collections.defaultdict(list);wordsets=[]
    for row in rows:
        reason=reject_reason(row)
        if reason:counts[reason]+=1;continue
        norm=normalize(row['text']).lower();key=sha(norm)
        if key in seen:counts['exact_duplicate']+=1;continue
        seen.add(key);words=shingles(norm);sig=signature(words)
        candidates=set()
        for b in range(8):candidates.update(bands[(b,sig[b*4:b*4+4])])
        if any(len(words&wordsets[i])/len(words|wordsets[i])>=.8 for i in candidates):counts['near_duplicate']+=1;continue
        idx=len(kept)
        for b in range(8):bands[(b,sig[b*4:b*4+4])].append(idx)
        wordsets.append(words);row={**row,'normalized_sha256':key,'split':split_for(row['group'])};kept.append(row)
    return kept,counts

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'manifest.json').exists():raise ValueError('Completed dataset exists; choose a fresh output')
    rows,sources,nbytes=fetch(a.out);kept,drops=curate(rows)
    print(json.dumps({'fetched':len(rows),'kept':len(kept),'rejections':drops}),flush=True)
    tokenizer=Tokenizer(models.BPE(unk_token='[UNK]'));tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False);tokenizer.decoder=decoders.ByteLevel()
    tokenizer.train_from_iterator((r['text'] for r in kept if r['split']=='train'),trainers.BpeTrainer(vocab_size=8192,special_tokens=SPECIAL,initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),show_progress=False))
    tokenizer.save(str(a.out/'tokenizer.json'))
    stats=collections.defaultdict(lambda:collections.Counter());provenance=[]
    handles={s:(a.out/f'{s}.jsonl').open('w') for s in ['train','valid','test']}
    control_ids={tokenizer.token_to_id(t) for t in SPECIAL}
    for row in kept:
        ids=tokenizer.encode(row['text']).ids
        if any(t in control_ids for t in ids):raise ValueError('Unexpected special token in document')
        roundtrip=tokenizer.decode(ids)==row['text']
        if not roundtrip:raise ValueError('Tokenizer round-trip failed')
        stat=stats[row['split']];stat['documents']+=1;stat['text_tokens']+=len(ids);stat[row['source']+'_tokens']+=len(ids)
        for i in range(0,len(ids),254):
            chunk=ids[i:i+254]
            record={'id':row['doc_id']+':'+str(i),'doc_id':row['doc_id'],'kind':'pretrain','token_ids':chunk,'document_end':i+254>=len(ids)}
            handles[row['split']].write(json.dumps(record)+'\n');stat['sequences']+=1;stat['supervised_tokens']+=len(chunk)+int(record['document_end'])
        provenance.append({k:v for k,v in row.items() if k!='text'})
    for h in handles.values():h.close()
    (a.out/'provenance.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in provenance))
    group_sets={s:{r['group'] for r in kept if r['split']==s} for s in handles}
    assert not any(group_sets[x]&group_sets[y] for x,y in [('train','valid'),('train','test'),('valid','test')])
    manifest={'created_at':datetime.now(timezone.utc).isoformat(),'purpose':'small real-text pretraining rehearsal, not product benchmark','fetched_rows':len(rows),'kept_documents':len(kept),'rejections':dict(drops),'sources':sources,'download_bytes':nbytes,'statistics':dict(stats),'tokenizer_vocab':tokenizer.get_vocab_size(),'tokenizer_training_split':'train only','split_policy':'hostname or synthetic seed group, stable 90/5/5 hash; actual counts vary','near_dedup':'5-word shingles, 32 MinHash, 8x4 LSH candidates, Jaccard >=0.8; approximate recall','decontamination':'benchmark-name screening only; full benchmark overlap audit NOT performed','PII':'email screening only; not complete PII removal','code_sha256':sha(Path(__file__).read_bytes()),'files':{p.name:sha(p.read_bytes()) for p in a.out.glob('*') if p.is_file()}}
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({'statistics':dict(stats),'tokenizer_vocab':tokenizer.get_vocab_size()}),flush=True)
if __name__=='__main__':main()
