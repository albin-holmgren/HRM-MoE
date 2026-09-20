"""Reuse prior verified DeepSeek examples and prepare a bounded high-reasoning queue.
No API requests are sent by this utility. New outputs are candidates until verified.
"""
import argparse,hashlib,json,shutil
from pathlib import Path
from tokenizers import Tokenizer

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--prior-curated',type=Path,required=True);p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('Choose a fresh output')
    a.out.mkdir(parents=True);tok=Tokenizer.from_file(str(a.tokenizer));shutil.copyfile(a.tokenizer,a.out/'tokenizer.json')
    counts={};source_hashes={};tasks=set()
    for original,dest in [('train','train'),('val','valid')]:
        path=a.prior_curated/f'{original}.jsonl';source_hashes[str(path)]=sha(path);rows=[]
        for r in map(json.loads,path.read_text().splitlines()):
            if r.get('generator_model')!='deepseek/deepseek-v4.1-flash' or not r.get('verified'):raise ValueError('Unexpected teacher or unverified prior row')
            if r['task_id'] in tasks:raise ValueError('Overlapping teacher train/validation task')
            tasks.add(r['task_id'])
            prompt,answer=r['prompt'],r['solution']
            n=len(tok.encode(prompt).ids)+len(tok.encode(answer).ids)+3
            if n>2048:raise ValueError('Teacher example exceeds pilot context; do not truncate')
            rows.append({'id':r['task_id'],'instruction':prompt,'response':answer,'teacher':r['generator_model'],'verification':'previous recorded '+str(r['verify_method'])+'; not rerun here','source':'prior Nord pilot24','tokens_with_control':n})
        (a.out/f'{dest}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));counts[dest]={'examples':len(rows),'tokens_with_control':sum(r['tokens_with_control'] for r in rows)}
    # Deterministic arithmetic task generator, independently checkable final answers.
    queue=[];oracle=[]
    for i in range(50):
        boxes=3+i%13;items=7+(i*7)%29;sold=2+(i*3)%17
        question=f'A warehouse has {boxes} boxes with {items} parts in each. It ships {sold} parts. How many parts remain?'
        request={'model':'deepseek-flash','thinking':{'type':'enabled'},'reasoning_effort':'high','max_tokens':4096,'response_format':{'type':'json_object'},'messages':[{'role':'system','content':'Solve the problem accurately. Return JSON with final_answer (an integer) and explanation (at most 3 short steps). Keep the teaching explanation concise. Do not include unrelated material.'},{'role':'user','content':question}]}
        queue.append({'id':f'arithmetic-teacher-{i}','request':request});oracle.append({'id':f'arithmetic-teacher-{i}','expected_answer':boxes*items-sold,'check':'exact integer answer; explanation requires separate review'})
    (a.out/'proposed_requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in queue))
    (a.out/'answer_checks.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in oracle))
    plan={'status':'prepared, NOT sent','provider':'DeepSeek direct; gateway rates may differ','current_alias':'deepseek-flash = DeepSeek-V4.1-Flash','verified_date':'2026-09-20','price_source':'https://api-docs.deepseek.com/quick_start/pricing/','thinking_source':'https://api-docs.deepseek.com/guides/thinking_mode/','terms_source':'https://cdn.deepseek.com/policies/en-US/deepseek-open-platform-terms-of-service.html','peak_input_per_million':.3,'peak_output_per_million':1.2,'offpeak_input_per_million':.15,'offpeak_output_per_million':.6,'proposal_calls':50,'output_cap_including_reasoning_per_call':4096,'assumed_input_tokens_per_call':1000,'estimated_batch_at_max_output_peak_usd':50*(1000*.3+4096*1.2)/1e6,'estimated_batch_at_max_output_offpeak_usd':50*(1000*.15+4096*.6)/1e6,'proposed_total_api_budget_usd':1,'budget_state':'proposal only; no provider-enforced cap configured','execution_requirements':['Funded DeepSeek API access; Verda balance does not pay API bills','Reserve worst-case cost before every request; include retries and reasoning tokens','No automatic retries of ambiguous/timeout calls; preserve results and usage ledger','Reject truncated, malformed or incorrect outputs; do not train unchecked explanations','Use independently generated different task families for evaluation; these 50 template variants are not a reasoning benchmark'],'prior_examples':counts,'prior_source_hashes':source_hashes,'reuse_policy':'Prompt and tested code answer only; long private teacher trace is not copied into tiny-student targets','new_calls':0}
    (a.out/'teacher_plan.json').write_text(json.dumps(plan,indent=2));print(json.dumps(plan))
if __name__=='__main__':main()
