import json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
os.environ['NORD_REFERENCE_ATTENTION']='1'
import torch
from nord_pilot.reference_attention import flash_attn_varlen_prefixlm
from nord_pilot.compare import compare
ROOT=Path(__file__).resolve().parent.parent

class PilotTests(unittest.TestCase):
    def test_prefix_mask_no_future_or_other_sample_leak(self):
        # Two packed sequences: two prefix positions followed by two causal positions.
        torch.manual_seed(1)
        q,k,v=[torch.randn(8,1,8,requires_grad=True) for _ in range(3)]
        args=dict(is_causal=False,prefix_lens=torch.tensor([2,2,0]),causal_lens=torch.tensor([2,2]),cu_seqlens=torch.tensor([0,4,8]),total_seqlen=8,numseqs=2)
        a=flash_attn_varlen_prefixlm(q,k,v,**args)
        vv=v.detach().clone();vv[3:]+=100
        b=flash_attn_varlen_prefixlm(q,k,vv,**args)
        torch.testing.assert_close(a[:3],b[:3])
        vv=v.detach().clone();vv[1]+=10
        c=flash_attn_varlen_prefixlm(q,k,vv,**args)
        self.assertFalse(torch.allclose(a[0],c[0]))
        a.sum().backward();self.assertTrue(torch.isfinite(q.grad).all())
    def test_fresh_process_resume_and_export(self):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td)
            common=[sys.executable,'nord_pilot/run.py','--device','cpu','--config','nord_pilot/configs/cpu.json','--batch-tokens','256','--minutes','2']
            def run(extra,ok=True):
                r=subprocess.run(common+extra,cwd=ROOT,capture_output=True,text=True)
                if ok:self.assertEqual(r.returncode,0,r.stdout+r.stderr)
                else:self.assertNotEqual(r.returncode,0)
                return r
            run(['--out',str(td/'full'),'--steps','12'])
            run(['--out',str(td/'split'),'--steps','6'])
            run(['--out',str(td/'split'),'--steps','12','--resume',str(td/'split/latest.pt')])
            result=compare(td/'full/latest.pt',td/'split/latest.pt','cpu');self.assertEqual(result['max_abs_difference'],0)
            summary=json.loads((td/'full/summary.json').read_text());self.assertLess(summary['last_train_loss'],summary['first_train_loss'])
            r=subprocess.run([sys.executable,'nord_pilot/infer.py','--device','cpu','--export',str(td/'split/export.pt')],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr);self.assertIn('reload_and_generation_pass',r.stdout)
            r=run(['--out',str(td/'bad'),'--steps','13','--seed','99','--resume',str(td/'split/latest.pt')],False)
            self.assertIn('fingerprint mismatch',r.stderr)
            run(['--out',str(td/'split'),'--steps','12'],False)
            run(['--out',str(td/'stress'),'--steps','2','--stress-context'])
            stress=json.loads((td/'stress/summary.json').read_text())
            self.assertEqual(stress['input_tokens'],512)
            run(['--out',str(td/'stop'),'--steps','10','--eval-every','2','--early-stop-patience','1','--min-delta','1000000'])
            stopped=json.loads((td/'stop/summary.json').read_text())
            self.assertEqual(stopped['status'],'early_stopped');self.assertEqual(stopped['step'],2)
            self.assertEqual(stopped['best_step'],0);self.assertEqual(stopped['validation_examples'],32)
            best=torch.load(td/'stop/best-export.pt',map_location='cpu',weights_only=False)
            latest=torch.load(td/'stop/export.pt',map_location='cpu',weights_only=False)
            self.assertEqual(best['step'],0);self.assertEqual(latest['step'],2)
            self.assertTrue(any(not torch.equal(best['model'][k],latest['model'][k]) for k in best['model']))
            run(['--out',str(td/'stop'),'--steps','10','--eval-every','2','--early-stop-patience','1','--min-delta','1000000','--resume',str(td/'stop/latest.pt')])
            recovered=json.loads((td/'stop/summary.json').read_text())
            self.assertEqual(recovered['status'],'early_stopped');self.assertEqual(recovered['step'],2)
            self.assertEqual(recovered['best_step'],0);self.assertEqual(recovered['input_tokens'],0)
            validation=[json.loads(l) for l in (td/'stop/validation.jsonl').read_text().splitlines()]
            self.assertEqual([v['step'] for v in validation],[0,2])
            self.assertTrue(all(v['examples']==32 for v in validation))
            metrics=[json.loads(l) for l in (td/'full/metrics.jsonl').read_text().splitlines()]
            self.assertTrue(all(sum(m['expert_counts'])>0 for m in metrics))
            print(json.dumps({'local_resume':result,'training_loss':[summary['first_train_loss'],summary['last_train_loss']],'export_reload':'pass','changed_config_rejected':True}))
if __name__=='__main__':unittest.main()
