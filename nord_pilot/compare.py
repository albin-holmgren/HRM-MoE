import argparse,json
import torch

def compare(left,right,device):
    a=torch.load(left,map_location='cpu',weights_only=False);b=torch.load(right,map_location='cpu',weights_only=False)
    assert a['fingerprint']==b['fingerprint']
    assert a['step']==b['step'] and a['cursor']==b['cursor']
    tol=0.0 if device=='cpu' else 1e-5
    count=[0];maximum=[0.0]
    def walk(x,y):
        if isinstance(x,torch.Tensor):
            assert x.shape==y.shape and x.dtype==y.dtype
            if x.is_floating_point():
                maximum[0]=max(maximum[0],float((x-y).abs().max()) if x.numel() else 0)
                torch.testing.assert_close(x,y,rtol=tol,atol=tol)
            else:assert torch.equal(x,y)
            count[0]+=1
        elif isinstance(x,dict):
            assert x.keys()==y.keys()
            for k in x:walk(x[k],y[k])
        elif isinstance(x,(tuple,list)):
            assert len(x)==len(y)
            for xx,yy in zip(x,y):walk(xx,yy)
        else:assert x==y
    for k in ('model','optim','rng','python_rng','cuda_rng','selection','best_model'):walk(a[k],b[k])
    return {'status':'resume_equivalence_pass','tensors_compared':count[0],'max_abs_difference':maximum[0],'step':a['step'],'tolerance':tol}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('left');p.add_argument('right');p.add_argument('--device',required=True);a=p.parse_args();print(json.dumps(compare(a.left,a.right,a.device)))
