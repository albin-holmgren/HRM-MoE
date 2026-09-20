"""Native FA3 PrefixLM + Triton expert forward/backward equivalence gates."""
import json,torch
from models.flash_attention_prefixlm_v2 import flash_attn_varlen_prefixlm as native
from nord_pilot.reference_attention import flash_attn_varlen_prefixlm as reference
from models.layers import SparseMoEGroupedExperts

def main():
    assert torch.cuda.is_available() and torch.cuda.get_device_capability()[0]==9
    torch.manual_seed(3)
    kwargs=dict(is_causal=False,prefix_lens=torch.tensor([5,3],device='cuda',dtype=torch.int32),
      causal_lens=torch.tensor([4,4],device='cuda',dtype=torch.int32),cu_seqlens=torch.tensor([0,9,16],device='cuda',dtype=torch.int32))
    kwargs.update({k:torch.tensor(v,dtype=torch.int32) for k,v in dict(total_seqlen=16,numseqs=2,max_seqlen_prefix=5,max_seqlen_causal=4,max_seqlen_all=9).items()})
    qkv=[torch.randn(16,4,128,device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
    other=[x.detach().clone().requires_grad_() for x in qkv]
    y=native(*qkv,**kwargs);z=reference(*other,**kwargs)
    torch.testing.assert_close(y,z,rtol=0.03,atol=0.03)
    grad=torch.randn_like(y);y.backward(grad);z.backward(grad)
    for a,b in zip(qkv,other):torch.testing.assert_close(a.grad,b.grad,rtol=0.05,atol=0.05)
    # Same packed parameter layout for the two expert kernels.
    base=SparseMoEGroupedExperts(128,128,8,backend='bmm').cuda()
    fast=SparseMoEGroupedExperts(128,128,8,backend='triton').cuda();fast.load_state_dict(base.state_dict())
    x=torch.randn(64,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    xx=x.detach().clone().requires_grad_()
    idx=torch.stack((torch.arange(64,device='cuda')%8,(torch.arange(64,device='cuda')+1)%8),dim=1)
    w=torch.full((64,2),0.5,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    ww=w.detach().clone().requires_grad_()
    a=base(x,idx,w,torch.zeros_like(x));b=fast(xx,idx,ww,torch.zeros_like(xx))
    torch.testing.assert_close(a,b,rtol=0.03,atol=0.03)
    g=torch.randn_like(a);a.backward(g);b.backward(g)
    for p,q in [(x,xx),(w,ww),*zip(base.parameters(),fast.parameters())]:
        torch.testing.assert_close(p.grad,q.grad,rtol=0.05,atol=0.05)
    print(json.dumps({'native_fa3_forward_backward':'pass','triton_experts_forward_backward':'pass','device':torch.cuda.get_device_name()}))
if __name__=='__main__':main()
