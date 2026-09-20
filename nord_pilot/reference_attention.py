"""Explicit slow mathematical reference; never selected silently on a GPU."""
import torch
import torch.nn.functional as F


def flash_attn_varlen_prefixlm(q, k, v, is_causal, prefix_lens, causal_lens,
                              cu_seqlens, total_seqlen, numseqs, **unused):
    # Match native FA3's shifted cu_seqlens convention, including terminal zero.
    assert len(prefix_lens) >= int(numseqs)+1 and int(prefix_lens[int(numseqs)]) == 0, 'prefix_lens needs a terminal zero sentinel'
    outputs = []
    for j in range(int(numseqs)):
        start, end = int(cu_seqlens[j]), int(cu_seqlens[j + 1])
        p = int(prefix_lens[j])
        n = end - start
        pos = torch.arange(n, device=q.device)
        mask = pos[:, None] >= pos[None, :]
        if not is_causal:
            mask = mask | ((pos[:, None] < p) & (pos[None, :] < p))
        out = F.scaled_dot_product_attention(q[start:end].transpose(0, 1),
                k[start:end].transpose(0, 1), v[start:end].transpose(0, 1), attn_mask=mask)
        outputs.append(out.transpose(0, 1))
    if int(total_seqlen) < len(q):
        outputs.append(torch.zeros_like(q[int(total_seqlen):]))
    return torch.cat(outputs)
