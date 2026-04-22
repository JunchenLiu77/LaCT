# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.nn.functional as F
import torch

from ..ttt_operation import silu_backprop, zeropower_via_newtonschulz5


@torch.compile()
def block_causal_lact_swiglu(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    chunk_size: int = 2048,  # test-time training chunk size
    use_muon: bool = False,
    momentum: torch.Tensor = None,  # [b, s, 1]
    weight_norm: bool = False,
):
    """
    Only do gradient ascent on w1. parallel form is copied from parallel/only_w1_no_wn.py

    About precision (following original.py convention):
        w0, w1, w2 are mostly likely fp32.
        q, k, v are bf16.
        lr1 is fp32.
        The forward, backward produce bf16 gradients, updated fast weights are fp32.
        The final output are bf16.
    """
    del lr0, lr2 # make sure lr0 and lr2 are not used.
    assert not weight_norm, "parallel form does not support weight norm."

    b, l, dk = q.shape
    dv = v.shape[-1]
    dh = w0.shape[1]
    n_blocks = (l + chunk_size - 1) // chunk_size
    l_pad = n_blocks * chunk_size
    pad = l_pad - l

    # 1. Concatenate weights for single projection launch
    # [b, 2*dh, dk]
    w02 = torch.cat([w0, w2], dim=1)

    # 2. Prepare inputs (Flatten B and N)
    q_pad = F.pad(q, (0, 0, 0, pad)) if pad > 0 else q
    k_pad = F.pad(k, (0, 0, 0, pad)) if pad > 0 else k
    v_pad = F.pad(v, (0, 0, 0, pad)) if pad > 0 else v

    q_flat = q_pad.view(b * n_blocks, chunk_size, dk)
    k_flat = k_pad.view(b * n_blocks, chunk_size, dk)
    v_flat = v_pad.view(b * n_blocks, chunk_size, dv)

    # 3. Single BMM for Gate & Hidden (instead of two)
    # [b*n, dk, c] -> [b*n, 2*dh, c]
    w02_exp = w02.repeat_interleave(n_blocks, dim=0)
    z_all = torch.bmm(w02_exp, k_flat.transpose(1, 2))

    # Slice gate and hidden
    gate_k, hidden_k = z_all.chunk(2, dim=1)
    # Fused SiLU and Mul
    gated_k = F.silu(gate_k) * hidden_k # [b*n, dh, c]

    # 4. Compute dw1 using BMM (Replaces einsum)
    # Cast to input dtype before bmm for gradient computation (like original.py line 102)
    # [b*n, dv, c] @ [b*n, c, dh] -> [b*n, dv, dh]
    lr1_pad = F.pad(lr1, (0, 0, 0, pad)) if pad > 0 else lr1
    # Cast (gated_k * lr1) to v's dtype before bmm, similar to original.py
    dw1 = torch.bmm(
        v_flat.transpose(1, 2),
        (gated_k.transpose(1, 2) * lr1_pad.view(b * n_blocks, chunk_size, -1)).type_as(v_flat)
    )
    dw1 = dw1.view(b, n_blocks, dv, dh)

    # 5. Momentum Recurrence
    if momentum is not None:
        m_pad = F.pad(momentum, (0, 0, 0, pad)) if pad > 0 else momentum
        m_blk = m_pad.view(b, n_blocks, chunk_size).mean(dim=2, keepdim=True)
        # Parallel scan via cumprod/cumsum
        m_prod = m_blk.cumprod(dim=1).unsqueeze(-1)
        dw1 = m_prod * torch.cumsum(dw1 / (m_prod + 1e-8), dim=1)

    if use_muon:
        # Use single flattened call for Muon
        dw1 = zeropower_via_newtonschulz5(dw1.reshape(-1, dv, dh)).reshape_as(dw1)

    # 6. Prefix-sum for w1
    prefix_exclusive = torch.cumsum(dw1, dim=1) - dw1
    w1_before = prefix_exclusive + w1.unsqueeze(1)

    # 7. Final Output Projection
    z_q = torch.bmm(w02_exp, q_flat.transpose(1, 2))
    gate_q, hidden_q = z_q.chunk(2, dim=1)
    gated_q = F.silu(gate_q) * hidden_q

    # [b*n, dv, dh] @ [b*n, dh, c] -> [b*n, dv, c]
    # Cast w1_before to input dtype before bmm to keep output in bf16
    out = torch.bmm(w1_before.view(b * n_blocks, dv, dh).type_as(gated_q), gated_q)

    out = out.transpose(1, 2).reshape(b, l_pad, dv)
    return out[:, :l, :]
