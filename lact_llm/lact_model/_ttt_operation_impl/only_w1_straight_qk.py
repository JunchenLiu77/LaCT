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
    weight_norm: bool = True,
    use_lr1: bool = True,
):
    """
    Only do gradient ascent on w1.
    """
    del lr0, lr2 # make sure lr0 and lr2 are not used.

    if weight_norm:
        # adding detach here sometimes improves stability.
        w1_norm = w1.norm(dim=2, keepdim=True)

    if momentum is not None:
        dw1_momentum = torch.zeros_like(w1)

    q = q.transpose(1, 2)  # [b, dk, l]
    v = v.transpose(1, 2)

    output = torch.zeros_like(v)

    e_index = 0
    seq_len = k.shape[1]
    for i in range(0, seq_len - chunk_size, chunk_size):
        s_index = i
        e_index = s_index + chunk_size

        # [b, l, dk]
        ki = k[:, s_index:e_index, :]  # bf16
        # [b, dv, l]
        vi = v[:, :, s_index:e_index]  # bf16
        # [b, dh, l]
        qi = q[:, :, s_index:e_index]

        if use_lr1:
            # [b, l, d/1] fp32
            lr1i = lr1[:, s_index:e_index, :]  # [b, l, d/1] fp32
        else:
            # keep the original lr1 in the gradient graph but set it to 1.0
            lr1i = lr1[:, s_index:e_index, :] * 0.0 + 1.0

        # get the final output
        # [b, dv, dh] @ [b, dh, l] -> [b, dv, l] -> [b, l, dv]
        output[:, :, s_index:e_index] = torch.bmm(w1, qi) # straight q as hidden.

        # straight k as hidden.
        hidden = ki.transpose(1, 2)

        # [b, d_2, l] @ [b, l, d_1] -> [b, d_2, d_1]
        # in bmm two mat is fp32, but the result is bf16.
        # it's better to cast the mat to bf16 before bmm.
        # [b, dv, l] @ [b, l, dh] -> [b, dv, dh]
        # it's better to cast the mat to bf16 before bmm.
        dw1 = torch.bmm(vi, (hidden.transpose(1, 2) * lr1i).type_as(vi))  # [b, d, d]

        if momentum is not None:
            m_i = momentum[:, s_index:e_index, :]
            m_i = m_i.mean(dim=1, keepdim=True)

            dw1 = dw1 + dw1_momentum * m_i
            dw1_momentum = dw1

        if use_muon:
            dw1 = zeropower_via_newtonschulz5(dw1)
            # legacy code for different global lr for muon. Conclusion: 1.0 is good
            # if muon_w0_lr is not None:
            #     # lr is fp32 (after softplus)
            #     # in future version, we can cast it before input. TODO
            #     dw1 = (dw1 * muon_w1_lr).type_as(w1)

        w1 = w1 + dw1

        if weight_norm:
            # Do channel-wise l2 norm.  conceptually like post-norm.
            w1 = w1 / (w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm

    # for the last chunk, don't update the fast weights, directly apply the fast weights to the query.
    s_index = e_index
    e_index = seq_len

    qi = q[:, :, s_index:e_index]
    # get the final output
    # [b, dv, dh] @ [b, dh, l] -> [b, dv, l] -> [b, l, dv]
    output[:, :, s_index:e_index] = torch.bmm(w1, qi) # straight q as hidden.

    # keep w0 and w2 in the gradient graph.
    return output.transpose(1, 2) + w0.sum() * 0.0 + w2.sum() * 0.0
