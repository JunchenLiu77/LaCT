import torch
import torch.nn.functional as F

from lact_ttt import zeropower_via_newtonschulz5

@torch.compile
def fn(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    ttt_ua_order: list,
    muon_update_steps: int = 0,
):
    """
    Note:
    Forward:
    (silu(x @ w0) * (x @ w2)) @ w1

    w0, w2: [b, d, dh]
    w1:     [b, dh, d]
    q: [b, l, d]
    k: [b, l, d]
    v: [b, l, d]
    lr0, lr1, lr2: [b, l, 1]
    """
    w1_norm = w1.detach().norm(dim=1, keepdim=True)

    output = []
    for start, end, update, apply in ttt_ua_order:
        w0_now, w1_now, w2_now = w0, w1, w2

        if update:
            ki, vi = k[:, start:end, :], v[:, start:end, :]  # bf16
            lr0i = lr0[:, start:end, :]  # [b, l, d/1] fp32
            lr1i = lr1[:, start:end, :]  # [b, l, d/1] fp32
            lr2i = lr2[:, start:end, :]  # [b, l, d/1] fp32

            gate_before_act = ki @ w0_now       # b[b, l, dh] = [b, l, d] @ [b, d, dh]
            hidden_before_mul = ki @ w2_now     # b[b, l, dh] = [b, l, d] @ [b, d, dh]
            hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul
            hidden = ki
            w1_grad = zeropower_via_newtonschulz5(
                (hidden * lr1i).transpose(-1, -2) @ vi, muon_update_steps
            )
            w1_now = w1_now + w1_grad

            # do weight norm here
            w1_now = w1_now / (w1_now.norm(dim=1, keepdim=True) + 1e-5) * w1_norm

            w1 = w1_now

        if apply:
            # Only calculate the output in the last repeat.
            qi = q[:, start:end, :]
            oi = qi @ w1_now
            output.append(oi)

    output = torch.cat(output, dim=1)
    output = output + torch.zeros_like(output) * (w0.sum() + w2.sum())

    return output, w0, w1, w2