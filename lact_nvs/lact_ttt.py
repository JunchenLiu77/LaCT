import collections
import math

import torch
from torch import nn

import torch.nn.functional as F
from einops import rearrange

TTTOperator = collections.namedtuple("TTTOperator", ["start", "end", "update", "apply"])
TTTOperator_fused = collections.namedtuple("TTTOperator_fused", ["num_update_tokens", "num_apply_tokens"])

VISUALIZE = False

@torch.compile
def inv_softplus(x):
    y = x + math.log(-math.expm1(-x))
    return y

@torch.compile
def silu_backprop(dy: torch.Tensor, x: torch.Tensor):
    """
    Args:
        dy: [b, d, l], gradient of the outer loss wrt the y
        x: [b, d, l], input of the silu activation
    outs:
        dx: [b, d, l], gradient of the outer loss wrt the x
        dx = dy * sigma * (1 + x * (1 - sigma))
    """
    sigma = torch.sigmoid(x)
    dx = dy * sigma * (1 + x * (1 - sigma))
    return dx

@torch.compile
def silu_backprop_(x: torch.Tensor):
    """
    Similar to silu_backprop, but don't take the upstream gradient
    Args:
        x: [b, d, l], input of the silu activation
    outs:
        dx: [b, d, l], gradient of the outer loss wrt the x
        dx = sigma * (1 + x * (1 - sigma))
    """
    sigma = torch.sigmoid(x)
    dx = sigma * (1 + x * (1 - sigma))
    return dx

@torch.compile
def fast_weight_swish_glu_fwd(k: torch.Tensor, w0: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, inplace: bool = True):
    # inplace operations will cause memory-hungry when combined with create_graph=True
    return (F.silu(k @ w0, inplace=inplace) * (k @ w2)) @ w1


@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    modified from https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py#L49
    Major change: G is [b, d, d] rather than [d, d]
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    Args:
        G: [b, d, d]
        steps: int
    Returns:
        X: [b, d, d]
    """
    assert len(G.shape) == 3
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.transpose(1, 2)
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    return X


@torch.compile
def fast_weight_swish_glu_weight_norm_mini_batch_apply(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    block_idx: int,
    ttt_ua_order: list,
    muon_update_steps: int = 0,
    ttt_loss_type: str = "dot_product",
    grad_calc_method: str = "mannual",
    no_query: bool = False,
):
    """
    Note:
    Forward:
    (silu(x @ w0) * (x @ w2)) @ w1

    w0, w2: [b, d, dh]
    w1:     [b, dh, d]
    q: [b, l, d], Optional
    k: [b, l, d]
    v: [b, l, d]
    lr0, lr1, lr2: [b, l, 1]
    """
    if no_query:
        assert q is None, "q should be None if no_query is True"
    else:
        assert q is not None, "q should not be None if no_query is False"

    w0_norm = w0.detach().norm(dim=1, keepdim=True)
    w1_norm = w1.detach().norm(dim=1, keepdim=True)
    w2_norm = w2.detach().norm(dim=1, keepdim=True)

    output = []
    for start, end, update, apply in ttt_ua_order:
        w0_now, w1_now, w2_now = w0, w1, w2

        if update:
            ki, vi = k[:, start:end, :], v[:, start:end, :]  # bf16
            lr0i = lr0[:, start:end, :]  # [b, l, d/1] fp32
            lr1i = lr1[:, start:end, :]  # [b, l, d/1] fp32
            lr2i = lr2[:, start:end, :]  # [b, l, d/1] fp32

            if grad_calc_method == "mannual":
                if ttt_loss_type == "design1":
                    assert not no_query, "design1 need query tensor"
                    # update: 0.5 * MLP(q) + 0.5 * MLP(k) -> v, dot product loss
                    k_gate_before_act = ki @ w0_now       # b[b, l, dh] = [b, l, d] @ [b, d, dh]
                    k_hidden_before_mul = ki @ w2_now     # b[b, l, dh] = [b, l, d] @ [b, d, dh]
                    k_hidden = F.silu(k_gate_before_act, inplace=False) * k_hidden_before_mul
                    k_vpi = k_hidden @ w1_now

                    qi = q[:, start:end, :]
                    q_gate_before_act = qi @ w0_now       # b[b, l, dh] = [b, l, d] @ [b, d, dh]
                    q_hidden_before_mul = qi @ w2_now     # b[b, l, dh] = [b, l, d] @ [b, d, dh]
                    q_hidden = F.silu(q_gate_before_act, inplace=False) * q_hidden_before_mul
                    q_vpi = q_hidden @ w1_now

                    # vpi = 0.5 * q_vpi + 0.5 * k_vpi
                    # dot product loss: -vpi * vi
                    dvpi = -vi
                    
                    k_dhidden = dvpi @ w1_now.transpose(-1, -2)  # [b, l, dh] = [b, l, d] @ [b, d, dh]
                    k_dhidden_before_mul = k_dhidden * F.silu(k_gate_before_act, inplace=False)
                    k_dgate = k_dhidden * k_hidden_before_mul
                    k_dgate_before_act = silu_backprop(k_dgate, k_gate_before_act)

                    q_dhidden = dvpi @ w1_now.transpose(-1, -2)  # [b, l, dh] = [b, l, d] @ [b, d, dh]
                    q_dhidden_before_mul = q_dhidden * F.silu(q_gate_before_act, inplace=False)
                    q_dgate = q_dhidden * q_hidden_before_mul
                    q_dgate_before_act = silu_backprop(q_dgate, q_gate_before_act)

                    w1_grad = ((k_hidden * lr1i).transpose(-1, -2) @ dvpi) + ((q_hidden * lr1i).transpose(-1, -2) @ dvpi) * 0.5
                    w0_grad = ((ki * lr0i).transpose(-1, -2) @ k_dgate_before_act) + ((qi * lr0i).transpose(-1, -2) @ q_dgate_before_act) * 0.5
                    w2_grad = ((ki * lr2i).transpose(-1, -2) @ k_dhidden_before_mul) + ((qi * lr2i).transpose(-1, -2) @ q_dhidden_before_mul) * 0.5
                elif ttt_loss_type == "design2":
                    assert not no_query, "design2 need query tensor"
                    # update: MLP(0.5 * q + 0.5 * k) -> v, dot product loss
                    qi = q[:, start:end, :]
                    mlp_input = 0.5 * qi + 0.5 * ki
                    gate_before_act = mlp_input @ w0_now       # b[b, l, dh] = [b, l, d] @ [b, d, dh]
                    hidden_before_mul = mlp_input @ w2_now     # b[b, l, dh] = [b, l, d] @ [b, d, dh]
                    hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul
                    vpi = hidden @ w1_now

                    dvpi = -vi
                    
                    dhidden = dvpi @ w1_now.transpose(-1, -2)  # [b, l, dh] = [b, l, d] @ [b, d, dh]
                    dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
                    dgate = dhidden * hidden_before_mul
                    dgate_before_act = silu_backprop(dgate, gate_before_act)

                    w1_grad = ((hidden * lr1i).transpose(-1, -2) @ dvpi)
                    w0_grad = ((mlp_input * lr0i).transpose(-1, -2) @ dgate_before_act)
                    w2_grad = ((mlp_input * lr2i).transpose(-1, -2) @ dhidden_before_mul)
                else:
                    # manually compute the gradient
                    gate_before_act = ki @ w0_now       # b[b, l, dh] = [b, l, d] @ [b, d, dh]
                    hidden_before_mul = ki @ w2_now     # b[b, l, dh] = [b, l, d] @ [b, d, dh]
                    hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul
                    vpi = hidden @ w1_now

                    if ttt_loss_type == "dot_product":
                        dvpi = -vi
                    else:
                        raise NotImplementedError(f"Unknown ttt_loss_type: {ttt_loss_type}")
                    
                    dhidden = dvpi @ w1_now.transpose(-1, -2)  # [b, l, dh] = [b, l, d] @ [b, d, dh]
                    dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
                    dgate = dhidden * hidden_before_mul
                    dgate_before_act = silu_backprop(dgate, gate_before_act)

                    w1_grad = ((hidden * lr1i).transpose(-1, -2) @ dvpi)
                    w0_grad = ((ki * lr0i).transpose(-1, -2) @ dgate_before_act)
                    w2_grad = ((ki * lr2i).transpose(-1, -2) @ dhidden_before_mul)
            else:
                raise ValueError(f"Unknown grad_calc_method: {grad_calc_method}")

            # orthogonalized gradients
            w1_grad = zeropower_via_newtonschulz5(w1_grad, muon_update_steps)
            w0_grad = zeropower_via_newtonschulz5(w0_grad, muon_update_steps)
            w2_grad = zeropower_via_newtonschulz5(w2_grad, muon_update_steps)

            w1_now = w1_now - w1_grad
            w0_now = w0_now - w0_grad
            w2_now = w2_now - w2_grad

            # do weight norm here
            w0_now = w0_now / (w0_now.norm(dim=1, keepdim=True) + 1e-5) * w0_norm
            w1_now = w1_now / (w1_now.norm(dim=1, keepdim=True) + 1e-5) * w1_norm
            w2_now = w2_now / (w2_now.norm(dim=1, keepdim=True) + 1e-5) * w2_norm

            w0, w1, w2 = w0_now, w1_now, w2_now

        if apply:
            # Only calculate the output in the last repeat.
            if ttt_loss_type in ["design1", "design2"]:
                # apply: o = MLP(0.5 * q + 0.5 * k)
                qi = q[:, start:end, :]
                ki = k[:, start:end, :]
                oi = fast_weight_swish_glu_fwd(0.5 * qi + 0.5 * ki, w0_now, w1_now, w2_now, inplace=True)
            elif ttt_loss_type == "dot_product":
                if no_query:
                    # reuse k as q when apply
                    ki = k[:, start:end, :]
                    oi = fast_weight_swish_glu_fwd(ki, w0_now, w1_now, w2_now, inplace=True)
                else:
                    qi = q[:, start:end, :]
                    oi = fast_weight_swish_glu_fwd(qi, w0_now, w1_now, w2_now, inplace=True)
            else:
                raise NotImplementedError(f"Unknown ttt_loss_type: {ttt_loss_type}")
            output.append(oi)

            if VISUALIZE:
                ki = k[:, start:end, :]
                vi = v[:, start:end, :]
                vpi = fast_weight_swish_glu_fwd(ki, w0_now, w1_now, w2_now, inplace=True)

                import os
                os.makedirs(f"output/vis/{ttt_loss_type}/{block_idx}", exist_ok=True)
                torch.save(ki, f"output/vis/{ttt_loss_type}/{block_idx}/k.pt")
                torch.save(vi, f"output/vis/{ttt_loss_type}/{block_idx}/v.pt")
                torch.save(oi, f"output/vis/{ttt_loss_type}/{block_idx}/o.pt")
                torch.save(vpi, f"output/vis/{ttt_loss_type}/{block_idx}/vp.pt")

                if not no_query:
                    qi = q[:, start:end, :]
                    torch.save(qi, f"output/vis/{ttt_loss_type}/{block_idx}/q.pt")
                print(f"Saved visualization for block {block_idx} to output/vis/{ttt_loss_type}/{block_idx}")

    output = torch.cat(output, dim=1)
    return output, w0, w1, w2


@torch.compile
def fast_weight_swish_glu_weight_norm_mini_batch_apply_fused(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    block_idx: int,
    ttt_op_order: list,
    muon_update_steps: int = 0,
    ttt_loss_type: str = "dot_product",
    grad_calc_method: str = "mannual",
    no_query: bool = False,
):
    """
    Note:
    Forward:
    (silu(x @ w0) * (x @ w2)) @ w1

    w0, w2: [b, d, dh]
    w1:     [b, dh, d]
    q: [b, l, d], Optional
    k: [b, l, d]
    v: [b, l, d]
    lr0, lr1, lr2: [b, l, 1]
    """
    
    w0_norm = w0.detach().norm(dim=1, keepdim=True)
    w1_norm = w1.detach().norm(dim=1, keepdim=True)
    w2_norm = w2.detach().norm(dim=1, keepdim=True)

    output = []

    num_update_tokens = ttt_op_order[0].num_update_tokens
    num_apply_tokens = ttt_op_order[0].num_apply_tokens

    # order: update then apply
    w0_now, w1_now, w2_now = w0, w1, w2
    ki, vi = k[:, :num_update_tokens, :], v[:, :num_update_tokens, :] # bf16
    if lr0 is not None:
        lr0i = lr0[:, :num_update_tokens, :] # [b, l, d/1] fp32
    if lr1 is not None:
        lr1i = lr1[:, :num_update_tokens, :] # [b, l, d/1] fp32
    if lr2 is not None:
        lr2i = lr2[:, :num_update_tokens, :] # [b, l, d/1] fp32

    def weight_norm(w, w_norm):
        return w / (w.norm(dim=1, keepdim=True) + 1e-5) * w_norm

    if grad_calc_method == "mannual":
        # manually compute the gradient
        if "straight_qk" in ttt_loss_type:
            # ki as hidden, skip w0 and w2, only use w1
            # w1 shape: [b, d, d] for straight_qk (not [b, dh, d])
            hidden = ki
            dvpi = -vi

            w1_grad = ((hidden * lr1i).transpose(-1, -2) @ dvpi)
            # w0 and w2 are not used in straight_qk
            assert "only_w1" in ttt_loss_type, "w0 and w2 are not used in straight_qk"
            w0_grad = torch.zeros_like(w0_now)
            w2_grad = torch.zeros_like(w2_now)
        else:
            gate_before_act = ki @ w0_now       # b[b, l, dh] = [b, l, d] @ [b, d, dh]
            hidden_before_mul = ki @ w2_now     # b[b, l, dh] = [b, l, d] @ [b, d, dh]
            hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul
            vpi = hidden @ w1_now

            if ttt_loss_type == "mse":
                dvpi = vpi - vi
            else:
                # by default, use dot-product loss
                dvpi = -vi
            
            dhidden = dvpi @ w1_now.transpose(-1, -2)  # [b, l, dh] = [b, l, d] @ [b, d, dh]
            dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
            dgate = dhidden * hidden_before_mul
            dgate_before_act = silu_backprop(dgate, gate_before_act)

            w1_grad = ((hidden * lr1i).transpose(-1, -2) @ dvpi)
            w0_grad = ((ki * lr0i).transpose(-1, -2) @ dgate_before_act)
            w2_grad = ((ki * lr2i).transpose(-1, -2) @ dhidden_before_mul)

        if "only_w1" in ttt_loss_type:
            w0_grad = w0_grad * 0.0
            w2_grad = w2_grad * 0.0

        if "ga" in ttt_loss_type:
            w1_grad = -w1_grad
            w0_grad = -w0_grad
            w2_grad = -w2_grad
    elif grad_calc_method == "unroll1":
        # unroll the gradient calculation formula, assume using dot-product loss
        w1_grad = ((F.silu(ki @ w0_now, inplace=False) * (ki @ w2_now)) * lr1i).transpose(-1, -2) @ -vi
        w0_grad = (ki * lr0i).transpose(-1, -2) @ silu_backprop((-vi @ w1_now.transpose(-1, -2)) * (ki @ w2_now), ki @ w0_now)
        w2_grad = (ki * lr2i).transpose(-1, -2) @ ((-vi @ w1_now.transpose(-1, -2)) * F.silu(ki @ w0_now, inplace=False))
    elif grad_calc_method == "simplify1":
        # remove activation function in w1_grad
        w1_grad = (((ki @ w0_now) * (ki @ w2_now)) * lr1i).transpose(-1, -2) @ -vi
        w0_grad = (ki * lr0i).transpose(-1, -2) @ silu_backprop((-vi @ w1_now.transpose(-1, -2)) * (ki @ w2_now), ki @ w0_now)
        w2_grad = (ki * lr2i).transpose(-1, -2) @ ((-vi @ w1_now.transpose(-1, -2)) * F.silu(ki @ w0_now, inplace=False))
    elif grad_calc_method == "simplify2":
        # based on simplify1, remove activation function in w2_grad
        w1_grad = (((ki @ w0_now) * (ki @ w2_now)) * lr1i).transpose(-1, -2) @ -vi
        w0_grad = (ki * lr0i).transpose(-1, -2) @ silu_backprop((-vi @ w1_now.transpose(-1, -2)) * (ki @ w2_now), ki @ w0_now)
        w2_grad = (ki * lr2i).transpose(-1, -2) @ ((-vi @ w1_now.transpose(-1, -2)) * (ki @ w0_now))
    elif grad_calc_method == "simplify3":
        # based on simplify2, replace silu_backprop with dy * x
        w1_grad = (((ki @ w0_now) * (ki @ w2_now)) * lr1i).transpose(-1, -2) @ -vi
        w0_grad = (ki * lr0i).transpose(-1, -2) @ ((-vi @ w1_now.transpose(-1, -2)) * (ki @ w2_now) * (ki @ w0_now))
        w2_grad = (ki * lr2i).transpose(-1, -2) @ ((-vi @ w1_now.transpose(-1, -2)) * (ki @ w0_now))
    elif grad_calc_method == "simplify4":
        # based on simplify2, replace silu_backprop with dy * x * (1 + x * (1 - x))
        w1_grad = (((ki @ w0_now) * (ki @ w2_now)) * lr1i).transpose(-1, -2) @ -vi
        w0_grad = (ki * lr0i).transpose(-1, -2) @ ((-vi @ w1_now.transpose(-1, -2)) * (ki @ w2_now) * (ki @ w0_now) * (1 + (ki @ w0_now) * (1 - (ki @ w0_now))))
        w2_grad = (ki * lr2i).transpose(-1, -2) @ ((-vi @ w1_now.transpose(-1, -2)) * (ki @ w0_now))
    elif grad_calc_method == "simplify5":
        # based on simplify2, replace silu_backprop with silu
        w1_grad = (((ki @ w0_now) * (ki @ w2_now)) * lr1i).transpose(-1, -2) @ -vi
        w0_grad = (ki * lr0i).transpose(-1, -2) @ ((-vi @ w1_now.transpose(-1, -2)) * (ki @ w2_now) * F.silu(ki @ w0_now, inplace=False))
        w2_grad = (ki * lr2i).transpose(-1, -2) @ ((-vi @ w1_now.transpose(-1, -2)) * (ki @ w0_now))
    elif grad_calc_method in ["unroll2", "simplify11", "simplify12"]:
        # unroll and use Ruilong's notation, also fuse update and apply to get the final output
        # simplify11: the same as baseline, but w1 is not optimized
        # simplify12: the same as baseline, but all the weights are not optimized
        k0 = F.silu(ki @ w0_now, inplace=False)
        k0p = silu_backprop_(ki @ w0_now)
        k2 = ki @ w2_now
        v1 = vi @ w1_now.transpose(-1, -2)
        s1 = lr0i * v1 * k2 * k0p
        s2 = lr2i * v1 * k0

        w1_now = weight_norm(w1_now + zeropower_via_newtonschulz5(((k0 * k2) * lr1i).transpose(-1, -2) @ vi, muon_update_steps), w1_norm)
        w0_now = weight_norm(w0_now + zeropower_via_newtonschulz5(ki.transpose(-1, -2) @ s1, muon_update_steps), w0_norm)
        w2_now = weight_norm(w2_now + zeropower_via_newtonschulz5(ki.transpose(-1, -2) @ s2, muon_update_steps), w2_norm)

        # apply
        apply_input = q[:, :num_apply_tokens, :] if not no_query else k[:, :num_apply_tokens, :]
        oi = F.silu(apply_input @ w0_now, inplace=True) * (apply_input @ w2_now) @ w1_now
    elif grad_calc_method == "simplify6":
        # based on unroll2, remove the lr1
        k0 = F.silu(ki @ w0_now, inplace=False)
        k0p = silu_backprop_(ki @ w0_now)
        k2 = ki @ w2_now
        v1 = vi @ w1_now.transpose(-1, -2)
        s1 = lr0i * v1 * k2 * k0p
        s2 = lr2i * v1 * k0

        w1_now = weight_norm(w1_now + zeropower_via_newtonschulz5(((k0 * k2)).transpose(-1, -2) @ vi, muon_update_steps), w1_norm)
        w0_now = weight_norm(w0_now + zeropower_via_newtonschulz5(ki.transpose(-1, -2) @ s1, muon_update_steps), w0_norm)
        w2_now = weight_norm(w2_now + zeropower_via_newtonschulz5(ki.transpose(-1, -2) @ s2, muon_update_steps), w2_norm)

        # apply
        apply_input = q[:, :num_apply_tokens, :] if not no_query else k[:, :num_apply_tokens, :]
        oi = F.silu(apply_input @ w0_now, inplace=True) * (apply_input @ w2_now) @ w1_now
    elif grad_calc_method == "simplify7":
        # based on unroll2, remove all the lrs
        k0 = F.silu(ki @ w0_now, inplace=False)
        k0p = silu_backprop_(ki @ w0_now)
        k2 = ki @ w2_now
        v1 = vi @ w1_now.transpose(-1, -2)
        s1 = v1 * k2 * k0p
        s2 = v1 * k0

        w1_now = weight_norm(w1_now + zeropower_via_newtonschulz5(((k0 * k2)).transpose(-1, -2) @ vi, muon_update_steps), w1_norm)
        w0_now = weight_norm(w0_now + zeropower_via_newtonschulz5(ki.transpose(-1, -2) @ s1, muon_update_steps), w0_norm)
        w2_now = weight_norm(w2_now + zeropower_via_newtonschulz5(ki.transpose(-1, -2) @ s2, muon_update_steps), w2_norm)

        # apply
        apply_input = q[:, :num_apply_tokens, :] if not no_query else k[:, :num_apply_tokens, :]
        oi = F.silu(apply_input @ w0_now, inplace=True) * (apply_input @ w2_now) @ w1_now
    elif grad_calc_method == "simplify9":
        # based on unroll2, remove all the weight norm
        k0 = F.silu(ki @ w0_now, inplace=False)
        k0p = silu_backprop_(ki @ w0_now)
        k2 = ki @ w2_now
        v1 = vi @ w1_now.transpose(-1, -2)
        s1 = lr0i * v1 * k2 * k0p
        s2 = lr2i * v1 * k0

        w1_now = w1_now + zeropower_via_newtonschulz5(((k0 * k2) * lr1i).transpose(-1, -2) @ vi, muon_update_steps)
        w0_now = w0_now + zeropower_via_newtonschulz5(ki.transpose(-1, -2) @ s1, muon_update_steps)
        w2_now = w2_now + zeropower_via_newtonschulz5(ki.transpose(-1, -2) @ s2, muon_update_steps)

        # apply
        apply_input = q[:, :num_apply_tokens, :] if not no_query else k[:, :num_apply_tokens, :]
        oi = F.silu(apply_input @ w0_now, inplace=True) * (apply_input @ w2_now) @ w1_now
    elif grad_calc_method == "simplify10":
        # based on unroll2, remove the newtonschulz5
        k0 = F.silu(ki @ w0_now, inplace=False)
        k0p = silu_backprop_(ki @ w0_now)
        k2 = ki @ w2_now
        v1 = vi @ w1_now.transpose(-1, -2)
        s1 = lr0i * v1 * k2 * k0p
        s2 = lr2i * v1 * k0

        w1_now = weight_norm(w1_now + ((k0 * k2) * lr1i).transpose(-1, -2) @ vi, w1_norm)
        w0_now = weight_norm(w0_now + ki.transpose(-1, -2) @ s1, w0_norm)
        w2_now = weight_norm(w2_now + ki.transpose(-1, -2) @ s2, w2_norm)

        # apply
        apply_input = q[:, :num_apply_tokens, :] if not no_query else k[:, :num_apply_tokens, :]
        oi = F.silu(apply_input @ w0_now, inplace=True) * (apply_input @ w2_now) @ w1_now
    else:
        raise ValueError(f"Unknown grad_calc_method: {grad_calc_method}")

    if grad_calc_method not in ["unroll2", "simplify6", "simplify7", "simplify9", "simplify10", "simplify11", "simplify12"]:
        # orthogonalized gradients
        w1_grad = zeropower_via_newtonschulz5(w1_grad, muon_update_steps)
        if "only_w1" not in ttt_loss_type:
            w0_grad = zeropower_via_newtonschulz5(w0_grad, muon_update_steps)
            w2_grad = zeropower_via_newtonschulz5(w2_grad, muon_update_steps)

        w1_now = w1_now - w1_grad
        w0_now = w0_now - w0_grad
        w2_now = w2_now - w2_grad

        # do weight norm here
        w1_now = weight_norm(w1_now, w1_norm)
        if "only_w1" not in ttt_loss_type:
            w0_now = weight_norm(w0_now, w0_norm)
            w2_now = weight_norm(w2_now, w2_norm)

        w0, w1, w2 = w0_now, w1_now, w2_now

        # apply
        if no_query:
            # reuse k as q when apply
            mlp_input = k[:, :num_apply_tokens, :]
        else:
            mlp_input = q[:, :num_apply_tokens, :]
            
        if "straight_qk" in ttt_loss_type:
            oi = mlp_input @ w1_now
        else:
            oi = fast_weight_swish_glu_fwd(mlp_input, w0_now, w1_now, w2_now, inplace=True)
    output.append(oi)

    if VISUALIZE:
        ki, vi = k[:, :num_apply_tokens, :], v[:, :num_apply_tokens, :]
        vpi = fast_weight_swish_glu_fwd(ki, w0_now, w1_now, w2_now, inplace=True)

        import os
        os.makedirs(f"output/vis/{ttt_loss_type}/{block_idx}", exist_ok=True)
        torch.save(ki, f"output/vis/{ttt_loss_type}/{block_idx}/k.pt")
        torch.save(vi, f"output/vis/{ttt_loss_type}/{block_idx}/v.pt")
        torch.save(oi, f"output/vis/{ttt_loss_type}/{block_idx}/o.pt")
        torch.save(vpi, f"output/vis/{ttt_loss_type}/{block_idx}/vp.pt")

        if not no_query:
            qi = q[:, :num_apply_tokens, :]
            torch.save(qi, f"output/vis/{ttt_loss_type}/{block_idx}/q.pt")
        print(f"Saved visualization for block {block_idx} to output/vis/{ttt_loss_type}/{block_idx}")

    output = torch.cat(output, dim=1)
    return output, w0, w1, w2

class FastWeightGluMLPMultihead(nn.Module):
    """
    On init of fast_weight:

    Let's start with the magnitude of the value.
    value_proj is initialized with uniform distribution with range [-1.0/sqrt(d), 1.0/sqrt(d)]
        x is layernormed. So during init, value is unit norm total (not per head, per head is 1.0/sqrt(num_head))
        After silu, value is around norm of 2.7 per head.  (why? seems wired)

    Then for the fast weight, assume initial lr = 0.
    Then with l2_norm of k, input is unit normed.
    if w0 is initialized with kaiming, relu(w0 @ k) is unit normed.
    Then w1 is initialized with kaiming, so w1 @ relu(w0 @ k) is of norm sqrt(2) per head
    Since I compute total norm, it is sqrt(2) * sqrt(num_head), which is around 2.7 for dim=512, num_head=4.
    """

    def __init__(
        self,
        dim: int,
        head_dim: int,
        block_idx: int,
        inter_multi: int = 1,
        bias: bool = False,
        base_lr=0.01,
        muon_update_steps=0,
        ttt_loss_type="dot_product",
        grad_calc_method: str = "mannual",
        use_fused: bool = False,
        no_query: bool = False,
    ):
        super().__init__()
        self.dim = dim
        assert dim % head_dim == 0
        self.num_heads = dim // head_dim
        self.muon_update_steps = muon_update_steps
        self.ttt_loss_type = ttt_loss_type
        self.grad_calc_method = grad_calc_method
        print(f"TTT loss type: {ttt_loss_type}, grad calculation method: {grad_calc_method}")

        self.block_idx = block_idx
        self.use_fused = use_fused
        self.no_query = no_query
        print(f"Use fused: {use_fused}, No query: {no_query}")

        d_in = d_out = head_dim
        d_h = int(head_dim * inter_multi)

        gain = math.sqrt(2)  # for relu activations
        if grad_calc_method in ["simplify12"]:
            # w0 is not optimized in simplify12
            self.register_buffer(
                "w0", torch.randn(self.num_heads, d_in, d_h) * gain / math.sqrt(d_in)
            )
        else:
            self.w0 = nn.Parameter(
                torch.randn(self.num_heads, d_in, d_h) * gain / math.sqrt(d_in)
            )  # [d_h * num_heads,  d_in]
        if grad_calc_method in ["simplify11", "simplify12"]:
            # w1 is not optimized in simplify11 and simplify12
            self.register_buffer(
                "w1", torch.randn(self.num_heads, d_h, d_out) * gain / math.sqrt(d_h)
            )
        else:
            self.w1 = nn.Parameter(
                torch.randn(self.num_heads, d_h, d_out) * gain / math.sqrt(d_h)
            )  # [d_in * num_heads,  d_h]
        if grad_calc_method in ["simplify12"]:
            # w2 is not optimized in simplify12
            self.register_buffer(
                "w2", torch.randn(self.num_heads, d_in, d_h) * gain / math.sqrt(d_in)
            )
        else:
            self.w2 = nn.Parameter(
                torch.randn(self.num_heads, d_in, d_h) * gain / math.sqrt(d_in)
            )  # [d_h * num_heads,  d_in]

        self.to_qkv = nn.Linear(dim, 2 * dim if no_query else 3 * dim, bias=bias)
        self.c_proj = nn.Linear(dim, dim, bias=bias)

        if self.grad_calc_method not in ['simplify7']:
            self.lr_dim = self.num_heads
            self.lr_fc = nn.Linear(dim, self.lr_dim * 3)
            self.base_lr_inv = inv_softplus(base_lr)
        else:
            self.lr_dim = None
            self.lr_fc = None
            self.base_lr_inv = None

        self.o_norm = torch.nn.RMSNorm(head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, x: torch.Tensor, info={}, *args):
        """
        x: (b, l, d)
        """
        qkv = F.silu(self.to_qkv(x), inplace=True)  # Silu - Linear
        if self.no_query:
            q = None
            k, v = rearrange(
                qkv, "b l (kv h d) -> kv (b h) l d",
                kv=2, h=self.num_heads
            )
            k = k / (k.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype)
        else:
            q, k, v = rearrange(
                qkv, "b l (qkv h d) -> qkv (b h) l d",
                qkv=3, h=self.num_heads
            )
            q = q / (q.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype)
            k = k / (k.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype)

        if self.lr_fc is not None:
            with torch.autocast(device_type="cuda", enabled=False):
                lr = self.lr_fc(x.float())  # [b, l, lr_dim]
            lr = torch.nn.functional.softplus(lr.float() + self.base_lr_inv)
            lr0, lr1, lr2 = rearrange(
                lr, "b l (lrs h d) -> lrs (b h) l d",
                lrs=3, h=self.num_heads
            )
        else: 
            lr0 = lr1 = lr2 = None

        if "w0" in info:
            assert "w1" in info and "w2" in info
            w0 = info["w0"]
            w1 = info["w1"]
            w2 = info["w2"]
        else:
            w0 = self.w0.repeat(x.shape[0], 1, 1)
            w1 = self.w1.repeat(x.shape[0], 1, 1)
            w2 = self.w2.repeat(x.shape[0], 1, 1)

        if self.use_fused:
            fn = fast_weight_swish_glu_weight_norm_mini_batch_apply_fused
        else:
            fn = fast_weight_swish_glu_weight_norm_mini_batch_apply
        output, w0, w1, w2 = fn(
            w0, w1, w2, q, k, v, lr0, lr1, lr2, self.block_idx, info["ttt_op_order"],
            muon_update_steps=self.muon_update_steps,
            ttt_loss_type=self.ttt_loss_type,
            grad_calc_method=self.grad_calc_method,
            no_query=self.no_query,
        )

        output = self.o_norm(output)
        output = rearrange(
            output, "(b h) l d -> b l (h d)", h=self.num_heads, b=x.shape[0]
        )

        output = self.c_proj(output)
        return output, {"w0": w0, "w1": w1, "w2": w2}

    def extra_repr(self) -> str:
        return (f"w0 shape: {self.w0.shape}, w1 shape: {self.w1.shape}, w2 shape: {self.w2.shape}, "
                f"Muon update steps: {self.muon_update_steps}, "
                f"Base lr: {math.log(1 + math.exp(self.base_lr_inv))}, "
                f"TTT loss type: {self.ttt_loss_type}, ")


