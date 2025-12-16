import collections
import math

import torch
from torch import nn

import torch.nn.functional as F
from einops import rearrange

TTTOperator = collections.namedtuple("TTTOperator", ["start", "end", "update", "apply"])

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


# @torch.compile
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
    ttt_ua_order: list,
    muon_update_steps: int = 0,
    use_learnable_opt: bool = False,
    opt_type: str = "",
    opts: nn.ModuleList = None,
    residual: bool = True,
    normalize_weight: bool = True,
    output_norm_method: str = "none",
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
    use_learnable_opt: if True, the opts will be used to update the weights.
    opt_type: "dit" or "mlp". If "dit", use the dit optimizer to update the weights. If "mlp", use a simple MLP to update the weights.
    opts: if use_learnable_opt is True, this should contains three learnable optimizers to update w0, w1, w2.
    residual: if True, use the residual update for the weights.
    normalize_weight: if True, normalize the weights after updating.
    output_norm_method: "none", "mean_std" or "affine". Method to normalize the output.
    """
    if use_learnable_opt:
        assert opt_type in ["dit", "mlp"], f"opt_type should be 'dit' or 'mlp', but got {opt_type}"
        assert opts is not None, "opts should be provided if use_learnable_opt is True"
        assert len(opts) == 3, "opts should contain 3 learnable optimizers"

    if normalize_weight:
        w0_norm = w0.detach().norm(dim=1, keepdim=True)
        w1_norm = w1.detach().norm(dim=1, keepdim=True)
        w2_norm = w2.detach().norm(dim=1, keepdim=True)

    d = w0.shape[1]

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

            dhidden = vi @ w1_now.transpose(-1, -2)  # [b, l, dh] = [b, l, d] @ [b, d, dh]
            dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
            dgate = dhidden * hidden_before_mul
            dgate_before_act = silu_backprop(dgate, gate_before_act)

            w1_grad = ((hidden * lr1i).transpose(-1, -2) @ vi)
            w0_grad = ((ki * lr0i).transpose(-1, -2) @ dgate_before_act)
            w2_grad = ((ki * lr2i).transpose(-1, -2) @ dhidden_before_mul)

            # orthogonalized gradients
            w1_grad = zeropower_via_newtonschulz5(w1_grad, muon_update_steps)
            w0_grad = zeropower_via_newtonschulz5(w0_grad, muon_update_steps)
            w2_grad = zeropower_via_newtonschulz5(w2_grad, muon_update_steps)

            if not use_learnable_opt:
                w1_update = w1_grad
                w0_update = w0_grad
                w2_update = w2_grad
            else:
                if opt_type == "dit":
                    # Add positional encoding on the sequence dimension L to the w_now part only
                    # For w1 (shape [b, dh, d])
                    L1, D1 = w1_now.shape[1], w1_now.shape[2]
                    pos1 = torch.arange(L1, device=w1_now.device, dtype=torch.float32).unsqueeze(1)
                    div1 = torch.exp(torch.arange(0, D1, 2, device=w1_now.device, dtype=torch.float32) * (-math.log(10000.0) / max(1, D1)))
                    pe1 = torch.zeros(L1, D1, device=w1_now.device, dtype=torch.float32)
                    pe1[:, 0::2] = torch.sin(pos1 * div1)
                    pe1[:, 1::2] = torch.cos(pos1 * div1)
                    pe1 = pe1.to(w1_now.dtype).unsqueeze(0)  # [1, L1, D1]
                    w1_now_pe = w1_now + pe1
                    # w1_now_pe = torch.zeros_like(w1_now) + pe1
                    opt1_input = torch.cat([w1_now_pe, w1_grad], dim=2)  # [b, L1, 2*D1]

                    # For w0 (shape [b, d, dh] -> [b, dh, d])
                    L0, D0 = w0_now.shape[2], w0_now.shape[1]
                    pos0 = torch.arange(L0, device=w0_now.device, dtype=torch.float32).unsqueeze(1)
                    div0 = torch.exp(torch.arange(0, D0, 2, device=w0_now.device, dtype=torch.float32) * (-math.log(10000.0) / max(1, D0)))
                    pe0 = torch.zeros(L0, D0, device=w0_now.device, dtype=torch.float32)
                    pe0[:, 0::2] = torch.sin(pos0 * div0)
                    pe0[:, 1::2] = torch.cos(pos0 * div0)
                    pe0 = pe0.to(w0_now.dtype).unsqueeze(0)  # [1, L0, D0]
                    w0_now_seq = rearrange(w0_now, "b d dh -> b dh d") + pe0
                    # w0_now_seq = rearrange(torch.zeros_like(w0_now), "b d dh -> b dh d") + pe0
                    w0_grad_seq = rearrange(w0_grad, "b d dh -> b dh d")
                    opt0_input = torch.cat([w0_now_seq, w0_grad_seq], dim=2)  # [b, L0, 2*D0]

                    # For w2 (shape [b, d, dh] -> [b, dh, d])
                    L2, D2 = w2_now.shape[2], w2_now.shape[1]
                    pos2 = torch.arange(L2, device=w2_now.device, dtype=torch.float32).unsqueeze(1)
                    div2 = torch.exp(torch.arange(0, D2, 2, device=w2_now.device, dtype=torch.float32) * (-math.log(10000.0) / max(1, D2)))
                    pe2 = torch.zeros(L2, D2, device=w2_now.device, dtype=torch.float32)
                    pe2[:, 0::2] = torch.sin(pos2 * div2)
                    pe2[:, 1::2] = torch.cos(pos2 * div2)
                    pe2 = pe2.to(w2_now.dtype).unsqueeze(0)  # [1, L2, D2]
                    w2_now_seq = rearrange(w2_now, "b d dh -> b dh d") + pe2
                    # w2_now_seq = rearrange(torch.zeros_like(w2_now), "b d dh -> b dh d") + pe2
                    w2_grad_seq = rearrange(w2_grad, "b d dh -> b dh d")
                    opt2_input = torch.cat([w2_now_seq, w2_grad_seq], dim=2)  # [b, L2, 2*D2]

                    t = 0.0 # only use one iterations for now
                    t_vec = torch.full((opt1_input.shape[0],), t, device=opt1_input.device)
                    w1_update = opts[1](opt1_input, t_vec)[..., d:]
                    w0_update = rearrange(opts[0](opt0_input, t_vec)[..., d:], "b dh d -> b d dh")
                    w2_update = rearrange(opts[2](opt2_input, t_vec)[..., d:], "b dh d -> b d dh")

                elif opt_type == "mlp":
                    # use a simple MLP to update the weights
                    L1, D1 = w1_now.shape[1], w1_now.shape[2]
                    opt1_input = torch.stack([w1_now, w1_grad], dim=-1).reshape(-1, 2)   # [b * L1 * D1, 2]
                    w1_update = opts[1](opt1_input) # [b * L1 * D1, 2] -> [b * L1 * D1, 1]
                    w1_update = w1_update.reshape(w1_now.shape[0], L1, D1) # [b * L1 * D1, 1] -> [b, L1, D1]

                    L0, D0 = w0_now.shape[2], w0_now.shape[1]
                    opt0_input = torch.stack([w0_now, w0_grad], dim=-1).reshape(-1, 2)   # [b * L0 * D0, 2]
                    w0_update = opts[0](opt0_input) # [b * L0 * D0, 2] -> [b * L0 * D0, 1]
                    w0_update = w0_update.reshape(w0_now.shape[0], D0, L0) # [b * L0 * D0, 1] -> [b, D0, L0]

                    L2, D2 = w2_now.shape[2], w2_now.shape[1]
                    opt2_input = torch.stack([w2_now, w2_grad], dim=-1).reshape(-1, 2)   # [b * L2 * D2, 2]
                    w2_update = opts[2](opt2_input) # [b * L2 * D2, 2] -> [b * L2 * D2, 1]
                    w2_update = w2_update.reshape(w2_now.shape[0], D2, L2) # [b * L2 * D2, 1] -> [b, D2, L2]

            if residual:
                w1_now = w1_now + w1_update
                w0_now = w0_now + w0_update
                w2_now = w2_now + w2_update
            else:
                w1_now = w1_update
                w0_now = w0_update
                w2_now = w2_update

            if normalize_weight:
                # do weight norm here
                w0_now = w0_now / (w0_now.norm(dim=1, keepdim=True) + 1e-5) * w0_norm
                w1_now = w1_now / (w1_now.norm(dim=1, keepdim=True) + 1e-5) * w1_norm
                w2_now = w2_now / (w2_now.norm(dim=1, keepdim=True) + 1e-5) * w2_norm

            w0, w1, w2 = w0_now, w1_now, w2_now

        if apply:
            # Only calculate the output in the last repeat.
            qi = q[:, start:end, :]
            oi = (F.silu(qi @ w0_now, inplace=True) * (qi @ w2_now)) @ w1_now
            # pull oi to the domain of vi
            if output_norm_method == "mean_std":
                oi = (oi - oi.mean(dim=1, keepdim=True)) / (oi.std(dim=1, keepdim=True) + 1e-5)
                oi = oi * vi.std(dim=1, keepdim=True) + vi.mean(dim=1, keepdim=True)
            elif output_norm_method == "affine":
                raise NotImplementedError("Affine output normalization is not implemented yet")
                # v_mean = vi.mean(dim=1, keepdim=True)
                # v_cent = vi - v_mean
                # o_cent = oi - v_mean
                
                # # affine projection
                # # compute in float32 for stability
                # v_cent_f = v_cent.float()
                # o_cent_f = o_cent.float()
                # # P = V+ @ V (project onto row space of V)
                # P = torch.linalg.pinv(v_cent_f) @ v_cent_f
                # oi = (o_cent_f @ P).to(oi.dtype) + v_mean
            elif output_norm_method == "none":
                pass
            else:
                 raise ValueError(f"Unknown output_norm_method: {output_norm_method}")

            output.append(oi)

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
    Then with l2_norm of q,k, input is unit normed.
    if w0 is initialized with kaiming, relu(w0 @ q) is unit normed.
    Then w1 is initialized with kaiming, so w1 @ relu(w0 @ q) is of norm sqrt(2) per head
    Since I compute total norm, it is sqrt(2) * sqrt(num_head), which is around 2.7 for dim=512, num_head=4.
    """

    def __init__(
        self,
        dim: int,
        head_dim: int,
        inter_multi: int = 1,
        bias: bool = False,
        base_lr=0.01,
        muon_update_steps=0,
        use_learnable_opt: bool = False,
        opt_type: str = "",
        n_blocks_per_opt: int = 2,
        opt_hidden_dim: int = 256,
        shared_opts: nn.ModuleList = None,
        residual: bool = True,
        normalize_weight: bool = True,
        output_norm_method: str = "none",
    ):
        super().__init__()
        self.dim = dim
        assert dim % head_dim == 0
        self.num_heads = dim // head_dim
        self.muon_update_steps = muon_update_steps

        d_in = d_out = head_dim
        d_h = int(head_dim * inter_multi)

        gain = math.sqrt(2)  # for relu activations
        self.w0 = nn.Parameter(
            torch.randn(self.num_heads, d_in, d_h) * gain / math.sqrt(d_in)
        )  # [num_heads, d_in, d_h]
        self.w1 = nn.Parameter(
            torch.randn(self.num_heads, d_h, d_out) * gain / math.sqrt(d_h)
        )  # [num_heads, d_h, d_out]
        self.w2 = nn.Parameter(
            torch.randn(self.num_heads, d_in, d_h) * gain / math.sqrt(d_in)
        )  # [num_heads, d_in, d_h]

        self.to_qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.c_proj = nn.Linear(dim, dim, bias=bias)

        self.lr_dim = self.num_heads
        self.lr_fc = nn.Linear(dim, self.lr_dim * 3)
        self.base_lr_inv = inv_softplus(base_lr)

        self.o_norm = nn.RMSNorm(head_dim, eps=1e-5, elementwise_affine=True)

        # learnable opt.
        self.use_learnable_opt = use_learnable_opt
        self.opt_type = opt_type
        self.residual = residual
        self.normalize_weight = normalize_weight
        self.output_norm_method = output_norm_method
        self.opt_hidden_dim = opt_hidden_dim
        self.n_blocks_per_opt = n_blocks_per_opt
        if use_learnable_opt:
            if shared_opts is not None:
                assert len(shared_opts) >= 3, f"shared_opts should contain at least 3 learnable optimizers, but got {len(shared_opts)}"
                self.opts = shared_opts
            else:
                raise NotImplementedError("shared_opts is not supported for learnable opt")
        
    def forward(self, x: torch.Tensor, info={}, *args):
        """
        x: (b, l, d)
        """
        qkv = F.silu(self.to_qkv(x), inplace=True)  # Silu - Linear
        q, k, v = rearrange(
            qkv, "b l (qkv h d) -> qkv (b h) l d",
            qkv=3, h=self.num_heads
        )
        q = q / (q.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype)
        k = k / (k.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype)

        with torch.autocast(device_type="cuda", enabled=False):
            lr = self.lr_fc(x)  # [b, l, lr_dim]

        lr = torch.nn.functional.softplus(lr + self.base_lr_inv)
        lr0, lr1, lr2 = rearrange(
            lr, "b l (lrs h d) -> lrs (b h) l d",
            lrs=3, h=self.num_heads
        )

        if "w0" in info:
            assert "w1" in info and "w2" in info
            w0 = info["w0"]
            w1 = info["w1"]
            w2 = info["w2"]
        else:
            w0 = self.w0.repeat(x.shape[0], 1, 1)
            w1 = self.w1.repeat(x.shape[0], 1, 1)
            w2 = self.w2.repeat(x.shape[0], 1, 1)

        output, w0, w1, w2 = fast_weight_swish_glu_weight_norm_mini_batch_apply(
            w0, w1, w2, q, k, v, lr0, lr1, lr2, info["ttt_op_order"],
            muon_update_steps=self.muon_update_steps,
            use_learnable_opt=self.use_learnable_opt,
            opt_type=self.opt_type,
            opts=self.opts if self.use_learnable_opt else None,
            residual=self.residual,
            normalize_weight=self.normalize_weight,
            output_norm_method=self.output_norm_method,
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
                f"Base lr: {math.log(1 + math.exp(self.base_lr_inv))}, ")


