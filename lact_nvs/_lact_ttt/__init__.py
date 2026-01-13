import torch

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
    block_idx: int,  # not used by baseline implementations, but kept for API compatibility
    ttt_op_order: list,
    muon_update_steps: int,
    ttt_loss_type: str,
    no_query: bool,
    grad_calc_method: str,
):  
    if ttt_loss_type == "dot_product":
        if not no_query:
            from .baseline import fn
        else:
            from .no_query import fn
    elif ttt_loss_type == "mse":
        from .mse import fn
    elif ttt_loss_type == "ga_dot_product":
        from .ga import fn
    elif ttt_loss_type == "only_w1":
        from .only_w1 import fn
    elif ttt_loss_type == "only_w1_straight_qk":
        from .only_w1_straight_qk import fn
    elif ttt_loss_type == "only_w1_straight_qk_no_lr1":
        from .only_w1_straight_qk_no_lr1 import fn
    elif ttt_loss_type == "only_w1_straight_qk_no_lr1_no_wn":
        from .only_w1_straight_qk_no_lr1_no_wn import fn
    elif ttt_loss_type == "only_w1_straight_qk_no_lr1_no_muon":
        from .only_w1_straight_qk_no_lr1_no_muon import fn
    elif ttt_loss_type == "only_w1_straight_qk_no_lr1_no_wn_muon":
        from .only_w1_straight_qk_no_lr1_no_wn_muon import fn
    else:
        raise NotImplementedError(f"Unknown ttt_loss_type: {ttt_loss_type}")
    return fn(w0, w1, w2, q, k, v, lr0, lr1, lr2, ttt_op_order, muon_update_steps=muon_update_steps)