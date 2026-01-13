import torch

def fast_weight_swish_glu_weight_norm_mini_batch_apply(
    *args,
    ttt_loss_type: str,
    no_query: bool,
    grad_calc_method: str,
    **kwargs,
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
    else:
        raise NotImplementedError(f"Unknown ttt_loss_type: {ttt_loss_type}")
    return fn(*args, **kwargs)