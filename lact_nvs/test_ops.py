import torch

# Disable torch.compile for unit testing to ensure numerical consistency
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True
# torch._dynamo.disable()

from _lact_ttt import fast_weight_swish_glu_weight_norm_mini_batch_apply
from lact_ttt import (
    fast_weight_swish_glu_weight_norm_mini_batch_apply_fused,
    TTTOperator,
    TTTOperator_fused,
)


def test_data():
    b = 2
    d = 5
    dh = 7
    l = 13
    num_update_tokens = 9
    num_apply_tokens = 13
    muon_update_steps = 5

    device = "cuda:0"
    dtype = torch.bfloat16
    w0 = torch.randn(b, d, dh, device=device, dtype=dtype)
    w1 = torch.randn(b, dh, d, device=device, dtype=dtype)
    w2 = torch.randn(b, d, dh, device=device, dtype=dtype)
    q = torch.randn(b, l, d, device=device, dtype=dtype)
    k = torch.randn(b, l, d, device=device, dtype=dtype)
    v = torch.randn(b, l, d, device=device, dtype=dtype)
    lr0 = torch.randn(b, l, 1, device=device, dtype=dtype)
    lr1 = torch.randn(b, l, 1, device=device, dtype=dtype)
    lr2 = torch.randn(b, l, 1, device=device, dtype=dtype)
    
    data = {
        "w0": w0,
        "w1": w1,
        "w2": w2,
        "q": q,
        "k": k,
        "v": v,
        "lr0": lr0,
        "lr1": lr1,
        "lr2": lr2,
        "num_update_tokens": num_update_tokens,
        "num_apply_tokens": num_apply_tokens,
        "muon_update_steps": muon_update_steps,
    }
    return data


def test():
    """
    Test that the baseline implementation matches the fused implementation
    with default settings (dot_product loss, mannual grad_calc_method).
    """
    data = test_data()
    
    # Baseline uses TTTOperator with (start, end, update, apply)
    # For baseline: update first, then apply
    ttt_ua_order = [
       TTTOperator(start=0, end=data["num_update_tokens"], update=True, apply=False),
       TTTOperator(start=0, end=data["num_apply_tokens"], update=False, apply=True),
    ]

    # Fused uses TTTOperator_fused
    ttt_op_order = [TTTOperator_fused(num_update_tokens=data["num_update_tokens"], num_apply_tokens=data["num_apply_tokens"])]
    
    args = {
        "baseline": {"ttt_loss_type": "dot_product", "grad_calc_method": "mannual", "no_query": False},
        "no_query": {"ttt_loss_type": "dot_product", "grad_calc_method": "mannual", "no_query": True},
        "mse": {"ttt_loss_type": "mse", "grad_calc_method": "mannual", "no_query": False},
        "ga_dot_product": {"ttt_loss_type": "ga_dot_product", "grad_calc_method": "mannual", "no_query": False},
        "only_w1": {"ttt_loss_type": "only_w1", "grad_calc_method": "mannual", "no_query": False},
        # "only_w1_straight_qk": {"ttt_loss_type": "only_w1_straight_qk", "grad_calc_method": "mannual", "no_query": False},
    }

    for name, kwargs in args.items():
        # Run seperate file version
        output, w0_out, w1_out, w2_out = fast_weight_swish_glu_weight_norm_mini_batch_apply(
            w0=data["w0"],
            w1=data["w1"],
            w2=data["w2"],
            q=data["q"],
            k=data["k"],
            v=data["v"],
            lr0=data["lr0"],
            lr1=data["lr1"],
            lr2=data["lr2"],
            ttt_ua_order=ttt_ua_order,
            muon_update_steps=data["muon_update_steps"],
            **kwargs,
        )
        
        # Run fused implementation
        output_, w0_out_, w1_out_, w2_out_ = fast_weight_swish_glu_weight_norm_mini_batch_apply_fused(
            w0=data["w0"],
            w1=data["w1"],
            w2=data["w2"],
            q=data["q"],
            k=data["k"],
            v=data["v"],
            lr0=data["lr0"],
            lr1=data["lr1"],
            lr2=data["lr2"],
            block_idx=0,
            ttt_op_order=ttt_op_order,
            muon_update_steps=data["muon_update_steps"],
            **kwargs,
        )
        
        # Compare outputs
        torch.testing.assert_close(output, output_, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(w0_out, w0_out_, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(w1_out, w1_out_, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(w2_out, w2_out_, atol=1e-2, rtol=1e-2)
        print(f"{name} passed")


if __name__ == "__main__":
    test()