# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

def block_causal_lact_swiglu(
    *args,
    **kwargs,
):
    assert "loss_type" in kwargs, "loss_type is required"
    loss_type = kwargs.pop("loss_type")

    # magic strings
    if "no_wn" in loss_type:
        kwargs["weight_norm"] = False
        loss_type = loss_type.replace("_no_wn", "")
    if "no_lr1" in loss_type:
        kwargs["use_lr1"] = False
        loss_type = loss_type.replace("_no_lr1", "")

    if loss_type == "dot_product":
        from .original import block_causal_lact_swiglu as fn
    elif loss_type == "no_query_dot_product":
        from .no_query import block_causal_lact_swiglu as fn
    elif loss_type == "only_w1":
        from .only_w1 import block_causal_lact_swiglu as fn
    elif loss_type == "ga_dot_product":
        from .ga import block_causal_lact_swiglu as fn
    elif loss_type == "only_w1_straight_qk":
        from .only_w1_straight_qk import block_causal_lact_swiglu as fn
    elif loss_type == "only_w1_parallel":
        from .only_w1_no_wn_parallel import block_causal_lact_swiglu as fn
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")

    return fn(*args, **kwargs)
