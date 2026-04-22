# TTTLA -- LaCT LLM Experiments

LLM Experiment code for [Test-Time Training with KV Binding Is Secretly Linear Attention](https://arxiv.org/abs/2602.21204) (TTTLA).

[[Project Page]](https://research.nvidia.com/labs/sil/projects/tttla/) [[Paper]](https://arxiv.org/abs/2602.21204)

We show analytically that TTT architectures with key-value binding reduce to learned linear attention operators. This directory contains the experiment implementations used in the paper, including empirical studies (Sec. 4) and the progressive reduction from TTT to linear attention (Sec. 6.1), built on top of the [LaCT](https://tianyuanzhang.com/projects/ttt-done-right/) codebase.

## Experiments and Variants

The `_ttt_operation_impl/` directory contains each TTT inner-loop variant, controlled by the `ttt_loss_type` field in the model config JSON. The dispatcher in `_ttt_operation_impl/__init__.py` routes to the correct implementation at runtime.

| Name | Config suffix | `ttt_loss_type` | Description |
|------|--------------|-----------------|-------------|
| Base (LaCT) | `muon` | `dot_product` (default) | Full SwiGLU TTT with Muon, updates w0/w1/w2 |
| GA | `ga_dot_product` | `ga_dot_product` | Gradient ascent instead of descent (Sec. 4.2) |
| No Query | `no_query_dot_product` | `no_query_dot_product` | Replace query with key in output projection (Sec. 4.4) |
| Variant 1 | `variant1` | `only_w1` | Update only final-layer params w1 (Sec. 6.1) |
| Variant 2 | `variant2` | `only_w1_no_wn` | Remove weight normalization (Sec. 6.1) |
| Variant 2 parallel | `variant2_parallel` | `only_w1_no_wn_parallel` | Parallel form of variant 2 (prefix-sum, up to 4x throughput) (Sec. 6.2) |
| Variant 3 | `variant3` | `only_w1_straight_qk_no_wn` | Replace multi-layer MLP with single linear layer (Sec. 6.1) |
| Variant 4 | `variant4` | `only_w1_straight_qk_no_lr1_no_wn` | Remove per-token learnable learning rates (Sec. 6.1) |
| Variant 5 | `variant5` | same as variant 4 | Remove momentum (`use_momentum=false`) (Sec. 6.1) |
| Variant 6 | `variant6` | same as variant 4 | Remove gradient orthogonalization (`use_muon=false, use_momentum=false`), reduces to standard linear attention (Sec. 6.1) |

Magic string modifiers parsed by the dispatcher:
- `_no_wn` in `ttt_loss_type` disables weight normalization
- `_no_lr1` in `ttt_loss_type` disables learned per-token LR (sets LR to 1.0)

## Launching

To run an ablation variant, swap the `--model.config` path to the corresponding config JSON. For example, to run the gradient ascent variant use `configs/760M_lact_swiglu_nh4_fwlow_rank_momentum_muon_ga_dot_product.json`. See the [Training with flame](#training-with-flame) section below for the full launch script.

---

# LaCT Language Model

Code  release for [LaCT](https://tianyuanzhang.com/projects/ttt-done-right/) (Large-Chunk TTT) language model.

The code is based on hugging face transformer [flash-linear-attention](https://github.com/fla-org/flash-linear-attention/tree/main), and [flame](https://github.com/fla-org/flame).



## TODO
- [x] LaCT language model code
  - [x] Test-time training with Momentum optimizer
  - [x] Test-time training with Muon optimizer
- [x] Training code and script [using flame]
- [] Optimized inference code 

## How to run

### Enviroment 
The model file need dependency on [flash-linear-attention](https://github.com/fla-org/flash-linear-attention/tree/main), specifically the `lact_model/layer_lact_swiglu.py` copies the RoPE and Fused RMSNorm from flash-linear-attention.

We recommend setting up the environment following the installation guide.
I used this scripts to install required dependencies: 

```
pip install -r requirement.txt
```


### Training with [flame](https://github.com/fla-org/flame)


[flame](https://github.com/fla-org/flame) is an opensource,  minimal and efficient lanugauge model training framework built on top of `torchtitan`, mostly developed by [Yu Zhang](https://github.com/yzhangcs) and [Songlin Yang](https://github.com/sustcsonglin).  We run all our language model experiments with flame.  

To train with flame, copy the entire `lact_model` folder into [`flame/custom_models/`](https://github.com/fla-org/flame/tree/main/custom_models), then follow flame’s [training instruction](https://github.com/fla-org/flame/tree/main?tab=readme-ov-file#training-recipes). 

Example model config files are provided in `configs/`


Here’s an example training script for the 760M model with Muon test-time training.
For the 3B model (with sequence length of 32K per GPU), using `--activation_checkpoint.selective_ac_option 1` only takes around 25GB memory (model shareded among 32 gpus). 

```
export WANDB_NAME="lact-32k-rope-swa2048-nh4-fw02-id-rank32-momentum-muon-760M-32K-40B-bs32-lr1e-3-cosine-warmup1024-steps40960-init0.02.32gpu"
export DATA_FILES="/mnt/localssd/dataset/hf-long-data-collections-sharded/data-*.arrow"  # change to your data-path

bash train.sh \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder exp/lact/lact-swa2048-rope-fw02-id-rank32-init0.5-gain0.5-nh4-momentum-muon-760M-32K-40B/batch1.seqlen32768.bs32.warmup1024.update1.steps40960.lr1e-3.cosine.32gpu \
  --model.config configs/760M_lact_swiglu_nh4_fwlow_rank_momentum_muon.json \
  --model.tokenizer_path ./fla-hub/transformer-1.3B-100B \
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr 1e-3 \
  --lr_scheduler.warmup_steps 1024 \
  --lr_scheduler.lr_min 0.1 \
  --lr_scheduler.decay_type cosine \
  --training.batch_size 1 \
  --training.seq_len 32768 \
  --training.context_len 32768 \
  --training.gradient_accumulation_steps 1 \
  --activation_checkpoint.mode selective \
  --activation_checkpoint.selective_ac_option 2 \  # you can disable activation checkpoint for faster training, but larger memory consuption. 
  --training.steps 40960 \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.dataset arrow \
  --training.dataset_split train \
  --training.num_workers 2 \
  --training.prefetch_factor 1 \
  --training.seed 42 \
  --training.compile \
  --checkpoint.interval 4096 \
  --checkpoint.load_step -1 \
  --checkpoint.keep_latest_k 2 \
  --metrics.log_freq 10 \
  --profiling.profile_freq 2000 \
  1>>./log/$JOB_NAME/$JOB_UUID.rank$RANK.log 2>>./log/$JOB_NAME/$JOB_UUID.rank$RANK.err
```

### Training with your own trainer

If you want to train the model with your own trainer, I recommend you to copy only the `lact_model/layer_lact_swiglu.py` your codebase and integrate it into your model implementation. Note  `lact_model/layer_lact_swiglu.py` reuses RoPE and Fused RMSNorm implementations from flash-linear-attention.