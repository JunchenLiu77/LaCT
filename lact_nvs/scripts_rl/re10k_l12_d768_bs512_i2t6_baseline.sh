#!/bin/bash

NUM_GPUS=4

# Environment variables
export OMP_NUM_THREADS=8
export IBV_FORK_SAFE=1

uv run torchrun \
    --nproc_per_node=$NUM_GPUS \
    --standalone \
    train.py \
    --config config/lact_l12_d768_ttt2x.yaml \
    --seed 77 \
    --expname='re10k_l12_d768_bs512_i2t6_baseline' \
    --data_path='data_example/re10k_preprocessed/train.zip' \
    --test_data_path='data_example/re10k_preprocessed/test.zip' \
    --dataset_type re10k \
    --save_every 500 \
    --log_every 10 \
    --actckpt \
    --bs_per_gpu 128 \
    --num_all_views 8 \
    --num_input_views 2 \
    --num_target_views 6 \
    --image_size 128 128 \
    --scene_pose_normalize \
    --lr 0.0004 \
    --warmup 3000 \
    --steps 20000 \
    --lpips_start 1000000 \
    --test_every 100 \
    --test_bs_per_gpu 128 \
    --scene_inference \
    --first_n 1 \
    --grad_clip 1.0 \
    --ttt_loss_type='dot_product' \
    --grad_calc_method='mannual' \
    --use_fused