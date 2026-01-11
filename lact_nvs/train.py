import argparse
import csv
import functools
import math
import os
import random

import lpips
import numpy as np
import omegaconf
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers.optimization import get_cosine_schedule_with_warmup
import wandb
import time

from data import NVSDataset, Re10kNVSDataset
from model import LaCTLVSM
from inference import get_turntable_cameras_with_zoom_in, get_interpolated_cameras
from PIL import Image
import imageio
import json


def run_evaluation(model, test_loader, lpips_loss_module, args, test_dir, 
                   first_n=None, save_images=True, ddp_local_rank=0):
    """
    Run evaluation on test set.
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader for test set
        lpips_loss_module: LPIPS loss module
        args: Command line arguments
        test_dir: Directory to save test results
        first_n: Number of batches per rank to evaluate (None = all)
        save_images: Whether to save visualization images
        ddp_local_rank: Local rank for distributed training
        
    Returns:
        List of per-scene results with per-view metrics
    """
    os.makedirs(test_dir, exist_ok=True)
    print(f"[{dist.get_rank():02d}, {ddp_local_rank:02d}] Testing to {test_dir}...")

    # Collect all test results for this rank
    rank_test_results = []
    test_iter = iter(test_loader)

    def tensor_to_numpy(tensor):
        """Convert tensor to numpy RGB image."""
        numpy_image = tensor.permute(1, 2, 0).cpu().numpy()
        numpy_image = np.clip(numpy_image * 255, 0, 255).astype(np.uint8)
        return numpy_image

    for sample_idx, data_dict in enumerate(test_iter):
        if first_n is not None and sample_idx >= first_n:
            break
        print(f"[{dist.get_rank():02d}, {ddp_local_rank:02d}] Testing sample {sample_idx:07d}...")
        indices = data_dict["indices"]
        scene_names = data_dict["scene_name"]
        data_dict = {key: value.cuda() for key, value in data_dict.items() if isinstance(value, torch.Tensor)}
        
        # For re10k test, always use 2 input views and 3 target views
        num_input_views = 2 if args.dataset_type == "re10k" else args.num_input_views
        num_target_views = 3 if args.dataset_type == "re10k" else args.num_target_views
        
        input_data_dict = {key: value[:, :num_input_views] for key, value in data_dict.items()}
        target_data_dict = {key: value[:, -num_target_views:] for key, value in data_dict.items()}

        with torch.autocast(dtype=torch.bfloat16, device_type="cuda", enabled=True) and torch.no_grad():
            rendering = model(input_data_dict, target_data_dict)

            batch_size, num_views = rendering.shape[:2]
            for batch_idx in range(batch_size):
                scene_name = scene_names[batch_idx]

                # Compute per-view metrics
                target = target_data_dict["image"][batch_idx]
                rendered = rendering[batch_idx]
                
                # Calculate PSNR per view
                mse_per_view = F.mse_loss(rendered, target, reduction='none').mean(dim=[1, 2, 3])
                psnr_per_view = -10.0 * torch.log10(mse_per_view)
                psnr_per_view_list = psnr_per_view.cpu().tolist()
                avg_psnr = sum(psnr_per_view_list) / len(psnr_per_view_list)

                # Calculate LPIPS per view
                lpips_per_view = lpips_loss_module(rendered, target, normalize=True)
                lpips_per_view_list = lpips_per_view.squeeze().cpu().tolist()
                if not isinstance(lpips_per_view_list, list):
                    lpips_per_view_list = [lpips_per_view_list]
                avg_lpips = sum(lpips_per_view_list) / len(lpips_per_view_list)

                # Determine indices used for input and target
                sample_indices = indices[batch_idx].cpu().tolist()
                total_views_count = len(sample_indices)
                input_indices_local = list(range(num_input_views))
                target_indices_local = list(range(total_views_count - num_target_views, total_views_count))

                # Map local indices to original dataset indices
                input_indices_original = [sample_indices[i] for i in input_indices_local]
                target_indices_original = [sample_indices[i] for i in target_indices_local]

                # Store metrics with per-view details
                result = {
                    "scene_name": scene_name,
                    "input_indices": input_indices_original,
                    "target_indices": target_indices_original,
                    "psnr_per_view": psnr_per_view_list,
                    "lpips_per_view": lpips_per_view_list,
                    "avg_psnr": avg_psnr,
                    "avg_lpips": avg_lpips,
                }
                rank_test_results.append(result)

                # Only save the first scene image in the first batch (one image per rank per test)
                if save_images and sample_idx == 0 and batch_idx == 0:
                    # Collect all images for this batch
                    input_images = []
                    rendered_images = []
                    target_images = []
                    
                    # Get input images
                    input_img_tensor = input_data_dict["image"][batch_idx]
                    for view_idx in range(input_img_tensor.shape[0]):
                        input_images.append(tensor_to_numpy(input_img_tensor[view_idx]))
                    
                    # Get target and rendered images
                    for view_idx in range(num_views):
                        rendered_images.append(tensor_to_numpy(rendered[view_idx]))
                        target_images.append(tensor_to_numpy(target[view_idx]))
                    
                    # Concatenate images horizontally
                    input_row = np.concatenate(input_images, axis=1)
                    target_row = np.concatenate(target_images, axis=1)
                    rendered_row = np.concatenate(rendered_images, axis=1)
                    
                    # Pad input_row to match target_row width if different number of views
                    if input_row.shape[1] != target_row.shape[1]:
                        pad_width = target_row.shape[1] - input_row.shape[1]
                        if pad_width > 0:
                            input_row = np.concatenate([input_row, np.zeros((input_row.shape[0], pad_width, 3), dtype=np.uint8)], axis=1)
                        else:
                            input_row = input_row[:, :target_row.shape[1], :]
                    
                    # Stack all three rows vertically: input, target (GT), rendered
                    combined_image = np.concatenate([input_row, target_row, rendered_row], axis=0)
                    
                    # Save the concatenated image directly in test_dir as scene_name.png
                    Image.fromarray(combined_image).save(os.path.join(test_dir, f"{scene_name}.png"))

    torch.cuda.empty_cache()
    dist.barrier()

    return rank_test_results


def aggregate_and_save_results(gathered_results, test_dir, wandb_prefix="test", now_iters=0, 
                               num_input_views=2, num_target_views=3):
    """
    Aggregate results from all ranks and save to CSV.
    
    Args:
        gathered_results: List of results from all ranks
        test_dir: Directory to save results
        wandb_prefix: Prefix for wandb logging (e.g., "test" or "final_test")
        now_iters: Current iteration for wandb logging
        num_input_views: Number of input views (for CSV header)
        num_target_views: Number of target views (for CSV header)
        
    Returns:
        Tuple of (avg_psnr, avg_lpips)
    """
    # Aggregate all results
    all_results = []
    psnr_list = []
    lpips_list = []
    # Also collect all per-view metrics for overall average
    all_psnr_per_view = []
    all_lpips_per_view = []
    
    for rank_results in gathered_results:
        all_results.extend(rank_results)
        for res in rank_results:
            psnr_list.append(res["avg_psnr"])
            lpips_list.append(res["avg_lpips"])
            # Collect all per-view metrics
            all_psnr_per_view.extend(res["psnr_per_view"])
            all_lpips_per_view.extend(res["lpips_per_view"])

    avg_psnr = np.array(psnr_list, dtype=np.float32).mean()
    avg_lpips = np.array(lpips_list, dtype=np.float32).mean()
    
    # Calculate overall average across all scenes and all views
    overall_avg_psnr = np.array(all_psnr_per_view, dtype=np.float32).mean()
    overall_avg_lpips = np.array(all_lpips_per_view, dtype=np.float32).mean()
    
    # Calculate per-view averages across all scenes (for the summary row)
    per_view_avg_psnr = []
    per_view_avg_lpips = []
    for view_idx in range(num_target_views):
        view_psnrs = [res["psnr_per_view"][view_idx] for res in all_results if view_idx < len(res["psnr_per_view"])]
        view_lpips = [res["lpips_per_view"][view_idx] for res in all_results if view_idx < len(res["lpips_per_view"])]
        per_view_avg_psnr.append(np.array(view_psnrs, dtype=np.float32).mean() if view_psnrs else 0.0)
        per_view_avg_lpips.append(np.array(view_lpips, dtype=np.float32).mean() if view_lpips else 0.0)
    
    # Build CSV header dynamically based on number of views
    header = ["scene_name"]
    for i in range(num_input_views):
        header.append(f"input_idx{i+1}")
    for i in range(num_target_views):
        header.append(f"target_idx{i+1}")
    for i in range(num_target_views):
        header.append(f"psnr{i+1}")
        header.append(f"lpips{i+1}")
    header.extend(["avg_psnr", "avg_lpips"])
    
    # Save results to CSV
    csv_path = os.path.join(test_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for res in all_results:
            row = [res["scene_name"]]
            # Add input indices (pad with empty if fewer than expected)
            for i in range(num_input_views):
                if i < len(res["input_indices"]):
                    row.append(res["input_indices"][i])
                else:
                    row.append("")
            # Add target indices
            for i in range(num_target_views):
                if i < len(res["target_indices"]):
                    row.append(res["target_indices"][i])
                else:
                    row.append("")
            # Add per-view psnr and lpips
            for i in range(num_target_views):
                if i < len(res["psnr_per_view"]):
                    row.append(f"{res['psnr_per_view'][i]:.4f}")
                else:
                    row.append("")
                if i < len(res["lpips_per_view"]):
                    row.append(f"{res['lpips_per_view'][i]:.4f}")
                else:
                    row.append("")
            # Add averages
            row.append(f"{res['avg_psnr']:.4f}")
            row.append(f"{res['avg_lpips']:.4f}")
            writer.writerow(row)
        
        # Add summary row with overall averages
        summary_row = ["AVERAGE"]
        # Empty cells for input indices
        for _ in range(num_input_views):
            summary_row.append("")
        # Empty cells for target indices
        for _ in range(num_target_views):
            summary_row.append("")
        # Per-view average PSNR and LPIPS
        for i in range(num_target_views):
            summary_row.append(f"{per_view_avg_psnr[i]:.4f}")
            summary_row.append(f"{per_view_avg_lpips[i]:.4f}")
        # Overall averages
        summary_row.append(f"{overall_avg_psnr:.4f}")
        summary_row.append(f"{overall_avg_lpips:.4f}")
        writer.writerow(summary_row)
    
    # Log to wandb
    print(f"[{now_iters:07d}] {wandb_prefix} - Average PSNR = {overall_avg_psnr:.2f}, Average LPIPS = {overall_avg_lpips:.4f}, Scenes = {len(all_results)}")
    wandb.log({
        f"{wandb_prefix}/psnr": overall_avg_psnr,
        f"{wandb_prefix}/lpips": overall_avg_lpips,
    }, step=now_iters)
    
    return overall_avg_psnr, overall_avg_lpips

def main():
    parser = argparse.ArgumentParser()
    # Basic info
    parser.add_argument("--config", type=str, default="config/lact")
    parser.add_argument("--expname", type=str, default="default")
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--data_path", type=str, default="data_example/gso_sample_data_path.json")
    parser.add_argument("--test_data_path", type=str, default=None, help="Path to test dataset (if different from train)")
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=95)

    # Training
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--actckpt", action="store_true")
    parser.add_argument("--bs_per_gpu", type=int, default=8)
    
    parser.add_argument("--num_all_views", type=int, default=15)
    parser.add_argument("--num_input_views", type=int, default=8)
    parser.add_argument("--num_target_views", type=int, default=8)  
    parser.add_argument("--image_size", nargs=2, type=int, default=[256, 256], help="Image size H, W")
    parser.add_argument("--scene_pose_normalize", action="store_true")
    parser.add_argument("--fdist_min", type=int, default=None, help="Minimum frame index (None = start of video)")
    parser.add_argument("--fdist_max", type=int, default=None, help="Maximum frame index (None = end of video)")
    parser.add_argument("--dataset_type", type=str, default="dl3dv", choices=["dl3dv", "re10k"], help="Dataset type: dl3dv or re10k")
    parser.add_argument("--target_has_input", action="store_true", default=True, help="[Re10k] Whether target views can overlap with input views")
    parser.add_argument("--no_target_has_input", action="store_false", dest="target_has_input", help="[Re10k] Target views cannot overlap with input views")

    # Inference
    parser.add_argument("--test_every", type=int, default=-1, help="Test every N iterations")
    parser.add_argument("--first_n", type=int, default=None, help="Number of test batches per GPU rank (total scenes = first_n * bs * num_gpus)")
    parser.add_argument("--test_bs_per_gpu", type=int, default=1)
    parser.add_argument("--scene_inference", action="store_true")

    # Optimizer
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=4000)
    parser.add_argument("--steps", type=int, default=80000)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--lpips_start", type=int, default=5000, help="Iteration to start LPIPS loss")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")

    # Model config
    parser.add_argument("--ttt_loss_type", type=str, default=None, help="TTT loss type: dot_product, mse, rmse, mae, etc.")
    parser.add_argument("--grad_calc_method", type=str, default="mannual", help="Gradient calculation method: mannual, autograd")
    parser.add_argument("--use_fused", action="store_true", help="Use fused TTT")
    parser.add_argument("--no_query", action="store_true", help="No query in TTT")
    
    args = parser.parse_args()
    model_config = omegaconf.OmegaConf.load(args.config)

    if args.ttt_loss_type is not None or args.grad_calc_method is not None or args.no_query is not None or args.use_fused is not None:
        for block in model_config.block_config:
            if block.type == "lact_ttt.FastWeightGluMLPMultihead":
                if args.ttt_loss_type is not None:
                    block.params.ttt_loss_type = args.ttt_loss_type
                if args.grad_calc_method is not None:
                    block.params.grad_calc_method = args.grad_calc_method
                if args.no_query is not None:
                    block.params.no_query = args.no_query
                if args.use_fused is not None:
                    block.params.use_fused = args.use_fused
                
    output_dir = f"output/{args.expname}"
    os.makedirs(output_dir, exist_ok=True)

    dist.init_process_group(backend="nccl")
    ddp_local_rank = int(os.environ.get("LOCAL_RANK", dist.get_rank() % 8))
    torch.cuda.set_device(ddp_local_rank)
    print(f"[{dist.get_rank():02d}, {ddp_local_rank:02d}] device: {torch.cuda.current_device()}")

    # Seed everything for reproducibility
    rank_specific_seed = args.seed + dist.get_rank()
    print(f"[{dist.get_rank():02d}, {ddp_local_rank:02d}] seed: {rank_specific_seed}")
    
    random.seed(rank_specific_seed)
    np.random.seed(rank_specific_seed)
    torch.manual_seed(rank_specific_seed)
    torch.cuda.manual_seed(rank_specific_seed)
    torch.cuda.manual_seed_all(rank_specific_seed)
    
    # cuDNN determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Enable deterministic algorithms
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    
    dataloader_seed_generator = torch.Generator()
    dataloader_seed_generator.manual_seed(rank_specific_seed)
    
    def worker_init_fn(worker_id):
        """Ensure each DataLoader worker has a unique but reproducible seed."""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    model = LaCTLVSM(**model_config).cuda()

    # Optimizers
    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, betas=(0.9, 0.95), fused=True)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup,
        num_training_steps=args.steps,
    )

    # Load checkpoint
    now_iters = 0
    load_paths = []
    # First try to resume from the output directory
    if os.path.exists(output_dir):
        load_paths.append(output_dir)
    # Then try the explicitly provided load path
    if args.load:
        load_paths.append(args.load)

    for try_load_path in load_paths:
        if try_load_path is None: continue
        try:
            if os.path.isdir(try_load_path):
                checkpoints = [f for f in os.listdir(try_load_path) if f.startswith("model_") and f.endswith(".pth")]
                if not checkpoints: continue
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("_")[1].split(".")[0]))
                checkpoint_path = os.path.join(try_load_path, latest_checkpoint)
            else:
                checkpoint_path = try_load_path
            
            print(f"[{dist.get_rank():02d}, {ddp_local_rank:02d}] Loading checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            now_iters = checkpoint["now_iters"]
            break
        except Exception as e:
            print(f"[{dist.get_rank():02d}, {ddp_local_rank:02d}] Failed to load from {try_load_path}: {e}")
            continue
            
    model = DDP(model, device_ids=[ddp_local_rank])

    # This activation checkpointing wrapper supports torch.compile
    if args.actckpt:
        torch._dynamo.config.optimize_ddp = False
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper as ptd_checkpoint_wrapper,
            apply_activation_checkpointing,
        )

        wrapper = functools.partial(ptd_checkpoint_wrapper, preserve_rng_state=False)

        def _check_fn(submodule) -> bool:
            from model import Block
            return isinstance(submodule, Block)

        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=wrapper,
            check_fn=_check_fn,
        )

    if args.compile:
        model = torch.compile(model)  

    def remove_module_prefix(state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            key = key.replace("_checkpoint_wrapped_module.", "")
            key = key.replace("_orig_mod.", "")
            while key.startswith("module."):
                key = key[len("module."):]
            new_state_dict[key] = value
        return new_state_dict

    # Data - Select dataset class based on dataset_type
    if args.dataset_type == "re10k":
        train_set = Re10kNVSDataset(
            data_path=args.data_path,
            num_input_views=args.num_input_views,
            num_target_views=args.num_target_views,
            image_size=tuple(args.image_size),
            inference=False,
            scene_pose_normalize=args.scene_pose_normalize,
            min_frame_dist=args.fdist_min if args.fdist_min is not None else 25,
            max_frame_dist=args.fdist_max if args.fdist_max is not None else 192,
            target_has_input=args.target_has_input,
        )
    else:
        train_set = NVSDataset(
            data_path=args.data_path, 
            num_views=args.num_all_views, 
            image_size=tuple(args.image_size), 
            scene_pose_normalize=args.scene_pose_normalize, 
            fdist_min=args.fdist_min, 
            fdist_max=args.fdist_max
        )
    train_sampler = DistributedSampler(train_set)
    train_loader = DataLoader(
        train_set,
        batch_size=args.bs_per_gpu,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2, 
        # if you encounter "RuntimeError: Trying to resize storage that is not resizable", try the settings below to expose the real problem
        # source: https://github.com/lucidrains/denoising-diffusion-pytorch/issues/248
        # num_workers=0,
        # pin_memory=False,
        drop_last=False,
        sampler=train_sampler,
        generator=dataloader_seed_generator,    # This ensures deterministic dataloader
        worker_init_fn=worker_init_fn,          # Seed workers for reproducibility
    )

    if args.test_every > 0:
        # Create test dataset based on dataset type
        if args.dataset_type == "re10k":
            # Re10k uses evaluation_index_re10k.json for inference
            # Indices file only support 2 input views and 3 target views
            test_set = Re10kNVSDataset(
                data_path=args.test_data_path if args.test_data_path else args.data_path.replace('train', 'test'),
                num_input_views=2,
                num_target_views=3,
                image_size=tuple(args.image_size),
                inference=True,
                scene_pose_normalize=args.scene_pose_normalize,
                # target_has_input=args.target_has_input, # testset doesn't have view selection
                eval_index_path="data_example/evaluation_index_re10k.json",
            )
        else:
            # DL3DV uses Long-LRM input indices
            fixed_indices_path = "data_example/dl3dv_fold_8_kmeans_input.json"
            assert os.path.exists(fixed_indices_path), f"Fixed test indices file not found at {fixed_indices_path}"
            assert args.num_target_views == args.num_input_views, "For DL3DV, we assume the number of target views equals input views."
            assert args.num_target_views in [16, 32, 64, 128], "DL3DV input indice file only contains 16, 32, 64, 128 views."
            
            if dist.get_rank() == 0:
                print(f"Loading fixed test indices from {fixed_indices_path}")
                
            with open(fixed_indices_path, "r") as f:
                test_indices_map = json.load(f)

            num_views_for_dataset = args.num_input_views + args.num_target_views
            test_set = NVSDataset(
                data_path=args.test_data_path if args.test_data_path else "data_example/dl3dv_benchmark_sample_data_path.json",
                num_views=num_views_for_dataset, 
                image_size=tuple(args.image_size), 
                sorted_indices=False,
                scene_pose_normalize=args.scene_pose_normalize,
                fixed_indices=test_indices_map
            )
        test_sampler = DistributedSampler(test_set)
        test_dataloader_seed_generator = torch.Generator()
        test_dataloader_seed_generator.manual_seed(rank_specific_seed)
        test_loader = DataLoader(
            test_set,
            batch_size=args.test_bs_per_gpu,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=2, 
            sampler=test_sampler,
            generator=test_dataloader_seed_generator,    # This ensures deterministic dataloader
            worker_init_fn=worker_init_fn,               # Seed workers for reproducibility
        )
        test_sampler.set_epoch(0)

    if dist.get_rank() == 0:
        # report training config only on the master
        print(f"Training config:")
        # print(model)
        # print(optimizer)
        # print(lr_scheduler)

        # print the model size
        model_size = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Model size: {model_size:.2f} MB")
        
        if args.actckpt:
            print(f"Activation checkpointing applied")
        print(f"Starting training from iter {now_iters}...")
        
        # Initialize wandb
        wandb_id = None
        wandb_id_path = os.path.join(output_dir, "wandb_id.txt")
        if os.path.exists(wandb_id_path):
            with open(wandb_id_path, "r") as f:
                wandb_id = f.read().strip()
            print(f"Resuming wandb run with id {wandb_id}")

        wandb.init(
            project="lact-nvs-ablation",
            name=args.expname,
            id=wandb_id,
            config={
                "config": args.config,
                "bs_per_gpu": args.bs_per_gpu,
                "num_all_views": args.num_all_views,
                "num_input_views": args.num_input_views,
                "num_target_views": args.num_target_views,
                "image_size": args.image_size,
                "lr": args.lr,
                "warmup": args.warmup,
                "steps": args.steps,
                "weight_decay": args.weight_decay,
                "lpips_start": args.lpips_start,
                "scene_pose_normalize": args.scene_pose_normalize,
                "compile": args.compile,
                "actckpt": args.actckpt,
                "ttt_loss_type": args.ttt_loss_type,
                "grad_calc_method": args.grad_calc_method,
                "no_query": args.no_query,
                "use_fused": args.use_fused,
                "dataset_type": args.dataset_type,
                "data_path": args.data_path,
            },
            resume="allow",
        )
        if wandb_id is None:
            wandb_id = wandb.run.id
            with open(wandb_id_path, "w") as f:
                f.write(wandb_id)
        
        wandb.watch(model, log="all", log_freq=args.log_every)
        print(f"Wandb initialized")

    remaining_steps = args.steps - now_iters
    lpips_loss_module = lpips.LPIPS(net="vgg").cuda().eval()
    for epoch in range((remaining_steps - 1) // len(train_loader) + 1):
        for data_dict in train_loader:
            if args.test_every > 0 and (now_iters % args.test_every == 0 or now_iters == 0):
                # Periodic evaluation during training (respects first_n)
                test_dir = f"output/{args.expname}/{now_iters:07d}_test"
                
                # Determine number of views for test (re10k uses fixed 2 input, 3 target)
                test_num_input = 2 if args.dataset_type == "re10k" else args.num_input_views
                test_num_target = 3 if args.dataset_type == "re10k" else args.num_target_views
                
                rank_test_results = run_evaluation(
                    model=model,
                    test_loader=test_loader,
                    lpips_loss_module=lpips_loss_module,
                    args=args,
                    test_dir=test_dir,
                    first_n=args.first_n,
                    save_images=True,
                    ddp_local_rank=ddp_local_rank,
                )

                # Collate metrics from all ranks
                gathered_results = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(gathered_results, rank_test_results)

                # Aggregate and save results at rank 0
                if dist.get_rank() == 0:
                    aggregate_and_save_results(
                        gathered_results=gathered_results,
                        test_dir=test_dir,
                        wandb_prefix="test",
                        now_iters=now_iters,
                        num_input_views=test_num_input,
                        num_target_views=test_num_target,
                    )
            
            data_dict = {key: value.cuda() for key, value in data_dict.items() if isinstance(value, torch.Tensor)}
            input_data_dict = {key: value[:, :args.num_input_views] for key, value in data_dict.items()}
            target_data_dict = {key: value[:, -args.num_target_views:] for key, value in data_dict.items()}
            
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(dtype=torch.bfloat16, device_type="cuda", enabled=True):
                rendering = model(input_data_dict, target_data_dict)
                target = target_data_dict["image"]

                l2_loss = F.mse_loss(rendering, target)
                psnr = -10.0 * torch.log10(l2_loss).item()
                if now_iters >= args.lpips_start:
                    lpips_loss = lpips_loss_module(rendering.flatten(0, 1), target.flatten(0, 1), normalize=True).mean()
                else:
                    lpips_loss = 0.0
                loss = l2_loss + lpips_loss
            loss.backward()

            # Gradident safeguard
            skip_optimizer_step = False
            if now_iters >= 0:
                global_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()

                if not math.isfinite(global_grad_norm):
                    skip_optimizer_step = True
                elif global_grad_norm > 4 * args.grad_clip:
                    skip_optimizer_step = True

            if not skip_optimizer_step:
                optimizer.step()
            lr_scheduler.step()     # Always step the lr scheduler and iters
            now_iters += 1

            if dist.get_rank() == 0:
                if now_iters % args.log_every == 0 or now_iters <= 100:
                    print(f"Iter {now_iters:07d}, Loss: {loss:.4f}, PSNR: {psnr:.2f}, LPIPS: {lpips_loss:.4f}")
                    
                    # Log to wandb
                    log_dict = {
                        "train/psnr": psnr,
                        "train/l2_loss": l2_loss.item(),
                        "train/lpips_loss": lpips_loss.item() if isinstance(lpips_loss, torch.Tensor) else lpips_loss,
                        "train/total_loss": loss.item(),
                        "train/lr": lr_scheduler.get_last_lr()[0],
                        "train/iter": now_iters,
                        "train/epoch": epoch,
                    }
                    if now_iters > 0:
                        log_dict["train/grad_norm"] = global_grad_norm
                        log_dict["train/skip_optimizer_step"] = int(skip_optimizer_step)
                    wandb.log(log_dict, step=now_iters)
                
                if now_iters % args.save_every == 0:
                    torch.save({
                        "model": remove_module_prefix(model.state_dict()),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "now_iters": now_iters,
                        "epoch": epoch,
                    }, f"{output_dir}/model_{now_iters:07d}.pth")
                    print(f"[{now_iters:07d}] Checkpoint saved to {output_dir}/model_{now_iters:07d}.pth")

                    # Clean up old checkpoints
                    # only keep the latest 4 ckpts and dont delete those ckpts that are multiply of 10K
                    checkpoints = [f for f in os.listdir(output_dir) if f.startswith("model_") and f.endswith(".pth")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_")[1].split(".")[0]))
                    for ckpt in checkpoints[:-4]:
                        iter_num = int(ckpt.split("_")[1].split(".")[0])
                        if iter_num % 10000 != 0:
                            os.remove(os.path.join(output_dir, ckpt))
                            print(f"Removed old checkpoint {ckpt}")
            
            if now_iters == args.steps:
                break

    # Final evaluation on ALL test scenes after training is complete
    if args.test_every > 0:
        print(f"[{dist.get_rank():02d}, {ddp_local_rank:02d}] Running final evaluation on ALL test scenes...")
        final_test_dir = f"output/{args.expname}/final_test"
        
        # Determine number of views for test (re10k uses fixed 2 input, 3 target)
        test_num_input = 2 if args.dataset_type == "re10k" else args.num_input_views
        test_num_target = 3 if args.dataset_type == "re10k" else args.num_target_views
        
        # Reset test sampler epoch for fresh iteration
        test_sampler.set_epoch(0)
        
        rank_test_results = run_evaluation(
            model=model,
            test_loader=test_loader,
            lpips_loss_module=lpips_loss_module,
            args=args,
            test_dir=final_test_dir,
            first_n=None,  # Evaluate ALL scenes
            save_images=True,
            ddp_local_rank=ddp_local_rank,
        )

        # Collate metrics from all ranks
        gathered_results = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_results, rank_test_results)

        # Aggregate and save results at rank 0
        if dist.get_rank() == 0:
            aggregate_and_save_results(
                gathered_results=gathered_results,
                test_dir=final_test_dir,
                wandb_prefix="final_test",
                now_iters=now_iters,
                num_input_views=test_num_input,
                num_target_views=test_num_target,
            )

    if dist.get_rank() == 0:
        wandb.finish()
        print("Wandb logging finished.")

if __name__ == "__main__":
    main()