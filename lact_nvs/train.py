import argparse
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

from data import NVSDataset
from model import LaCTLVSM
from inference import get_turntable_cameras_with_zoom_in, get_interpolated_cameras
from PIL import Image
import imageio
import json

def main():
    parser = argparse.ArgumentParser()
    # Basic info
    parser.add_argument("--config", type=str, default="config/lact")
    parser.add_argument("--expname", type=str, default="default")
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--data_path", type=str, default="data_example/gso_sample_data_path.json")
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=100)

    # Training
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--actckpt", action="store_true")
    parser.add_argument("--bs_per_gpu", type=int, default=8)
    
    parser.add_argument("--num_all_views", type=int, default=15)
    parser.add_argument("--num_input_views", type=int, default=8)
    parser.add_argument("--num_target_views", type=int, default=8)  
    parser.add_argument("--image_size", nargs=2, type=int, default=[256, 256], help="Image size H, W")
    parser.add_argument("--scene_pose_normalize", action="store_true")

    # Inference
    parser.add_argument("--test_every", type=int, default=-1, help="Test every N iterations")
    parser.add_argument("--first_n", type=int, default=None)
    parser.add_argument("--test_bs_per_gpu", type=int, default=1)
    parser.add_argument("--scene_inference", action="store_true")

    # Optimizer
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_opts", type=float, default=None)
    parser.add_argument("--warmup", type=int, default=4000)
    parser.add_argument("--steps", type=int, default=80000)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--lpips_start", type=int, default=5000, help="Iteration to start LPIPS loss")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")

    # Model
    parser.add_argument("--use_learnable_opt", action="store_true")
    parser.add_argument("--opt_type", type=str, default="", help="Type of optimizer")
    parser.add_argument("--residual", type=str, default="add", choices=["none", "add", "minus"])
    parser.add_argument("--normalize_weight", action="store_true")
    parser.add_argument("--no_normalize_weight", action="store_true")
    parser.add_argument("--opt_hidden_dim", type=int, default=256)
    parser.add_argument("--n_blocks_per_opt", type=int, default=2)
    parser.add_argument("--only_train_opts", action="store_true")
    parser.add_argument("--use_shared_opts", action="store_true")
    parser.add_argument("--output_norm_method", type=str, default="none", choices=["none", "mean_std", "affine"])

    args = parser.parse_args()
    model_config = omegaconf.OmegaConf.load(args.config)
    
    # Overwrite the use_learnable_opt param in model_config if specified in args
    if hasattr(args, "use_learnable_opt") and args.use_learnable_opt:
        model_config.block_config[1]["params"]["use_learnable_opt"] = args.use_learnable_opt
    if hasattr(args, "n_blocks_per_opt") and args.n_blocks_per_opt:
        model_config.block_config[1]["params"]["n_blocks_per_opt"] = args.n_blocks_per_opt
    if hasattr(args, "use_shared_opts") and args.use_shared_opts:
        model_config.use_shared_opts = args.use_shared_opts
    if hasattr(args, "opt_type") and args.opt_type:
        model_config.block_config[1]["params"]["opt_type"] = args.opt_type
    if hasattr(args, "residual") and args.residual:
        model_config.block_config[1]["params"]["residual"] = args.residual
    if hasattr(args, "normalize_weight") and args.normalize_weight:
        assert not (hasattr(args, "no_normalize_weight") and args.no_normalize_weight), "normalize_weight and no-normalize_weight cannot be set at the same time"
        model_config.block_config[1]["params"]["normalize_weight"] = args.normalize_weight
    if hasattr(args, "no_normalize_weight") and args.no_normalize_weight:
        assert not (hasattr(args, "normalize_weight") and args.normalize_weight), "normalize_weight and no-normalize_weight cannot be set at the same time"
        model_config.block_config[1]["params"]["normalize_weight"] = False
    if hasattr(args, "opt_hidden_dim") and args.opt_hidden_dim:
        model_config.block_config[1]["params"]["opt_hidden_dim"] = args.opt_hidden_dim
    if hasattr(args, "output_norm_method") and args.output_norm_method:
        model_config.block_config[1]["params"]["output_norm_method"] = args.output_norm_method
    output_dir = f"output/{args.expname}"
    os.makedirs(output_dir, exist_ok=True)

    dist.init_process_group(backend="nccl")
    ddp_local_rank = int(os.environ.get("LOCAL_RANK", dist.get_rank() % 8))
    torch.cuda.set_device(ddp_local_rank)
    print(f"[{dist.get_rank():02d}, {ddp_local_rank:02d}] device: {torch.cuda.current_device()}")

    # Seed everything
    rank_specific_seed = 95 + dist.get_rank()
    print(f"[{dist.get_rank():02d}, {ddp_local_rank:02d}] seed: {rank_specific_seed}")
    torch.manual_seed(rank_specific_seed)
    np.random.seed(rank_specific_seed)
    random.seed(rank_specific_seed)
    dataloader_seed_generator = torch.Generator()
    dataloader_seed_generator.manual_seed(rank_specific_seed)

    model = LaCTLVSM(**model_config).cuda()

    # Config learnable params
    opts_params = []
    other_params = []
    learnable_param_names = []
    frozen_param_names = []
    learnable_params_count = 0
    frozen_params_count = 0
    
    for n, p in model.named_parameters():
        if not args.only_train_opts or ("opts" in n):
            if "opts" in n:
                opts_params.append(p)
            else:
                other_params.append(p)
            learnable_param_names.append(n)
            learnable_params_count += p.numel()
            p.requires_grad = True
        else:
            p.requires_grad = False
            frozen_param_names.append(n)
            frozen_params_count += p.numel()
    
    # Config optimizer
    optim_groups = []
    
    # Regular parameters
    if other_params:
        decay_params = [p for p in other_params if p.dim() >= 2]
        nodecay_params = [p for p in other_params if p.dim() < 2]
        optim_groups.append({"params": decay_params, "weight_decay": args.weight_decay, "lr": args.lr})
        optim_groups.append({"params": nodecay_params, "weight_decay": 0.0, "lr": args.lr})

    # Opts parameters
    if opts_params:
        lr_opts = args.lr_opts if args.lr_opts is not None else args.lr
        decay_params = [p for p in opts_params if p.dim() >= 2]
        nodecay_params = [p for p in opts_params if p.dim() < 2]
        optim_groups.append({"params": decay_params, "weight_decay": args.weight_decay, "lr": lr_opts})
        optim_groups.append({"params": nodecay_params, "weight_decay": 0.0, "lr": lr_opts})

    optimizer = torch.optim.AdamW(optim_groups, betas=(0.9, 0.95), fused=True)
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

    # Data
    train_set = NVSDataset("data_example/dl3dv_10k_sample_data_path.json", args.num_all_views, tuple(args.image_size), scene_pose_normalize=args.scene_pose_normalize)
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
    )

    if args.test_every > 0:
        # Load fixed test indices
        num_total_needed = args.num_input_views + args.num_target_views
        fixed_indices_filename = f"test_indices_in{args.num_input_views}_tar{args.num_target_views}.json"
        fixed_indices_path = os.path.join("data_example", fixed_indices_filename)
        
        assert os.path.exists(fixed_indices_path), (
            f"Fixed test indices file not found at {fixed_indices_path}. "
            f"Please run `python lact_nvs/generate_indices.py --data_path {args.data_path} "
            f"--num_input_views {args.num_input_views} --num_target_views {args.num_target_views}` first."
        )
        
        if dist.get_rank() == 0:
            print(f"Loading fixed test indices from {fixed_indices_path}")
            
        with open(fixed_indices_path, "r") as f:
            test_indices_map = json.load(f)

        # Use num_total_needed as num_views to ensure we get exactly what we put in the map
        num_views_for_dataset = args.num_input_views + args.num_target_views

        test_set = NVSDataset(
            "data_example/dl3dv_benchmark_sample_data_path.json", 
            num_views_for_dataset, 
            tuple(args.image_size), 
            sorted_indices=False,  # Important: false to preserve Input/Target block ordering
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
        )
        test_sampler.set_epoch(0)

    if dist.get_rank() == 0:
        # report training config only on the master
        print(f"Training config:")
        # print(model)
        # print(optimizer)
        # print(lr_scheduler)
        print(f"Learnable params: {learnable_param_names}, {learnable_params_count / 1e6:.2f} MB in total")
        print(f"Frozen params: {frozen_param_names}, {frozen_params_count / 1e6:.2f} MB in total")

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
            project="lact-nvs",
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
                # instantiate a new data iter each time
                test_iter = iter(test_loader)
                test_dir = f"output/{args.expname}/{now_iters:07d}_test"
                os.makedirs(test_dir, exist_ok=True)
                print(f"[{dist.get_rank():02d}, {ddp_local_rank:02d}] Testing from iter {now_iters:07d}...")

                # Collect all test results for this rank
                rank_test_results = []

                for sample_idx, data_dict in enumerate(test_iter):
                    print(f"[{dist.get_rank():02d}, {ddp_local_rank:02d}] Testing sample {sample_idx:07d}...")
                    if args.first_n is not None and sample_idx >= args.first_n:
                        break
                    indices = data_dict["indices"]
                    scene_names = data_dict["scene_name"]
                    data_dict = {key: value.cuda() for key, value in data_dict.items() if isinstance(value, torch.Tensor)}
                    input_data_dict = {key: value[:, :args.num_input_views] for key, value in data_dict.items()}
                    target_data_dict = {key: value[:, -args.num_target_views:] for key, value in data_dict.items()}

                    with torch.autocast(dtype=torch.bfloat16, device_type="cuda", enabled=True) and torch.no_grad():
                        rendering = model(input_data_dict, target_data_dict)

                        # target = target_data_dict["image"]
                        # psnr = -10.0 * torch.log10(F.mse_loss(rendering, target)).item()
                        # lpips_loss = lpips_loss_module(rendering.flatten(0, 1), target.flatten(0, 1), normalize=True).mean().item()
                        # test_psnr_list.append(psnr)
                        # test_lpips_list.append(lpips_loss)
                        # print(f"Sample {sample_idx}: PSNR = {psnr:.2f}, LPIPS = {lpips_loss:.4f}")
                        
                        # Save rendered images
                        def tensor_to_numpy(tensor):
                            """Convert tensor to numpy RGB image."""
                            numpy_image = tensor.permute(1, 2, 0).cpu().numpy()
                            numpy_image = np.clip(numpy_image * 255, 0, 255).astype(np.uint8)
                            return numpy_image

                        batch_size, num_views = rendering.shape[:2]
                        for batch_idx in range(batch_size):
                            scene_name = scene_names[batch_idx]
                            scene_dir = os.path.join(test_dir, scene_name)
                            os.makedirs(scene_dir, exist_ok=True)

                            # Compute metrics
                            target = target_data_dict["image"][batch_idx]
                            rendered = rendering[batch_idx]
                            
                            # Calculate PSNR per view and then average (Mean of Metrics), 
                            # instead of PSNR of average MSE (Metric of Mean Error).
                            # This aligns with LPIPS calculation and standard evaluation protocols.
                            mse_per_view = F.mse_loss(rendered, target, reduction='none').mean(dim=[1, 2, 3])
                            psnr_per_view = -10.0 * torch.log10(mse_per_view)
                            psnr = psnr_per_view.mean().item()

                            lpips_loss = lpips_loss_module(rendered, target, normalize=True).mean().item()

                            # Determine indices used for input and target
                            sample_indices = indices[batch_idx].cpu().tolist()
                            total_views_count = len(sample_indices)
                            input_indices_local = list(range(args.num_input_views))
                            target_indices_local = list(range(total_views_count - args.num_target_views, total_views_count))

                            # Map local indices (0..N-1) to original dataset indices
                            # sample_indices contains the original indices corresponding to positions 0..N-1
                            # Sort them for the output JSON as requested
                            input_indices_original = sorted([sample_indices[i] for i in input_indices_local])
                            target_indices_original = sorted([sample_indices[i] for i in target_indices_local])

                            # Store metrics
                            rank_test_results.append({
                                "scene_name": scene_name,
                                "psnr": psnr,
                                "lpips": lpips_loss,
                                "input_indices": input_indices_original,
                                "target_indices": target_indices_original
                            })

                            # Collect all images for this batch
                            rendered_images = []
                            target_images = []
                            for view_idx in range(num_views):
                                rendered_images.append(tensor_to_numpy(rendered[view_idx]))
                                target_images.append(tensor_to_numpy(target[view_idx]))
                            
                            # Concatenate images horizontally (all views side by side)
                            target_row = np.concatenate(target_images, axis=1)
                            rendered_row = np.concatenate(rendered_images, axis=1)
                            
                            # Stack rendered and target rows vertically
                            combined_image = np.concatenate([target_row, rendered_row], axis=0)
                            
                            # Save the concatenated image
                            Image.fromarray(combined_image).save(os.path.join(scene_dir, f"sample_{sample_idx:06d}_batch_{batch_idx:02d}.png"))

                        
                        # Rendering a video to circularly rotate the camera views
                        # if args.scene_inference:
                        #     target_cameras = get_interpolated_cameras(
                        #         cameras=input_data_dict,
                        #         num_views=2,
                        #     )
                        # else:
                        #     target_cameras = get_turntable_cameras_with_zoom_in(
                        #         batch_size=1,
                        #         num_views=120,
                        #         w=args.image_size[0],
                        #         h=args.image_size[1],
                        #         min_radius=1.7,
                        #         max_radius=3.0,
                        #         elevation=30,
                        #         up_vector=np.array([0, 0, 1]),
                        #         device=torch.device("cuda"),
                        #     )
                        # # print(target_cameras["c2w"].shape, target_cameras["fxfycxcy"].shape)
                        # states = model.module.reconstruct(input_data_dict)
                        # rendering = model.module.rendering(target_cameras, states, args.image_size[0], args.image_size[1])
                        # video_path = os.path.join(test_dir, f"sample_{sample_idx:06d}_turntable.gif")
                        # frames = (rendering[0].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
                        # imageio.mimsave(video_path, frames, fps=30, quality=8)
                        # print(f"Saved turntable video to {video_path}")
                
                torch.cuda.empty_cache()
                dist.barrier()

                # Collate metrics from all ranks and dump one json only from the main rank
                # Use dist.all_gather_object to collect results from all ranks
                gathered_results = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(gathered_results, rank_test_results)

                # Compute average metrics at rank 0
                if dist.get_rank() == 0:
                    # Aggregate all results
                    all_results = []
                    psnr_list = []
                    lpips_list = []
                    
                    for rank_results in gathered_results:
                        all_results.extend(rank_results)
                        for res in rank_results:
                            psnr_list.append(res["psnr"])
                            lpips_list.append(res["lpips"])

                    avg_psnr = np.array(psnr_list, dtype=np.float32).mean()
                    avg_lpips = np.array(lpips_list, dtype=np.float32).mean()
                    
                    # Save aggregated results to metrics.json
                    final_metrics = {
                        "average_psnr": float(avg_psnr),
                        "average_lpips": float(avg_lpips),
                        "scene_results": all_results
                    }
                    
                    with open(os.path.join(test_dir, "metrics.json"), "w") as f:
                        json.dump(final_metrics, f, indent=4)
                    
                    # log in wandb
                    print(f"[{now_iters:07d}] Average PSNR = {avg_psnr:.2f}, Average LPIPS = {avg_lpips:.4f}")
                    wandb.log({
                        "test/psnr": avg_psnr,
                        "test/lpips": avg_lpips,
                    }, step=now_iters)
            
            exit()
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
            if now_iters > 1000:
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
                    if now_iters > 1000:
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

    if dist.get_rank() == 0:
        wandb.finish()
        print("Wandb logging finished.")

if __name__ == "__main__":
    main()