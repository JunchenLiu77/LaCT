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
    parser.add_argument("--scene_inference", action="store_true")

    # Optimizer
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=4000)
    parser.add_argument("--steps", type=int, default=80000)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--lpips_start", type=int, default=5000, help="Iteration to start LPIPS loss")

    args = parser.parse_args()
    model_config = omegaconf.OmegaConf.load(args.config)
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
    for try_load_path in [args.load]:
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
        except:
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
    train_set = NVSDataset(args.data_path, args.num_all_views, tuple(args.image_size), scene_pose_normalize=args.scene_pose_normalize)
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
        test_set = NVSDataset(args.data_path, args.num_all_views, tuple(args.image_size), sorted_indices=True, scene_pose_normalize=args.scene_pose_normalize)
        test_dataloader_seed_generator = torch.Generator()
        test_dataloader_seed_generator.manual_seed(rank_specific_seed)
        test_loader = DataLoader(
            test_set,
            batch_size=args.bs_per_gpu,
            shuffle=False,
            generator=test_dataloader_seed_generator,    # This ensures deterministic dataloader
        )

    if dist.get_rank() == 0:
        # report training config only on the master
        print(f"Training config:")
        print(model)
        print(optimizer)
        print(lr_scheduler)
        if args.actckpt:
            print(f"Activation checkpointing applied")
        print(f"Starting training from iter {now_iters}...")
        
        # Initialize wandb
        wandb.init(
            project="lact-nvs",
            name=args.expname,
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
            resume="allow" if now_iters > 0 else None,
        )
        wandb.watch(model, log="all", log_freq=args.log_every)
        print(f"Wandb initialized")

    remaining_steps = args.steps - now_iters
    lpips_loss_module = lpips.LPIPS(net="vgg").cuda().eval()
    for epoch in range((remaining_steps - 1) // len(train_loader) + 1):
        for data_dict in train_loader:
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
                global_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()

                if not math.isfinite(global_grad_norm):
                    skip_optimizer_step = True
                elif global_grad_norm > 4.0:
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

            if args.test_every > 0 and now_iters % args.test_every == 0:
                # instantiate a new data iter each time
                test_iter = iter(test_loader)
                output_dir = f"output/{args.expname}/{now_iters:07d}_test"
                os.makedirs(output_dir, exist_ok=True)
                print(f"[{dist.get_rank():02d}, {ddp_local_rank:02d}] Testing from iter {now_iters:07d}...")
                for sample_idx, data_dict in enumerate(test_iter):
                    print(f"[{dist.get_rank():02d}, {ddp_local_rank:02d}] Testing sample {sample_idx:07d}...")
                    if args.first_n is not None and sample_idx >= args.first_n:
                        break
                    scene_names = data_dict["scene_name"]
                    data_dict = {key: value.cuda() for key, value in data_dict.items() if isinstance(value, torch.Tensor)}
                    if args.scene_inference:
                        # Randomly select input views and use remaining as target
                        total_views = data_dict["image"].shape[1]
                        all_indices = torch.randperm(total_views)
                        input_indices = torch.sort(all_indices[:args.num_input_views])[0]   # Sort for video rendering only; model forward is permutation-invariant
                        target_indices = all_indices[-args.num_target_views:]
                        
                        input_data_dict = {key: value[:, input_indices] for key, value in data_dict.items()}
                        target_data_dict = {key: value[:, target_indices] for key, value in data_dict.items()}
                    else:
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
                            scene_dir = os.path.join(output_dir, scene_name)
                            os.makedirs(scene_dir, exist_ok=True)

                            # Compute metrics
                            target = target_data_dict["image"][batch_idx]
                            rendered = rendering[batch_idx]
                            psnr = -10.0 * torch.log10(F.mse_loss(rendered, target)).item()
                            lpips_loss = lpips_loss_module(rendered, target, normalize=True).mean().item()

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

                            # Log the metrics as csv file
                            with open(os.path.join(scene_dir, "metrics.csv"), "a") as f:
                                f.write(f"{sample_idx:06d},{batch_idx:02d},{scene_name},{psnr:.2f},{lpips_loss:.4f}\n")
                        
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
                        # video_path = os.path.join(output_dir, f"sample_{sample_idx:06d}_turntable.gif")
                        # frames = (rendering[0].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
                        # imageio.mimsave(video_path, frames, fps=30, quality=8)
                        # print(f"Saved turntable video to {video_path}")
                
                torch.cuda.empty_cache()
                dist.barrier()

                # Compute average metrics at rank 0
                if dist.get_rank() == 0:
                    scenes = os.listdir(output_dir)
                    psnr_list = []
                    lpips_list = []
                    for scene in scenes:
                        scene_dir = os.path.join(output_dir, scene)
                        metrics_path = os.path.join(scene_dir, "metrics.csv")
                        if os.path.exists(metrics_path):
                            metrics = np.loadtxt(metrics_path, delimiter=",", dtype=str)
                            psnr_list.append(np.array(metrics[3], dtype=np.float32).mean())
                            lpips_list.append(np.array(metrics[4], dtype=np.float32).mean())
                    avg_psnr = np.array(psnr_list, dtype=np.float32).mean()
                    avg_lpips = np.array(lpips_list, dtype=np.float32).mean()
                    print(f"[{now_iters:07d}] Average PSNR = {avg_psnr:.2f}, Average LPIPS = {avg_lpips:.4f}")
                    wandb.log({
                        "test/psnr": avg_psnr,
                        "test/lpips": avg_lpips,
                    }, step=now_iters)
            
            if now_iters == args.steps:
                break

    if dist.get_rank() == 0:
        wandb.finish()
        print("Wandb logging finished.")

if __name__ == "__main__":
    main()