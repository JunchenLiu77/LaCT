#!/usr/bin/env python3
"""
LaCT SLURM Script Generator

This script generates customized SLURM job scripts for training and inference
with flexible configuration options.

Usage:
    python generator.py [options]
    
Examples:
    # Basic training
    python generator.py --config lact_l14_d768_ttt2x --steps 80000
    
    # Training with custom settings
    python generator.py --config lact_l24_d768_ttt2x --bs-per-gpu 4 --lr 1e-4
    
    # Inference mode
    python generator.py --inference --load weight/obj_res256.pt --data-path data_example/gso_sample_data_path.json
"""

import argparse
import os
from datetime import datetime


class Generator:
    def __init__(self):
        self.model_config = {
            'lact_l14_d768_ttt2x': 'config/lact_l14_d768_ttt2x.yaml',
            'lact_l24_d768_ttt2x': 'config/lact_l24_d768_ttt2x.yaml',
            'lact_l24_d768_ttt4x': 'config/lact_l24_d768_ttt4x.yaml',
        }
        
        # Default SLURM settings
        self.slurm_defaults = {
            'job_name': 'lact',
            'account': 'aip-fsanja',
            'time': '01-00:00:00',
            'nodes': 1,
            'mem': '48GB',
            'cpus_per_task': 8,
            'gpus_per_node': 'l40s:2',
        }
        
    def generate_script(self, args):
        """Generate the complete SLURM script"""
        
        # Generate output directory
        if args.expname is not None:
            exp_name = args.expname
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = args.config.replace('config/', '').replace('.yaml', '')
            exp_name = f"{model_name}_{timestamp}"
        
        if args.inference:
            output_dir = args.output_dir if args.output_dir else f"output/{exp_name}"
        else:
            output_dir = f"output/{exp_name}"
        
        script_path = f"{output_dir}/run.sh"
        
        # Determine config file
        if args.config in self.model_config:
            config_file = self.model_config[args.config]
        else:
            config_file = args.config
        
        # Build command line arguments
        cmd_args = self._build_cmd_args(args, config_file, output_dir)
        
        # Generate the script content
        script_content = self._generate_slurm_script(
            args, config_file, cmd_args, output_dir
        )
        
        # Save the script
        os.makedirs(output_dir, exist_ok=True)
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make it executable
        os.chmod(script_path, 0o755)
        
        return script_path
    
    def _build_cmd_args(self, args, config_file, output_dir):
        """Build command line arguments"""
        cmd_args = []
        
        # Config file
        cmd_args.append(f'--config {config_file}')
        
        if args.inference:
            # Inference arguments
            if args.load:
                cmd_args.append(f'--load {args.load}')
            if args.data_path:
                cmd_args.append(f'--data_path {args.data_path}')
            if args.output_dir or output_dir:
                cmd_args.append(f'--output_dir {output_dir}')
            if args.num_all_views is not None:
                cmd_args.append(f'--num_all_views {args.num_all_views}')
            if args.num_input_views is not None:
                cmd_args.append(f'--num_input_views {args.num_input_views}')
            if args.num_target_views is not None:
                cmd_args.append(f'--num_target_views {args.num_target_views}')
            if args.scene_inference:
                cmd_args.append('--scene_inference')
            if args.image_size is not None:
                cmd_args.append(f'--image_size {args.image_size[0]} {args.image_size[1]}')
            if args.first_n is not None:
                cmd_args.append(f'--first_n {args.first_n}')
        else:
            # Training arguments
            cmd_args.append(f'--expname {args.expname if args.expname else output_dir.split("/")[-1]}')
            
            if args.data_path:
                cmd_args.append(f'--data_path {args.data_path}')
            if args.load:
                cmd_args.append(f'--load {args.load}')
            if args.save_every is not None:
                cmd_args.append(f'--save_every {args.save_every}')
            if args.log_every is not None:
                cmd_args.append(f'--log_every {args.log_every}')
            if args.compile:
                cmd_args.append('--compile')
            if args.actckpt:
                cmd_args.append('--actckpt')
            if args.bs_per_gpu is not None:
                cmd_args.append(f'--bs_per_gpu {args.bs_per_gpu}')
            if args.num_all_views is not None:
                cmd_args.append(f'--num_all_views {args.num_all_views}')
            if args.num_input_views is not None:
                cmd_args.append(f'--num_input_views {args.num_input_views}')
            if args.num_target_views is not None:
                cmd_args.append(f'--num_target_views {args.num_target_views}')
            if args.image_size is not None:
                cmd_args.append(f'--image_size {args.image_size[0]} {args.image_size[1]}')
            if args.scene_pose_normalize:
                cmd_args.append('--scene_pose_normalize')
            if args.lr is not None:
                cmd_args.append(f'--lr {args.lr}')
            if args.warmup is not None:
                cmd_args.append(f'--warmup {args.warmup}')
            if args.steps is not None:
                cmd_args.append(f'--steps {args.steps}')
            if args.weight_decay is not None:
                cmd_args.append(f'--weight_decay {args.weight_decay}')
            if args.lpips_start is not None:
                cmd_args.append(f'--lpips_start {args.lpips_start}')
        
        return cmd_args
    
    def _generate_slurm_script(self, args, config_file, cmd_args, output_dir):
        """Generate the complete SLURM script content"""
        
        # Update SLURM settings based on arguments
        slurm = self.slurm_defaults.copy()
        if args.time is not None:
            slurm['time'] = args.time
        if args.nodes is not None:
            slurm['nodes'] = args.nodes
        if args.gpus is not None:
            slurm['gpus_per_node'] = f'l40s:{args.gpus}'
        if args.memory is not None:
            slurm['mem'] = args.memory
        
        # Determine which Python script to use
        python_script = 'inference.py' if args.inference else 'train.py'
        
        # Build the command arguments string
        cmd_args_str = ' \\\n    '.join(cmd_args) if cmd_args else ''
        if cmd_args_str:
            cmd_args_str = ' \\\n    ' + cmd_args_str
        
        # Generate the script
        script_parts = []
        script_parts.append('#!/bin/bash')
        script_parts.append(f'#SBATCH --job-name={slurm["job_name"]}')
        script_parts.append(f'#SBATCH --account={slurm["account"]}')
        script_parts.append(f'#SBATCH --output={output_dir}/%x_%j.out')
        script_parts.append(f'#SBATCH --error={output_dir}/%x_%j.err')
        script_parts.append(f'#SBATCH --time={slurm["time"]}')
        script_parts.append(f'#SBATCH --nodes={slurm["nodes"]}')
        script_parts.append(f'#SBATCH --mem={slurm["mem"]}')
        script_parts.append(f'#SBATCH --cpus-per-task={slurm["cpus_per_task"]}')
        script_parts.append(f'#SBATCH --gpus-per-node={slurm["gpus_per_node"]}')
        script_parts.append('#SBATCH --ntasks-per-node=1')
        script_parts.append('')
        
        script_parts.append(f'# Generated by generator.py on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        script_parts.append(f'# Config: {config_file}')
        script_parts.append(f'# Mode: {"Inference" if args.inference else "Training"}')
        script_parts.append('')
        
        script_parts.append('echo "=============================================="')
        script_parts.append(f'echo "LaCT {"Inference" if args.inference else "Training"}"')
        script_parts.append('echo "=============================================="')
        script_parts.append('echo "Job ID: $SLURM_JOB_ID"')
        script_parts.append('echo "Node: $SLURMD_NODENAME"')
        script_parts.append(f'echo "GPUs: {args.gpus or 2}x L40s"')
        script_parts.append('echo "Start time: $(date)"')
        script_parts.append(f'echo "Config: {config_file}"')
        script_parts.append('echo "=============================================="')
        script_parts.append('')
        
        script_parts.append('# Load modules')
        script_parts.append('module load python/3.12')
        script_parts.append('module load StdEnv/2023 intel/2023.2.1')
        script_parts.append('module load cuda/11.8')
        script_parts.append('')
        
        script_parts.append('# Environment variables')
        script_parts.append('export OMP_NUM_THREADS=4')
        script_parts.append('export IBV_FORK_SAFE=1')
        script_parts.append('export MASTER_ADDR=localhost')
        script_parts.append('export MASTER_PORT=$(shuf -i 20000-65000 -n 1)')
        script_parts.append('')
        
        script_parts.append('# Optimized NCCL settings')
        script_parts.append('# export NCCL_IB_DISABLE=1  # Uncomment to disable InfiniBand')
        script_parts.append('# export NCCL_P2P_DISABLE=1  # Uncomment to disable P2P')
        script_parts.append('')
        
        script_parts.append('# Debug settings (optional)')
        script_parts.append('# export NCCL_DEBUG=INFO')
        script_parts.append('# export CUDA_LAUNCH_BLOCKING=1')
        script_parts.append('')
        
        script_parts.append('# Suppress libibverbs warnings')
        script_parts.append('exec 3>&2')
        script_parts.append('exec 2> >(grep -v "libibverbs: Warning" >&3)')
        script_parts.append('')
        
        script_parts.append('echo')
        script_parts.append(f'echo "Starting {"inference" if args.inference else "training"}..."')
        script_parts.append('echo "Command arguments:"')
        
        # Add command arguments display
        for arg in cmd_args:
            script_parts.append(f'echo "  {arg}"')
        
        script_parts.append('echo')
        script_parts.append('')
        script_parts.append(f'# Run the {"inference" if args.inference else "training"}')
        
        if args.inference:
            # Inference doesn't use torchrun
            script_parts.append(f'srun --time {slurm["time"]} uv run python {python_script}{cmd_args_str}')
        else:
            # Training uses torchrun
            script_parts.append(f'srun --time {slurm["time"]} uv run torchrun \\')
            script_parts.append(f'    --nproc_per_node={args.gpus or 2} \\')
            script_parts.append(f'    --master_addr=$MASTER_ADDR \\')
            script_parts.append(f'    --master_port=$MASTER_PORT \\')
            script_parts.append(f'    {python_script}{cmd_args_str}')
        
        script_parts.append('')
        script_parts.append('# Restore stderr')
        script_parts.append('exec 2>&3')
        script_parts.append('exec 3>&-')
        script_parts.append('')
        script_parts.append('echo')
        script_parts.append('echo "=============================================="')
        script_parts.append(f'echo "{"Inference" if args.inference else "Training"} completed at: $(date)"')
        script_parts.append('echo "=============================================="')
        script_parts.append('')
        script_parts.append('exit 0')
        
        return '\n'.join(script_parts)


def main():
    parser = argparse.ArgumentParser(
        description='Generate SLURM scripts for LaCT training and inference',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Model/Config selection
    parser.add_argument('--config', 
                        choices=['lact_l14_d768_ttt2x', 'lact_l24_d768_ttt2x', 'lact_l24_d768_ttt4x'],
                        default='lact_l24_d768_ttt2x',
                        help='Model configuration to use')
    
    # Mode selection
    parser.add_argument('--inference', action='store_true',
                        help='Generate inference script instead of training')
    
    # Common arguments (both training and inference)
    parser.add_argument('--load', type=str,
                        help='Checkpoint path to load')
    parser.add_argument('--data-path', type=str,
                        help='Path to dataset JSON file')
    parser.add_argument('--num-all-views', type=int,
                        help='Total number of views')
    parser.add_argument('--num-input-views', type=int,
                        help='Number of input views')
    parser.add_argument('--num-target-views', type=int,
                        help='Number of target views')
    parser.add_argument('--image-size', nargs=2, type=int,
                        help='Image size [H W]')
    
    # Training-only arguments
    parser.add_argument('--expname', type=str,
                        help='Experiment name (training only)')
    parser.add_argument('--save-every', type=int,
                        help='Save checkpoint every N iterations')
    parser.add_argument('--log-every', type=int,
                        help='Log every N iterations')
    parser.add_argument('--compile', action='store_true',
                        help='Enable torch.compile')
    parser.add_argument('--actckpt', action='store_true',
                        help='Enable activation checkpointing')
    parser.add_argument('--bs-per-gpu', type=int,
                        help='Batch size per GPU')
    parser.add_argument('--scene-pose-normalize', action='store_true',
                        help='Normalize scene poses')
    parser.add_argument('--lr', type=float,
                        help='Learning rate')
    parser.add_argument('--warmup', type=int,
                        help='Warmup steps')
    parser.add_argument('--steps', type=int,
                        help='Total training steps')
    parser.add_argument('--weight-decay', type=float,
                        help='Weight decay')
    parser.add_argument('--lpips-start', type=int,
                        help='Iteration to start LPIPS loss')
    
    # Inference-only arguments
    parser.add_argument('--output-dir', type=str,
                        help='Output directory for inference results')
    parser.add_argument('--scene-inference', action='store_true',
                        help='Use scene inference mode')
    parser.add_argument('--first-n', type=int,
                        help='First N samples to process')
    
    # SLURM configuration
    parser.add_argument('--time', type=str,
                        help='Wall time limit (e.g., 01-00:00:00)')
    parser.add_argument('--nodes', type=int,
                        help='Number of nodes')
    parser.add_argument('--gpus', type=int,
                        help='Number of GPUs per node')
    parser.add_argument('--memory', type=str,
                        help='Memory allocation (e.g., 48GB)')
    
    # Other options
    parser.add_argument('--dry-run', action='store_true',
                        help='Print script without saving')
    parser.add_argument('--submit', action='store_true',
                        help='Submit the job immediately after generation')
    
    args = parser.parse_args()
    
    # Generate the script
    generator = Generator()
    script_path = generator.generate_script(args)
    
    print(f"Generated script: {script_path}")
    
    if args.dry_run:
        print("\n--- Script Content ---")
        with open(script_path, 'r') as f:
            print(f.read())
        os.remove(script_path)  # Clean up in dry-run mode
        try:
            os.rmdir(os.path.dirname(script_path))  # Remove empty dir
        except OSError:
            pass  # Directory not empty, that's fine
    
    if args.submit and not args.dry_run:
        print(f"Submitting job...")
        os.system(f"sbatch {script_path}")
    else:
        if not args.dry_run:
            print(f"To submit: sbatch {script_path}")


if __name__ == '__main__':
    main()
