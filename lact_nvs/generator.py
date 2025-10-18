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
            'gpu_count': 2,
            'gpu_type': 'l40s',
        }
        
    def generate_script(self, args):
        # Generate output directory
        if args.expname is not None:
            exp_name = args.expname
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = args.config.replace('config/', '').replace('.yaml', '')
            exp_name = f"{model_name}_{timestamp}"
        output_dir = f"output/{exp_name}"
        script_path = f"{output_dir}/run.sh"
        if args.config in self.model_config:
            config_file = self.model_config[args.config]
        else:
            config_file = args.config

        # Prepare command argument string for training/inference
        if args.inference:
            # inference command line
            cmd = f"python inference.py --config {config_file}"
            if args.load:
                cmd += f" --load {args.load}"
            if args.data_path:
                cmd += f" --data_path {args.data_path}"
            if args.expname:
                cmd += f" --expname {args.expname}"
            if args.num_all_views is not None:
                cmd += f" --num_all_views {args.num_all_views}"
            if args.num_input_views is not None:
                cmd += f" --num_input_views {args.num_input_views}"
            if args.num_target_views is not None:
                cmd += f" --num_target_views {args.num_target_views}"
            if args.scene_inference:
                cmd += " --scene_inference"
            if args.image_size is not None:
                cmd += f" --image_size {args.image_size[0]} {args.image_size[1]}"
            if args.first_n is not None:
                cmd += f" --first_n {args.first_n}"
            run_cmd = f"srun --time {args.time or self.slurm_defaults['time']} uv run {cmd}"
        else:
            # training: torchrun \ --nproc_per_node=2 \ --standalone \ train.py --config ... [other args]
            gpus = args.gpus if args.gpus is not None else 2
            base = f"torchrun --nproc_per_node={gpus} --standalone train.py --config {config_file}"
            # Add remaining training CLI options
            if args.expname:
                base += f" --expname {args.expname}"
            if args.data_path:
                base += f" --data_path {args.data_path}"
            if args.load:
                base += f" --load {args.load}"
            if args.save_every is not None:
                base += f" --save_every {args.save_every}"
            if args.log_every is not None:
                base += f" --log_every {args.log_every}"
            if args.compile:
                base += " --compile"
            if args.actckpt:
                base += " --actckpt"
            if args.bs_per_gpu is not None:
                base += f" --bs_per_gpu {args.bs_per_gpu}"
            if args.num_all_views is not None:
                base += f" --num_all_views {args.num_all_views}"
            if args.num_input_views is not None:
                base += f" --num_input_views {args.num_input_views}"
            if args.num_target_views is not None:
                base += f" --num_target_views {args.num_target_views}"
            if args.image_size is not None:
                base += f" --image_size {args.image_size[0]} {args.image_size[1]}"
            if args.scene_pose_normalize:
                base += " --scene_pose_normalize"
            if args.lr is not None:
                base += f" --lr {args.lr}"
            if args.warmup is not None:
                base += f" --warmup {args.warmup}"
            if args.steps is not None:
                base += f" --steps {args.steps}"
            if args.weight_decay is not None:
                base += f" --weight_decay {args.weight_decay}"
            if args.lpips_start is not None:
                base += f" --lpips_start {args.lpips_start}"
            if args.test_every is not None:
                base += f" --test_every {args.test_every}"
            if args.scene_inference:
                base += " --scene_inference"
            if args.first_n is not None:
                base += f" --first_n {args.first_n}"
            run_cmd = f"srun --time {args.time or self.slurm_defaults['time']} uv run {base}"

        # Generate the script content
        slurm = self.slurm_defaults.copy()
        if args.time is not None:
            slurm['time'] = args.time
        if args.nodes is not None:
            slurm['nodes'] = args.nodes
        if args.gpus is not None:
            slurm['gpu_count'] = args.gpus
        if args.gpu_type is not None:
            slurm['gpu_type'] = args.gpu_type
        if args.memory is not None:
            slurm['mem'] = args.memory

        script_lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={slurm['job_name']}",
            f"#SBATCH --account={slurm['account']}",
            f"#SBATCH --output={output_dir}/%x_%j.out",
            f"#SBATCH --error={output_dir}/%x_%j.err",
            f"#SBATCH --time={slurm['time']}",
            f"#SBATCH --nodes={slurm['nodes']}",
            f"#SBATCH --mem={slurm['mem']}",
            f"#SBATCH --cpus-per-task={slurm['cpus_per_task']}",
            f"#SBATCH --gpus-per-node={slurm['gpu_type']}:{slurm['gpu_count']}",
            "#SBATCH --ntasks-per-node=1",
            "",
            f"# Generated by generator.py on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Config: {config_file}",
            f"# Mode: {'Inference' if args.inference else 'Training'}",
            "",
            "echo \"==============================================\"",
            f"echo \"LaCT {'Inference' if args.inference else 'Training'}\"",
            "echo \"==============================================\"",
            "echo \"Job ID: $SLURM_JOB_ID\"",
            "echo \"Node: $SLURMD_NODENAME\"",
            f"echo \"GPUs: {slurm['gpu_count']}x {slurm['gpu_type']}\"",
            "echo \"Start time: $(date)\"",
            f"echo \"Config: {config_file}\"",
            "echo \"==============================================\"",
            "",
            "# Load modules",
            "module load python/3.12",
            "module load StdEnv/2023 intel/2023.2.1",
            "module load cuda/11.8",
            "",
            "# Environment variables",
            "export OMP_NUM_THREADS={slurm['cpus_per_task']/slurm['gpu_count']}",
            "export IBV_FORK_SAFE=1",
            "",
            "# Suppress libibverbs warnings",
            "exec 3>&2",
            "exec 2> >(grep -v \"libibverbs: Warning\" >&3)",
            "",
            "echo",
            f"echo \"Starting {'inference' if args.inference else 'training'}...\"",
            "echo \"Command line:\"",
            f"echo '{run_cmd}'",
            "echo",
            "",
            "# Run the job",
            run_cmd,
            "",
            "# Restore stderr",
            "exec 2>&3",
            "exec 3>&-",
            "",
            "echo",
            "echo \"==============================================\"",
            f"echo \"{'Inference' if args.inference else 'Training'} completed at: $(date)\"",
            "echo \"==============================================\"",
            "",
            "exit 0"
        ]
        os.makedirs(output_dir, exist_ok=True)
        with open(script_path, 'w') as f:
            f.write('\n'.join(script_lines))
        os.chmod(script_path, 0o755)
        return script_path

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
    parser.add_argument('--expname', type=str,
                        help='Experiment name')
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
    parser.add_argument('--scene-inference', action='store_true',
                        help='Use scene inference mode')
    parser.add_argument('--first-n', type=int,
                        help='First N samples to process')
    
    # Training-only arguments
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
    parser.add_argument('--test-every', type=int,
                        help='Test every N iterations')
    
    # SLURM configuration
    parser.add_argument('--time', type=str,
                        help='Wall time limit (e.g., 01-00:00:00)')
    parser.add_argument('--nodes', type=int,
                        help='Number of nodes')
    parser.add_argument('--gpus', type=int,
                        help='Number of GPUs per node')
    parser.add_argument('--gpu-type', type=str,
                        help='GPU type')
    parser.add_argument('--memory', type=str,
                        help='Memory allocation (e.g., 48GB)')
    
    # Other options
    parser.add_argument('--dry-run', action='store_true',
                        help='Print script without saving')
    parser.add_argument('--submit', action='store_true',
                        help='Submit the job immediately after generation')
    
    args = parser.parse_args()
    generator = Generator()
    script_path = generator.generate_script(args)
    print(f"Generated script: {script_path}")
    if args.dry_run:
        print("\n--- Script Content ---")
        with open(script_path, 'r') as f:
            print(f.read())
        os.remove(script_path)
        try:
            os.rmdir(os.path.dirname(script_path))
        except OSError:
            pass
    if args.submit and not args.dry_run:
        print(f"Submitting job...")
        os.system(f"sbatch {script_path}")
    else:
        if not args.dry_run:
            print(f"To submit: sbatch {script_path}")


if __name__ == '__main__':
    main()
