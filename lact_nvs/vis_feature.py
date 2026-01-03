"""
Visualize high-dimensional tensor distributions using PCA.
Creates separate figures for each loss type.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import argparse


def load_tensor(base_path, loss_type, block_idx, tensor_name):
    """Load a tensor from the saved .pt file."""
    path = os.path.join(base_path, loss_type, str(block_idx), f"{tensor_name}.pt")
    tensor = torch.load(path, map_location='cpu')
    print(f"Loaded tensor {tensor_name} from {path} with shape {tensor.shape}")
    # Shape: [B, L, D] -> flatten to [B*L, D] for PCA
    if tensor.dim() == 3:
        B, L, D = tensor.shape
        tensor = tensor.reshape(B * L, D)
        
    return tensor.numpy()


def create_summary_figure(base_path, loss_types, save_dir='output/vis'):
    """Create a summary figure for each loss type separately."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Validate loss types exist
    for lt in loss_types:
        lt_path = os.path.join(base_path, lt)
        if not os.path.isdir(lt_path):
            print(f"Warning: Loss type directory not found: {lt_path}")
    
    print(f"Using loss types: {loss_types}")
    
    # Find all available block indices (from first loss type)
    first_loss_path = os.path.join(base_path, loss_types[0])
    block_indices = sorted([int(d) for d in os.listdir(first_loss_path) 
                           if os.path.isdir(os.path.join(first_loss_path, d))])
    
    n_blocks = len(block_indices)
    n_cols = 4
    n_rows = (n_blocks + n_cols - 1) // n_cols
    
    # Colors for different tensor names
    tensor_colors = {
        'q': '#E63946',   # Red
        'k': '#457B9D',   # Blue
        'o': '#2A9D8F',   # Teal
        'v': '#E9C46A',   # Yellow/Gold
        'vp': '#9B5DE5',  # Purple
    }
    
    # Markers for different tensor names
    tensor_markers = {
        'q': 'o',
        'k': 's',
        'o': '^',
        'v': 'D',
        'vp': 'v',
    }
    
    # Create separate figure for each loss type
    for loss_type in loss_types:
        # for group_name, tensor_names in [('qk', ['q', 'k']), ('ovvp', ['o', 'v', 'vp'])]:
        for group_name, tensor_names in [('qk', ['q', 'k'])]:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
            axes = axes.flatten() if n_blocks > 1 else [axes]
            
            for ax_idx, block_idx in enumerate(block_indices):
                ax = axes[ax_idx]
                
                # Collect data for this loss type only
                all_data = []
                data_info = []
                for tensor_name in tensor_names:
                    try:
                        data = load_tensor(base_path, loss_type, block_idx, tensor_name)
                        all_data.append(data)
                        data_info.append((tensor_name, data.shape[0]))
                    except FileNotFoundError:
                        print(f"Warning: Missing {tensor_name} for {loss_type}/block {block_idx}")
                        continue
                
                if not all_data:
                    ax.set_title(f'Block {block_idx} (no data)', fontsize=10)
                    continue
                    
                all_data = np.vstack(all_data)
                
                # PCA dimensionality reduction
                reducer = PCA(n_components=2, random_state=42)
                embeddings = reducer.fit_transform(all_data)
                
                # Plot
                idx = 0
                for tensor_name, n_points in data_info:
                    emb = embeddings[idx:idx + n_points]
                    idx += n_points
                    ax.scatter(
                        emb[:, 0], emb[:, 1],
                        c=tensor_colors[tensor_name],
                        marker=tensor_markers.get(tensor_name, 'o'),
                        alpha=0.6,
                        s=15,
                        edgecolors='white',
                        linewidths=0.3
                    )
                
                ax.set_title(f'Block {block_idx}', fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(True, alpha=0.3)
            
            # Hide unused axes
            for ax_idx in range(len(block_indices), len(axes)):
                axes[ax_idx].axis('off')
            
            # Create legend
            from matplotlib.lines import Line2D
            legend_elements = []
            for tensor_name in tensor_names:
                legend_elements.append(
                    Line2D([0], [0], marker=tensor_markers.get(tensor_name, 'o'), color='w',
                           markerfacecolor=tensor_colors[tensor_name], markersize=8,
                           label=tensor_name,
                           markeredgecolor='black')
                )
            
            fig.legend(handles=legend_elements, loc='upper center', 
                       ncol=len(tensor_names),
                       bbox_to_anchor=(0.5, 1.02), fontsize=10)
            
            plt.suptitle(f'{loss_type}: {group_name.upper()} Features Across Blocks (PCA)', fontsize=14, y=1.06)
            plt.tight_layout()
            
            save_path = os.path.join(save_dir, f'summary_{group_name}_{loss_type}_pca.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved summary: {save_path}")


if __name__ == '__main__':
    # Usage: uv run python vis_feature.py --loss_types design1 design2 dot_product
    parser = argparse.ArgumentParser(description='Visualize tensor feature distributions using PCA')
    parser.add_argument('--base_path', type=str, default='output/vis',
                        help='Base path to vis directory containing loss_type subdirs')
    parser.add_argument('--loss_types', type=str, nargs='+', required=True,
                        help='Loss types to visualize (e.g., design1 design2)')
    parser.add_argument('--save_dir', type=str, default='output/vis',
                        help='Directory to save plots')
    args = parser.parse_args()
    
    create_summary_figure(args.base_path, args.loss_types, args.save_dir)
    
    print("\n" + "=" * 50)
    print("Visualization complete!")
    print(f"Plots saved to: {args.save_dir}")
