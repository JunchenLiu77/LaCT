"""
Visualize high-dimensional tensor distributions using t-SNE.
Compares 'add' vs 'minus' residuals for each block.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import argparse


def load_tensor(base_path, residual, block_idx, tensor_name):
    """Load a tensor from the saved .pt file."""
    path = os.path.join(base_path, residual, str(block_idx), f"{tensor_name}.pt")
    tensor = torch.load(path, map_location='cpu')
    # Shape: [B, L, D] where B=1, squeeze to [L, D]
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)
    return tensor.numpy()


def normalize_data(data_dict, method='zscore'):
    """
    Normalize tensors to make them comparable.
    
    Args:
        data_dict: Dict of {(residual, tensor_name): data array}
        method: 'zscore' (standardization), 'l2' (unit norm per sample), 
                'per_tensor' (z-score within each tensor type), or 'none'
    
    Returns:
        Normalized data_dict
    """
    if method == 'none':
        return data_dict
    
    if method == 'zscore':
        # Global standardization across all data
        all_data = np.vstack(list(data_dict.values()))
        scaler = StandardScaler()
        scaler.fit(all_data)
        return {k: scaler.transform(v) for k, v in data_dict.items()}
    
    elif method == 'l2':
        # L2 normalize each sample (row) to unit norm
        normalized = {}
        for k, v in data_dict.items():
            norms = np.linalg.norm(v, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
            normalized[k] = v / norms
        return normalized
    
    elif method == 'per_tensor':
        # Normalize each tensor type separately (across both residuals)
        # Group by tensor_name
        tensor_groups = {}
        for (residual, tensor_name), data in data_dict.items():
            if tensor_name not in tensor_groups:
                tensor_groups[tensor_name] = []
            tensor_groups[tensor_name].append(((residual, tensor_name), data))
        
        normalized = {}
        for tensor_name, items in tensor_groups.items():
            # Fit scaler on all data of this tensor type
            all_tensor_data = np.vstack([item[1] for item in items])
            scaler = StandardScaler()
            scaler.fit(all_tensor_data)
            # Transform each
            for key, data in items:
                normalized[key] = scaler.transform(data)
        return normalized
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def visualize_group(base_path, block_idx, tensor_names, group_name, method='tsne', 
                    normalize='per_tensor', save_dir='output/vis_feature/plots'):
    """
    Visualize a group of tensors for both 'add' and 'minus' residuals.
    
    Args:
        base_path: Path to vis_feature directory
        block_idx: Block index to visualize
        tensor_names: List of tensor names (e.g., ['o', 'v', 'vp'] or ['q', 'k'])
        group_name: Name for the group (used in title and filename)
        method: 'tsne' or 'pca'
        normalize: 'zscore', 'l2', 'per_tensor', or 'none'
        save_dir: Directory to save plots
    """
    residuals = ['add', 'minus']
    
    # Color palette - distinguishable colors for each tensor
    colors = {
        'q': '#E63946',   # Red
        'k': '#457B9D',   # Blue
        'o': '#2A9D8F',   # Teal
        'v': '#E9C46A',   # Yellow/Gold
        'vp': '#9B5DE5',  # Purple
    }
    
    # Markers for residual types
    markers = {
        'add': 'o',    # Circle
        'minus': 'x',  # X
    }
    
    # Collect all data
    data_dict = {}
    for residual in residuals:
        for tensor_name in tensor_names:
            data = load_tensor(base_path, residual, block_idx, tensor_name)
            data_dict[(residual, tensor_name)] = data
    
    # Normalize data
    data_dict = normalize_data(data_dict, method=normalize)
    
    # Stack for dimensionality reduction (maintain order)
    all_data = []
    data_info = []
    for residual in residuals:
        for tensor_name in tensor_names:
            data = data_dict[(residual, tensor_name)]
            all_data.append(data)
            data_info.append((residual, tensor_name, data.shape[0]))
    
    all_data = np.vstack(all_data)  # Shape: [total_samples, D]
    
    # Apply dimensionality reduction
    if method == 'tsne':
        # Adjust perplexity based on number of samples
        n_samples = all_data.shape[0]
        perplexity = min(30, max(5, n_samples // 5))
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
        embeddings = reducer.fit_transform(all_data)
    else:  # PCA
        reducer = PCA(n_components=2, random_state=42)
        embeddings = reducer.fit_transform(all_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot each group
    idx = 0
    for residual, tensor_name, n_points in data_info:
        emb = embeddings[idx:idx + n_points]
        idx += n_points
        
        label = f"{tensor_name} ({residual})"
        ax.scatter(
            emb[:, 0], emb[:, 1],
            c=colors[tensor_name],
            marker=markers[residual],
            alpha=0.6,
            s=30,
            label=label,
            edgecolors='white' if markers[residual] == 'o' else None,
            linewidths=0.5 if markers[residual] == 'o' else 1.5
        )
    
    ax.set_xlabel(f'{method.upper()} Dimension 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} Dimension 2', fontsize=12)
    norm_label = f", norm={normalize}" if normalize != 'none' else ""
    ax.set_title(f'Block {block_idx}: {group_name} ({method.upper()}{norm_label})\n○ = add, × = minus', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Save figure
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'block{block_idx}_{group_name}_{method}_{normalize}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def visualize_all_blocks(base_path, method='tsne', normalize='per_tensor', save_dir='output/vis_feature/plots'):
    """Visualize all blocks for both groups."""
    # Find all available block indices
    add_path = os.path.join(base_path, 'add')
    block_indices = sorted([int(d) for d in os.listdir(add_path) if os.path.isdir(os.path.join(add_path, d))])
    
    print(f"Found {len(block_indices)} blocks: {block_indices}")
    print(f"Using {method.upper()} for dimensionality reduction")
    print(f"Normalization: {normalize}")
    print("-" * 50)
    
    for block_idx in block_indices:
        # Group 1: o, v, vp
        visualize_group(base_path, block_idx, ['o', 'v', 'vp'], 'ovvp', method, normalize, save_dir)
        # Group 2: q, k
        visualize_group(base_path, block_idx, ['q', 'k'], 'qk', method, normalize, save_dir)


def create_summary_figure(base_path, method='tsne', normalize='per_tensor', save_dir='output/vis_feature/plots'):
    """Create a summary figure with all blocks in a grid."""
    add_path = os.path.join(base_path, 'add')
    block_indices = sorted([int(d) for d in os.listdir(add_path) if os.path.isdir(os.path.join(add_path, d))])
    
    n_blocks = len(block_indices)
    n_cols = 4
    n_rows = (n_blocks + n_cols - 1) // n_cols
    
    residuals = ['add', 'minus']
    
    # Colors and markers
    colors = {
        'q': '#E63946', 'k': '#457B9D',
        'o': '#2A9D8F', 'v': '#E9C46A', 'vp': '#9B5DE5',
    }
    markers = {'add': 'o', 'minus': 'x'}
    
    # for group_name, tensor_names in [('qk', ['q', 'k']), ('ovvp', ['o', 'v', 'vp'])]:
    for group_name, tensor_names in [('o', ['o']), ('v', ['v']), ('vp', ['vp'])]:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = axes.flatten() if n_blocks > 1 else [axes]
        
        for ax_idx, block_idx in enumerate(block_indices):
            ax = axes[ax_idx]
            
            # Collect data
            data_dict = {}
            for residual in residuals:
                for tensor_name in tensor_names:
                    data = load_tensor(base_path, residual, block_idx, tensor_name)
                    data_dict[(residual, tensor_name)] = data
            
            # Normalize data
            data_dict = normalize_data(data_dict, method=normalize)
            
            # Stack for dimensionality reduction
            all_data = []
            data_info = []
            for residual in residuals:
                for tensor_name in tensor_names:
                    data = data_dict[(residual, tensor_name)]
                    all_data.append(data)
                    data_info.append((residual, tensor_name, data.shape[0]))
            
            all_data = np.vstack(all_data)
            
            # Dimensionality reduction
            if method == 'tsne':
                n_samples = all_data.shape[0]
                perplexity = min(30, max(5, n_samples // 5))
                reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
                embeddings = reducer.fit_transform(all_data)
            else:
                reducer = PCA(n_components=2, random_state=42)
                embeddings = reducer.fit_transform(all_data)
            
            # Plot
            idx = 0
            for residual, tensor_name, n_points in data_info:
                emb = embeddings[idx:idx + n_points]
                idx += n_points
                ax.scatter(
                    emb[:, 0], emb[:, 1],
                    c=colors[tensor_name],
                    marker=markers[residual],
                    alpha=0.6,
                    s=15,
                    edgecolors='white' if markers[residual] == 'o' else None,
                    linewidths=0.3 if markers[residual] == 'o' else 1.0
                )
            
            ax.set_title(f'Block {block_idx}', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(True, alpha=0.3)
        
        # Hide unused axes
        for ax_idx in range(len(block_indices), len(axes)):
            axes[ax_idx].axis('off')
        
        # Create legend
        legend_elements = []
        from matplotlib.lines import Line2D
        for tensor_name in tensor_names:
            for residual in residuals:
                legend_elements.append(
                    Line2D([0], [0], marker=markers[residual], color='w',
                           markerfacecolor=colors[tensor_name], markersize=8,
                           label=f'{tensor_name} ({residual})',
                           markeredgecolor='black' if markers[residual] == 'o' else colors[tensor_name])
                )
        
        fig.legend(handles=legend_elements, loc='upper center', ncol=len(tensor_names) * 2,
                   bbox_to_anchor=(0.5, 1.02), fontsize=10)
        
        norm_label = f", norm={normalize}" if normalize != 'none' else ""
        plt.suptitle(f'{group_name.upper()} Features Across Blocks ({method.upper()}{norm_label})\n○ = add, × = minus',
                     fontsize=14, y=1.06)
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f'summary_{group_name}_{method}_{normalize}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved summary: {save_path}")


if __name__ == '__main__':
    # uv run python vis_feature.py --method pca --normalize none --summary_only
    # uv run python vis_feature.py --method pca --normalize per_tensor --summary_only
    # uv run python vis_feature.py --method pca --normalize l2 --summary_only
    parser = argparse.ArgumentParser(description='Visualize tensor feature distributions')
    parser.add_argument('--base_path', type=str, default='output/vis_feature',
                        help='Base path to vis_feature directory')
    parser.add_argument('--method', type=str, default='tsne', choices=['tsne', 'pca'],
                        help='Dimensionality reduction method (tsne or pca)')
    parser.add_argument('--normalize', type=str, default='per_tensor', 
                        choices=['zscore', 'l2', 'per_tensor', 'none'],
                        help='Normalization method: zscore (global standardization), '
                             'l2 (unit norm per sample), per_tensor (z-score per tensor type), '
                             'none (no normalization)')
    parser.add_argument('--save_dir', type=str, default='output/vis_feature/plots',
                        help='Directory to save plots')
    parser.add_argument('--summary_only', action='store_true',
                        help='Only generate summary figures')
    args = parser.parse_args()
    
    if not args.summary_only:
        visualize_all_blocks(args.base_path, args.method, args.normalize, args.save_dir)
    
    create_summary_figure(args.base_path, args.method, args.normalize, args.save_dir)
    
    print("\n" + "=" * 50)
    print("Visualization complete!")
    print(f"Plots saved to: {args.save_dir}")

