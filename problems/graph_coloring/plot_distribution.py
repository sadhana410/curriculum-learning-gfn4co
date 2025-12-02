# problems/graph_coloring/plot_distribution.py
"""
Plot sample distribution from a trained GFlowNet checkpoint.

This script samples from a trained model and visualizes the distribution
of solution qualities (number of colors used), demonstrating that GFlowNets
sample proportionally to reward.
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from problems.graph_coloring.env import GraphColoringEnv
from problems.graph_coloring.policy import ConditionalGNNPolicy, ConditionalGNNPolicyWrapper
from problems.graph_coloring.utils import load_col_file

# Data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

# Chromatic numbers
CHROMATIC_NUMBERS = {
    "myciel2.col": 3,
    "myciel3.col": 4,
    "myciel4.col": 5,
    "myciel5.col": 6,
    "myciel6.col": 7,
    "myciel7.col": 8,
}


def load_checkpoint(checkpoint_path, device):
    """Load checkpoint and create policy."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    forward_state = checkpoint['forward_policy']
    
    # Check key format
    sample_key = list(forward_state.keys())[0]
    has_prefix = sample_key.startswith('shared_policy.')
    
    if has_prefix:
        emb_key = 'shared_policy.color_embedding.weight'
        gnn_prefix = 'shared_policy.gnn_layers.'
    else:
        emb_key = 'color_embedding.weight'
        gnn_prefix = 'gnn_layers.'
    
    emb_weight = forward_state[emb_key]
    num_colors = emb_weight.shape[0] - 1
    hidden_dim = emb_weight.shape[1]
    
    num_layers = 0
    for key in forward_state.keys():
        if gnn_prefix in key:
            after_prefix = key.split(gnn_prefix)[1]
            layer_idx = int(after_prefix.split('.')[0])
            num_layers = max(num_layers, layer_idx + 1)
    
    shared_policy = ConditionalGNNPolicy(
        num_colors=num_colors,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    ).to(device)
    
    if has_prefix:
        forward = ConditionalGNNPolicyWrapper(shared_policy, mode='forward')
        forward.load_state_dict(forward_state)
        forward.eval()
        return shared_policy, num_colors, checkpoint.get('step', '?')
    else:
        shared_policy.load_state_dict(forward_state)
        shared_policy.eval()
        return shared_policy, num_colors, checkpoint.get('step', '?')


def count_conflicts(adj, state):
    """Count edge conflicts."""
    adj_np = adj if isinstance(adj, np.ndarray) else adj.cpu().numpy()
    colored_mask = state != -1
    same_color = (state[:, None] == state[None, :]) & colored_mask[:, None] & colored_mask[None, :]
    return int(np.sum(adj_np * same_color) // 2)


def sample_solutions(policy, adj, num_colors, device, num_samples=1000):
    """Sample solutions and return distribution."""
    N = adj.shape[0]
    instance = {'adj': adj}
    env = GraphColoringEnv(instance, num_colors=num_colors)
    
    color_counts = defaultdict(int)  # colors -> count
    all_colors = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            state = env.reset()
            
            while not env.is_terminal(state):
                logits = policy(state, adj, device=device)
                mask = torch.tensor(env.allowed_actions(state), dtype=torch.float32, device=device)
                
                if mask.sum() == 0:
                    break
                
                masked_logits = logits.clone()
                masked_logits[mask == 0] = float('-inf')
                probs = torch.softmax(masked_logits, dim=0)
                action = torch.multinomial(probs, 1).item()
                
                state, reward, done = env.step(state, action)
                if done:
                    break
            
            conflicts = count_conflicts(adj, state)
            colors_used = len(set(c for c in state if c != -1))
            
            # Only count valid solutions
            if conflicts == 0 and np.sum(state != -1) == N:
                color_counts[colors_used] += 1
                all_colors.append(colors_used)
    
    return dict(color_counts), all_colors


def plot_distribution(distributions, graph_names, chromatic_numbers, output_path=None, title=None):
    """
    Plot sample distributions for multiple graphs.
    
    Args:
        distributions: Dict of graph_name -> {colors: count}
        graph_names: List of graph names
        chromatic_numbers: Dict of graph_name -> chromatic number
        output_path: Path to save figure (optional)
        title: Plot title (optional)
    """
    n_graphs = len(graph_names)
    
    if n_graphs == 1:
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        axes = [axes]
    else:
        cols = min(3, n_graphs)
        rows = (n_graphs + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        axes = axes.flatten() if n_graphs > 1 else [axes]
    
    colors_palette = plt.cm.viridis(np.linspace(0.2, 0.8, 20))
    
    for idx, graph_name in enumerate(graph_names):
        ax = axes[idx]
        dist = distributions[graph_name]
        chromatic = chromatic_numbers.get(graph_name, None)
        
        if not dist:
            ax.text(0.5, 0.5, 'No valid samples', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(graph_name)
            continue
        
        # Prepare data
        min_c = min(dist.keys())
        max_c = max(dist.keys())
        x = list(range(min_c, max_c + 1))
        y = [dist.get(c, 0) for c in x]
        total = sum(y)
        y_pct = [100 * v / total for v in y]
        
        # Color bars - highlight optimal
        bar_colors = []
        for c in x:
            if chromatic and c == chromatic:
                bar_colors.append('green')
            elif chromatic and c < chromatic:
                bar_colors.append('darkgreen')
            else:
                bar_colors.append('steelblue')
        
        bars = ax.bar(x, y_pct, color=bar_colors, edgecolor='black', alpha=0.8)
        
        # Add count labels on bars
        for bar, count, pct in zip(bars, y, y_pct):
            if pct > 2:  # Only label if visible
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{count}', ha='center', va='bottom', fontsize=8)
        
        # Statistics
        all_colors = []
        for c, count in dist.items():
            all_colors.extend([c] * count)
        mean_c = np.mean(all_colors)
        std_c = np.std(all_colors)
        
        # Title and labels
        title_str = f"{graph_name}"
        if chromatic:
            title_str += f" (χ={chromatic})"
        ax.set_title(title_str, fontsize=12, fontweight='bold')
        ax.set_xlabel('Number of Colors', fontsize=10)
        ax.set_ylabel('Percentage (%)', fontsize=10)
        
        # Add statistics text
        stats_text = f"μ={mean_c:.2f}, σ={std_c:.2f}\nn={total}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Mark chromatic number
        if chromatic and chromatic >= min_c and chromatic <= max_c:
            ax.axvline(x=chromatic, color='red', linestyle='--', linewidth=2, label=f'χ={chromatic}')
            ax.legend(loc='upper left', fontsize=8)
        
        ax.set_xticks(x)
        ax.grid(axis='y', alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_graphs, len(axes)):
        axes[idx].set_visible(False)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot sample distribution from trained GFlowNet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot distribution for a single graph
  python plot_distribution.py --checkpoint checkpoints/model.pt --graphs myciel5

  # Plot for multiple graphs
  python plot_distribution.py --checkpoint checkpoints/model.pt --graphs myciel3 myciel4 myciel5

  # Plot all graphs with more samples
  python plot_distribution.py --checkpoint checkpoints/model.pt --all-graphs --samples 1000
        """
    )
    
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file")
    parser.add_argument("--graphs", nargs="+", default=None,
                        help="Specific graphs to evaluate")
    parser.add_argument("--all-graphs", action="store_true",
                        help="Evaluate all available graphs")
    parser.add_argument("--samples", type=int, default=500,
                        help="Number of samples per graph (default: 500)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for plot (e.g., distribution.png)")
    parser.add_argument("--title", type=str, default=None,
                        help="Plot title")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    policy, num_colors, step = load_checkpoint(args.checkpoint, device)
    print(f"  K={num_colors}, step={step}")
    
    # Determine graphs
    if args.all_graphs:
        graph_files = [f for f in sorted(os.listdir(DATA_DIR)) if f.endswith('.col')]
    elif args.graphs:
        graph_files = []
        for g in args.graphs:
            if not g.endswith('.col'):
                g = g + '.col'
            if os.path.exists(os.path.join(DATA_DIR, g)):
                graph_files.append(g)
    else:
        print("Error: Specify --graphs or --all-graphs")
        return
    
    print(f"\nSampling {args.samples} solutions per graph...")
    
    distributions = {}
    graph_names = []
    
    for graph_file in graph_files:
        graph_name = graph_file.replace('.col', '')
        graph_names.append(graph_name)
        
        path = os.path.join(DATA_DIR, graph_file)
        adj = load_col_file(path)
        chromatic = CHROMATIC_NUMBERS.get(graph_file, None)
        
        print(f"  {graph_name} ({adj.shape[0]} nodes, χ={chromatic})...", end=" ", flush=True)
        
        dist, all_colors = sample_solutions(policy, adj, num_colors, device, args.samples)
        distributions[graph_name] = dist
        
        if all_colors:
            mean_c = np.mean(all_colors)
            std_c = np.std(all_colors)
            print(f"μ={mean_c:.2f}±{std_c:.2f}, range=[{min(all_colors)},{max(all_colors)}]")
        else:
            print("no valid samples")
    
    # Create chromatic dict with graph names (without .col)
    chromatic_dict = {k.replace('.col', ''): v for k, v in CHROMATIC_NUMBERS.items()}
    
    # Plot
    title = args.title or f"GFlowNet Sample Distribution (step {step})"
    output_path = args.output or os.path.join(os.path.dirname(__file__), "sample_distribution.png")
    
    plot_distribution(distributions, graph_names, chromatic_dict, output_path, title)


if __name__ == "__main__":
    main()
