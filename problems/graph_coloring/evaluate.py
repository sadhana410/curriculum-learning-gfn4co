# problems/graph_coloring/evaluate.py

import argparse
import os
import sys
import json
from datetime import datetime
import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from problems.graph_coloring.env import GraphColoringEnv
from problems.graph_coloring.policy import GNNPolicy
from problems.graph_coloring.utils import load_col_file

# Data and checkpoint directories relative to this file
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")

# Myciel graphs chromatic numbers
CHROMATIC_NUMBERS = {
    "myciel2.col": 3,
    "myciel3.col": 4,
    "myciel4.col": 5,
    "myciel5.col": 6,
    "myciel6.col": 7,
    "myciel7.col": 8,
}


def load_checkpoint(checkpoint_path, device):
    """Load a trained model from checkpoint, auto-detecting model parameters."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Auto-detect model parameters from checkpoint
    # embedding.weight shape is (K+1, hidden_dim) where K is num_colors
    emb_weight = checkpoint['forward_policy']['embedding.weight']
    hidden_dim = emb_weight.shape[1]
    num_colors = emb_weight.shape[0] - 1  # K+1 embeddings (including -1 state)
    
    # out.weight shape is (K, hidden_dim)
    out_weight = checkpoint['forward_policy']['out.weight']
    
    # W_self.weight shape is (hidden_dim, hidden_dim)
    # We can infer num_nodes from the checkpoint if stored, otherwise need to pass it
    
    return checkpoint, num_colors, hidden_dim


def create_policy_from_checkpoint(checkpoint, num_nodes, device):
    """Create and load policy from checkpoint."""
    emb_weight = checkpoint['forward_policy']['embedding.weight']
    hidden_dim = emb_weight.shape[1]
    num_colors = emb_weight.shape[0] - 1
    
    forward = GNNPolicy(num_nodes=num_nodes, num_colors=num_colors, hidden_dim=hidden_dim).to(device)
    forward.load_state_dict(checkpoint['forward_policy'])
    forward.eval()
    
    return forward, num_colors, hidden_dim


def count_conflicts(adj, state):
    """Count number of edge conflicts in a coloring."""
    adj_np = adj.cpu().numpy() if torch.is_tensor(adj) else adj
    colored_mask = state != -1
    same_color = (state[:, None] == state[None, :]) & colored_mask[:, None] & colored_mask[None, :]
    conflicts = int(np.sum(adj_np * same_color) // 2)
    return conflicts


def sample_greedy(env, policy, adj, device):
    """Sample a solution using greedy (argmax) policy."""
    state = env.reset()
    
    with torch.no_grad():
        while not env.is_terminal(state):
            logits = policy(state, adj, device=device)
            mask = torch.tensor(env.allowed_actions(state), dtype=torch.float32, device=device)
            
            if mask.sum() == 0:
                break
            
            # Greedy: pick best action
            masked_logits = logits.clone()
            masked_logits[mask == 0] = float('-inf')
            action = masked_logits.argmax().item()
            
            state, reward, done = env.step(state, action)
            if done:
                break
    
    return state


def sample_stochastic(env, policy, adj, device, num_samples=100):
    """Sample multiple solutions stochastically and return the best."""
    best_state = None
    best_colors = float('inf')
    best_conflicts = float('inf')
    
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
            
            # Prefer: no conflicts, then fewer colors
            if conflicts < best_conflicts or (conflicts == best_conflicts and colors_used < best_colors):
                best_conflicts = conflicts
                best_colors = colors_used
                best_state = state.copy()
    
    return best_state


def evaluate_graph(filename, checkpoint_path, device, num_samples=100):
    """Evaluate a trained model on a specific graph."""
    # Load graph
    path = os.path.join(DATA_DIR, filename)
    adj = load_col_file(path)
    if not isinstance(adj, torch.Tensor):
        adj = torch.from_numpy(adj) if isinstance(adj, np.ndarray) else torch.tensor(adj)
    adj = adj.to(device)
    
    N = adj.shape[0]
    chromatic = CHROMATIC_NUMBERS.get(filename, 4)
    
    # Load checkpoint and auto-detect parameters
    checkpoint, num_colors, hidden_dim = load_checkpoint(checkpoint_path, device)
    
    # Create environment with detected num_colors
    instance = {"adj": adj}
    env = GraphColoringEnv(instance, num_colors=num_colors, chromatic_number=chromatic)
    
    # Create model from checkpoint
    forward, num_colors, hidden_dim = create_policy_from_checkpoint(checkpoint, N, device)
    
    # Greedy evaluation
    greedy_state = sample_greedy(env, forward, adj, device)
    greedy_conflicts = count_conflicts(adj, greedy_state)
    greedy_colors = len(set(c for c in greedy_state if c != -1))
    greedy_colored = np.sum(greedy_state != -1)
    
    # Stochastic evaluation (best of N samples)
    stoch_state = sample_stochastic(env, forward, adj, device, num_samples)
    stoch_conflicts = count_conflicts(adj, stoch_state)
    stoch_colors = len(set(c for c in stoch_state if c != -1))
    stoch_colored = np.sum(stoch_state != -1)
    
    # Results
    results = {
        'filename': filename,
        'nodes': N,
        'num_colors': num_colors,
        'chromatic': chromatic,
        'greedy_colors': greedy_colors,
        'greedy_conflicts': greedy_conflicts,
        'greedy_colored': greedy_colored,
        'greedy_state': greedy_state,
        'stochastic_colors': stoch_colors,
        'stochastic_conflicts': stoch_conflicts,
        'stochastic_colored': stoch_colored,
        'stochastic_state': stoch_state,
        'training_step': checkpoint.get('step', '?'),
    }
    
    return results


def find_checkpoint(graph_name, checkpoint_dir):
    """Find the latest checkpoint for a specific graph."""
    # Remove .col extension for matching
    base_name = graph_name.replace('.col', '')
    checkpoints = []
    
    if os.path.exists(checkpoint_dir):
        for f in os.listdir(checkpoint_dir):
            if f.endswith('.pt'):
                # Prefer checkpoints matching the graph name
                if f.startswith(f'{base_name}_'):
                    checkpoints.append(os.path.join(checkpoint_dir, f))
    
    # If no graph-specific checkpoints, fall back to generic ones
    if not checkpoints and os.path.exists(checkpoint_dir):
        for f in os.listdir(checkpoint_dir):
            if f.endswith('.pt') and f.startswith('checkpoint_'):
                checkpoints.append(os.path.join(checkpoint_dir, f))
    
    if not checkpoints:
        return None
    
    # Sort by step number (highest first)
    def get_step(path):
        name = os.path.basename(path)
        if 'step_' in name:
            try:
                return int(name.split('step_')[1].split('.')[0])
            except:
                pass
        return 0
    
    checkpoints.sort(key=get_step, reverse=True)
    return checkpoints[0]


def list_graphs():
    """List available graph files."""
    graphs = []
    if os.path.exists(DATA_DIR):
        for f in sorted(os.listdir(DATA_DIR)):
            if f.endswith('.col'):
                graphs.append(f)
    return graphs


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained GFlowNet on Graph Coloring")
    parser.add_argument("--graph", type=str, default=None,
                        help="Graph file to evaluate (e.g., myciel3.col)")
    parser.add_argument("--chromatic", type=int, default=None,
                        help="Select graph by chromatic number")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file")
    parser.add_argument("--num-colors", type=int, default=None,
                        help="Number of colors (K). Default: chromatic + 1")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of stochastic samples")
    parser.add_argument("--hidden-dim", type=int, default=64,
                        help="Hidden dimension (must match training)")
    parser.add_argument("--list", action="store_true",
                        help="List available graphs")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    graphs = list_graphs()
    
    if args.list or (args.graph is None and args.chromatic is None):
        print("Available Graph Coloring Problems:")
        print("-" * 50)
        print(f"{'File':<15} {'Nodes':<8} {'Chromatic':<10}")
        print("-" * 50)
        for g in graphs:
            path = os.path.join(DATA_DIR, g)
            adj = load_col_file(path)
            N = adj.shape[0]
            chrom = CHROMATIC_NUMBERS.get(g, "?")
            print(f"{g:<15} {N:<8} {chrom:<10}")
        print("-" * 50)
        print("\nUse --graph <file> or --chromatic <num> to evaluate")
        return
    
    # Determine which graph to evaluate
    if args.chromatic:
        # Find graph by chromatic number
        for fname, chrom in CHROMATIC_NUMBERS.items():
            if chrom == args.chromatic:
                args.graph = fname
                break
        if args.graph is None:
            print(f"Error: No graph with chromatic number {args.chromatic}")
            return
    
    if args.graph not in graphs:
        print(f"Error: Graph '{args.graph}' not found.")
        print(f"Available: {graphs}")
        return
    
    chromatic = CHROMATIC_NUMBERS.get(args.graph, 4)
    
    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = find_checkpoint(args.graph, CHECKPOINT_DIR)
    
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print(f"No checkpoint found in {CHECKPOINT_DIR}")
        return
    
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\nGraph: {args.graph}")
    print(f"Checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"Chromatic number: {chromatic}")
    
    try:
        results = evaluate_graph(
            args.graph, checkpoint_path, device, args.samples
        )
        
        num_colors = results['num_colors']
        print(f"Colors available (K): {num_colors}")
        print(f"\nNodes: {results['nodes']}")
        print(f"Training step: {results['training_step']}")
        
        print(f"\nGreedy solution:")
        print(f"  Colored: {results['greedy_colored']}/{results['nodes']}")
        print(f"  Colors used: {results['greedy_colors']}/{num_colors}")
        print(f"  Conflicts: {results['greedy_conflicts']}")
        if results['greedy_conflicts'] == 0 and results['greedy_colored'] == results['nodes']:
            if results['greedy_colors'] <= chromatic:
                print(f"  ✓ OPTIMAL (chromatic number achieved!)")
            else:
                print(f"  ✓ Valid coloring (+{results['greedy_colors'] - chromatic} extra colors)")
        
        print(f"\nStochastic (best of {args.samples}):")
        print(f"  Colored: {results['stochastic_colored']}/{results['nodes']}")
        print(f"  Colors used: {results['stochastic_colors']}/{num_colors}")
        print(f"  Conflicts: {results['stochastic_conflicts']}")
        if results['stochastic_conflicts'] == 0 and results['stochastic_colored'] == results['nodes']:
            if results['stochastic_colors'] <= chromatic:
                print(f"  ✓ OPTIMAL (chromatic number achieved!)")
            else:
                print(f"  ✓ Valid coloring (+{results['stochastic_colors'] - chromatic} extra colors)")
        
        # Show color distribution
        print(f"\nColor distribution (stochastic):")
        color_counts = {}
        for c in results['stochastic_state']:
            if c != -1:
                color_counts[c] = color_counts.get(c, 0) + 1
        for c in sorted(color_counts.keys()):
            print(f"  Color {c}: {color_counts[c]} nodes")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Save evaluation log
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"eval_{args.graph.replace('.col', '')}_{timestamp}.json"
    log_path = os.path.join(LOG_DIR, log_name)
    
    log_data = {
        "type": "evaluation",
        "timestamp": timestamp,
        "graph": args.graph,
        "checkpoint": os.path.basename(checkpoint_path),
        "chromatic_number": chromatic,
        "num_colors": results['num_colors'],
        "nodes": results['nodes'],
        "training_step": results['training_step'],
        "num_samples": args.samples,
        "greedy": {
            "colored": int(results['greedy_colored']),
            "colors_used": results['greedy_colors'],
            "conflicts": results['greedy_conflicts'],
            "is_valid": results['greedy_conflicts'] == 0 and results['greedy_colored'] == results['nodes'],
            "is_optimal": results['greedy_conflicts'] == 0 and results['greedy_colors'] <= chromatic,
            "state": results['greedy_state'].tolist(),
        },
        "stochastic": {
            "colored": int(results['stochastic_colored']),
            "colors_used": results['stochastic_colors'],
            "conflicts": results['stochastic_conflicts'],
            "is_valid": results['stochastic_conflicts'] == 0 and results['stochastic_colored'] == results['nodes'],
            "is_optimal": results['stochastic_conflicts'] == 0 and results['stochastic_colors'] <= chromatic,
            "state": results['stochastic_state'].tolist(),
        },
    }
    
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    
    print(f"\nEvaluation log saved to: {log_path}")


if __name__ == "__main__":
    main()
