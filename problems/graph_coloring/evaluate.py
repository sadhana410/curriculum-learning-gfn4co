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

from problems.graph_coloring.env import GraphColoringEnv, ConditionalGraphColoringEnv, GraphInstanceDataset
from problems.graph_coloring.policy import GNNPolicy, ConditionalGNNPolicy
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
    """
    Sample multiple solutions stochastically and return the best.
    
    Returns:
        best_state: Best coloring found
        sample_distribution: Dict mapping colors -> count (for valid solutions)
    """
    best_state = None
    best_colors = float('inf')
    best_conflicts = float('inf')
    
    # Track sample distribution
    sample_distribution = {}  # colors -> count
    
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
            
            # Track sample distribution (only valid colorings)
            if conflicts == 0 and np.sum(state != -1) == env.N:
                sample_distribution[colors_used] = sample_distribution.get(colors_used, 0) + 1
            
            # Prefer: no conflicts, then fewer colors
            if conflicts < best_conflicts or (conflicts == best_conflicts and colors_used < best_colors):
                best_conflicts = conflicts
                best_colors = colors_used
                best_state = state.copy()
    
    return best_state, sample_distribution


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
    stoch_state, sample_dist = sample_stochastic(env, forward, adj, device, num_samples)
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
        'sample_distribution': sample_dist,
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
        
        # Show sample distribution (mean ± std)
        sample_dist = results.get('sample_distribution', {})
        if sample_dist:
            colors_list = []
            for c, count in sample_dist.items():
                colors_list.extend([c] * count)
            if colors_list:
                mean_colors = np.mean(colors_list)
                std_colors = np.std(colors_list)
                min_colors = min(sample_dist.keys())
                max_colors = max(sample_dist.keys())
                total_valid = sum(sample_dist.values())
                print(f"\nSample Distribution (n={total_valid} valid samples):")
                print(f"  Mean: {mean_colors:.2f} ± {std_colors:.2f} colors")
                print(f"  Range: [{min_colors}, {max_colors}]")
                print(f"  Chromatic: {chromatic}")
                breakdown = ", ".join(f"{c}:{n}" for c, n in sorted(sample_dist.items()))
                print(f"  Breakdown: {breakdown}")
            
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
            "sample_distribution": {str(k): v for k, v in results.get('sample_distribution', {}).items()},
        },
    }
    
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    
    print(f"\nEvaluation log saved to: {log_path}")


# ============================================================================
# Conditional GFlowNet Evaluation
# ============================================================================

def load_conditional_checkpoint(checkpoint_path, device):
    """Load a conditional GFlowNet checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get num_colors from checkpoint
    num_colors = checkpoint.get('num_colors', None)
    
    # Try to infer from model weights if not stored
    if num_colors is None:
        emb_weight = checkpoint['forward_policy']['color_embedding.weight']
        num_colors = emb_weight.shape[0] - 1
    
    # Infer hidden_dim
    emb_weight = checkpoint['forward_policy']['color_embedding.weight']
    hidden_dim = emb_weight.shape[1]
    
    # Infer num_layers
    num_layers = 0
    for key in checkpoint['forward_policy'].keys():
        if key.startswith('gnn_layers.'):
            layer_idx = int(key.split('.')[1])
            num_layers = max(num_layers, layer_idx + 1)
    
    return checkpoint, num_colors, hidden_dim, num_layers


def create_conditional_policy_from_checkpoint(checkpoint, device):
    """Create and load conditional policy from checkpoint."""
    forward_state = checkpoint['forward_policy']
    
    # Check if keys have 'shared_policy.' prefix (new format) or not (old format)
    sample_key = list(forward_state.keys())[0]
    has_prefix = sample_key.startswith('shared_policy.')
    
    # Find the embedding weight key
    if has_prefix:
        emb_key = 'shared_policy.color_embedding.weight'
        gnn_prefix = 'shared_policy.gnn_layers.'
    else:
        emb_key = 'color_embedding.weight'
        gnn_prefix = 'gnn_layers.'
    
    # Infer dimensions from checkpoint
    emb_weight = forward_state[emb_key]
    num_colors = emb_weight.shape[0] - 1
    hidden_dim = emb_weight.shape[1]
    
    num_layers = 0
    for key in forward_state.keys():
        if gnn_prefix in key:
            # Extract layer index
            after_prefix = key.split(gnn_prefix)[1]
            layer_idx = int(after_prefix.split('.')[0])
            num_layers = max(num_layers, layer_idx + 1)
    
    # Create policy
    shared_policy = ConditionalGNNPolicy(
        num_colors=num_colors,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    ).to(device)
    
    if has_prefix:
        # New format: load via wrapper
        from problems.graph_coloring.policy import ConditionalGNNPolicyWrapper
        forward = ConditionalGNNPolicyWrapper(shared_policy, mode='forward')
        forward.load_state_dict(forward_state)
        forward.eval()
        # Return the underlying shared policy for evaluation
        return shared_policy, num_colors, hidden_dim, num_layers
    else:
        # Old format: load directly
        shared_policy.load_state_dict(forward_state)
        shared_policy.eval()
        return shared_policy, num_colors, hidden_dim, num_layers


def sample_greedy_conditional(adj, num_colors, policy, device):
    """Sample a solution using greedy policy for conditional GFlowNet."""
    N = adj.shape[0]
    state = -1 * np.ones(N, dtype=int)
    
    # Create temporary env for allowed_actions
    instance = {'adj': adj}
    env = GraphColoringEnv(instance, num_colors=num_colors)
    
    with torch.no_grad():
        while not env.is_terminal(state):
            logits = policy(state, adj, device=device)
            mask = torch.tensor(env.allowed_actions(state), dtype=torch.float32, device=device)
            
            if mask.sum() == 0:
                break
            
            masked_logits = logits.clone()
            masked_logits[mask == 0] = float('-inf')
            action = masked_logits.argmax().item()
            
            state, reward, done = env.step(state, action)
            if done:
                break
    
    return state


def sample_stochastic_conditional(adj, num_colors, policy, device, num_samples=100):
    """
    Sample multiple solutions stochastically for conditional GFlowNet.
    
    Returns:
        best_state: Best coloring found
        sample_distribution: Dict mapping (colors, conflicts) -> count
    """
    N = adj.shape[0]
    best_state = None
    best_colors = float('inf')
    best_conflicts = float('inf')
    
    # Track distribution of samples
    sample_distribution = {}  # (colors, conflicts) -> count
    
    # Create temporary env
    instance = {'adj': adj}
    env = GraphColoringEnv(instance, num_colors=num_colors)
    
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
            
            # Track sample distribution
            key = (colors_used, conflicts)
            sample_distribution[key] = sample_distribution.get(key, 0) + 1
            
            if conflicts < best_conflicts or (conflicts == best_conflicts and colors_used < best_colors):
                best_conflicts = conflicts
                best_colors = colors_used
                best_state = state.copy()
    
    return best_state, sample_distribution


def evaluate_conditional(checkpoint_path, graphs, device, num_samples=100):
    """
    Evaluate a conditional GFlowNet on multiple graphs.
    
    Args:
        checkpoint_path: Path to checkpoint
        graphs: List of graph names or 'all' for all graphs
        device: torch device
        num_samples: Number of stochastic samples per graph
        
    Returns:
        Dictionary of results per graph
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    forward, num_colors, hidden_dim, num_layers = create_conditional_policy_from_checkpoint(checkpoint, device)
    
    print(f"Loaded conditional model:")
    print(f"  Colors (K): {num_colors}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  GNN layers: {num_layers}")
    print(f"  Training step: {checkpoint.get('step', '?')}")
    print()
    
    # Determine which graphs to evaluate
    if graphs == ['all'] or graphs is None:
        graph_files = [f for f in sorted(os.listdir(DATA_DIR)) if f.endswith('.col')]
    else:
        graph_files = []
        for g in graphs:
            if not g.endswith('.col'):
                g = g + '.col'
            if os.path.exists(os.path.join(DATA_DIR, g)):
                graph_files.append(g)
    
    results = {}
    
    for graph_file in graph_files:
        path = os.path.join(DATA_DIR, graph_file)
        adj = load_col_file(path)
        N = adj.shape[0]
        chromatic = CHROMATIC_NUMBERS.get(graph_file, None)
        
        print(f"Evaluating: {graph_file} ({N} nodes, chromatic={chromatic or '?'})")
        
        # Greedy evaluation
        greedy_state = sample_greedy_conditional(adj, num_colors, forward, device)
        greedy_conflicts = count_conflicts(adj, greedy_state)
        greedy_colors = len(set(c for c in greedy_state if c != -1))
        greedy_colored = np.sum(greedy_state != -1)
        
        # Stochastic evaluation
        stoch_state, sample_dist = sample_stochastic_conditional(adj, num_colors, forward, device, num_samples)
        stoch_conflicts = count_conflicts(adj, stoch_state)
        stoch_colors = len(set(c for c in stoch_state if c != -1))
        stoch_colored = np.sum(stoch_state != -1)
        
        greedy_valid = greedy_conflicts == 0 and greedy_colored == N
        stoch_valid = stoch_conflicts == 0 and stoch_colored == N
        
        results[graph_file] = {
            'nodes': N,
            'chromatic': chromatic,
            'greedy': {
                'colors': greedy_colors,
                'conflicts': greedy_conflicts,
                'colored': int(greedy_colored),
                'is_valid': bool(greedy_valid),
                'is_optimal': bool(greedy_valid and chromatic and greedy_colors <= chromatic),
            },
            'stochastic': {
                'colors': stoch_colors,
                'conflicts': stoch_conflicts,
                'colored': int(stoch_colored),
                'is_valid': bool(stoch_valid),
                'is_optimal': bool(stoch_valid and chromatic and stoch_colors <= chromatic),
            }
        }
        
        # Print summary
        greedy_valid = results[graph_file]['greedy']['is_valid']
        stoch_valid = results[graph_file]['stochastic']['is_valid']
        greedy_status = "✓" if greedy_valid else "✗"
        stoch_status = "✓" if stoch_valid else "✗"
        
        greedy_info = f"{greedy_colors} colors, {greedy_conflicts} conflicts"
        if greedy_colored < N:
            greedy_info += f", {greedy_colored}/{N} colored"
        print(f"  Greedy:     {greedy_info} {greedy_status}")
        
        stoch_info = f"{stoch_colors} colors, {stoch_conflicts} conflicts"
        if stoch_colored < N:
            stoch_info += f", {stoch_colored}/{N} colored"
        print(f"  Stochastic: {stoch_info} {stoch_status}")
        
        # Store states and sample distribution for output
        results[graph_file]['greedy']['state'] = greedy_state.tolist()
        results[graph_file]['stochastic']['state'] = stoch_state.tolist()
        results[graph_file]['stochastic']['sample_distribution'] = sample_dist
    
    return results


def main_conditional_eval(args):
    """Main function for conditional GFlowNet evaluation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    print("=" * 60)
    print("CONDITIONAL GFLOWNET EVALUATION")
    print("=" * 60)
    
    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        # Find latest conditional checkpoint
        checkpoints = []
        if os.path.exists(CHECKPOINT_DIR):
            for f in os.listdir(CHECKPOINT_DIR):
                if f.endswith('.pt') and 'conditional' in f:
                    checkpoints.append(os.path.join(CHECKPOINT_DIR, f))
        
        if not checkpoints:
            print("No conditional checkpoint found!")
            return
        
        # Sort by step number
        def get_step(path):
            name = os.path.basename(path)
            if 'step_' in name:
                try:
                    return int(name.split('step_')[1].split('.')[0])
                except:
                    pass
            return 0
        
        checkpoints.sort(key=get_step, reverse=True)
        checkpoint_path = checkpoints[0]
    
    print(f"Checkpoint: {os.path.basename(checkpoint_path)}\n")
    
    # Determine graphs to evaluate
    if args.graphs:
        graphs = args.graphs
    elif args.all_graphs:
        graphs = ['all']
    else:
        graphs = ['all']  # Default to all
    
    # Run evaluation
    results = evaluate_conditional(checkpoint_path, graphs, device, args.samples)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_valid = sum(1 for r in results.values() if r['stochastic']['is_valid'])
    total_optimal = sum(1 for r in results.values() if r['stochastic']['is_optimal'])
    
    print(f"Valid colorings: {total_valid}/{len(results)}")
    print(f"Optimal colorings: {total_optimal}/{len(results)}")
    
    # Print best solutions
    print("\n" + "=" * 60)
    print("BEST SOLUTIONS (Stochastic)")
    print("=" * 60)
    for graph_file, r in results.items():
        print(f"\n{graph_file}:")
        state = r['stochastic']['state']
        colors_used = r['stochastic']['colors']
        conflicts = r['stochastic']['conflicts']
        is_valid = r['stochastic']['is_valid']
        is_optimal = r['stochastic']['is_optimal']
        
        status = "OPTIMAL" if is_optimal else ("VALID" if is_valid else "INVALID")
        print(f"  Status: {status}")
        print(f"  Colors used: {colors_used}, Conflicts: {conflicts}")
        print(f"  Coloring: {state}")
        
        # Show sample distribution statistics (mean ± std)
        sample_dist = r['stochastic'].get('sample_distribution', {})
        if sample_dist:
            # Compute mean and std of colors from valid samples
            colors_list = []
            for (num_colors, num_conflicts), count in sample_dist.items():
                if num_conflicts == 0:  # Only valid colorings
                    colors_list.extend([num_colors] * count)
            
            if colors_list:
                mean_colors = np.mean(colors_list)
                std_colors = np.std(colors_list)
                min_colors = min(c for (c, conf), _ in sample_dist.items() if conf == 0)
                max_colors = max(c for (c, conf), _ in sample_dist.items() if conf == 0)
                chromatic = r.get('chromatic', '?')
                print(f"\n  Sample Distribution (n={len(colors_list)} valid samples):")
                print(f"    Mean: {mean_colors:.2f} ± {std_colors:.2f} colors")
                print(f"    Range: [{min_colors}, {max_colors}]")
                print(f"    Chromatic: {chromatic}")
                
                # Show histogram-style breakdown
                color_counts = {}
                for (c, conf), count in sample_dist.items():
                    if conf == 0:
                        color_counts[c] = color_counts.get(c, 0) + count
                
                print(f"    Breakdown: ", end="")
                breakdown = ", ".join(f"{c}:{n}" for c, n in sorted(color_counts.items()))
                print(breakdown)
    
    # Save results
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"eval_conditional_{timestamp}.json"
    log_path = os.path.join(LOG_DIR, log_name)
    
    # Convert tuple keys to strings for JSON serialization
    results_json = {}
    for graph_file, r in results.items():
        r_copy = dict(r)
        if 'sample_distribution' in r_copy.get('stochastic', {}):
            # Convert (colors, conflicts) tuple keys to "colors,conflicts" strings
            dist = r_copy['stochastic']['sample_distribution']
            r_copy['stochastic'] = dict(r_copy['stochastic'])
            r_copy['stochastic']['sample_distribution'] = {
                f"{k[0]},{k[1]}": v for k, v in dist.items()
            }
        results_json[graph_file] = r_copy
    
    log_data = {
        "type": "conditional_evaluation",
        "timestamp": timestamp,
        "checkpoint": os.path.basename(checkpoint_path),
        "num_samples": args.samples,
        "results": results_json,
        "summary": {
            "total_graphs": len(results),
            "valid_colorings": total_valid,
            "optimal_colorings": total_optimal,
        }
    }
    
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    
    print(f"\nResults saved to: {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained GFlowNet on Graph Coloring")
    
    # Mode selection
    parser.add_argument("--conditional", action="store_true",
                        help="Evaluate conditional GFlowNet")
    
    # Common arguments
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of stochastic samples")
    
    # Single-instance arguments
    parser.add_argument("--graph", type=str, default=None,
                        help="[Single] Graph file to evaluate")
    parser.add_argument("--chromatic", type=int, default=None,
                        help="[Single] Select graph by chromatic number")
    parser.add_argument("--num-colors", type=int, default=None,
                        help="[Single] Number of colors (K)")
    parser.add_argument("--hidden-dim", type=int, default=64,
                        help="[Single] Hidden dimension (must match training)")
    parser.add_argument("--list", action="store_true",
                        help="[Single] List available graphs")
    
    # Conditional arguments
    parser.add_argument("--graphs", nargs="+", default=None,
                        help="[Conditional] Specific graphs to evaluate")
    parser.add_argument("--all-graphs", action="store_true",
                        help="[Conditional] Evaluate on all graphs")
    
    args = parser.parse_args()
    
    if args.conditional:
        main_conditional_eval(args)
    else:
        # Original single-instance evaluation
        main_single_instance_eval(args)


def main_single_instance_eval(args):
    """Original single-instance evaluation (refactored from old main)."""
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
        
        # Output best solution
        print(f"\n" + "=" * 60)
        print("BEST SOLUTION")
        print("=" * 60)
        best_state = results['stochastic_state']
        best_conflicts = results['stochastic_conflicts']
        best_colors = results['stochastic_colors']
        best_colored = results['stochastic_colored']
        
        is_valid = best_conflicts == 0 and best_colored == results['nodes']
        is_optimal = is_valid and best_colors <= chromatic
        
        status = "OPTIMAL" if is_optimal else ("VALID" if is_valid else "INVALID")
        print(f"Status: {status}")
        print(f"Colors used: {best_colors}, Conflicts: {best_conflicts}")
        print(f"Coloring: {best_state.tolist()}")
            
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
