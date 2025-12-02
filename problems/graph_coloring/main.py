# problems/graph_coloring/main.py

import argparse
import os
import sys
import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from problems.graph_coloring.env import GraphColoringEnv, ConditionalGraphColoringEnv, GraphInstanceDataset
from problems.graph_coloring.policy import GNNPolicy, GNNPolicyWrapper, ConditionalGNNPolicy, ConditionalGNNPolicyWrapper
from problems.graph_coloring.utils import load_col_file
from problems.graph_coloring.trainer import train, train_conditional
from losses.trajectorybalance import TrajectoryBalance
from losses.detailedbalance import DetailedBalance
from losses.subtrajectorybalance import SubTrajectoryBalance


# Data directory relative to this file
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Myciel graphs chromatic numbers (known values)
# Note: myciel2 is triangle-free but needs 3 colors.
# The sequence is: myciel2 -> 3 colors, myciel3 -> 4 colors, myciel4 -> 5 colors...
CHROMATIC_NUMBERS = {
    "myciel2.col": 3,
    "myciel3.col": 4,
    "myciel4.col": 5,
    "myciel5.col": 6,
    "myciel6.col": 7,
    "myciel7.col": 8,
}


def main_single_instance(args):
    """Train on a single graph instance (original behavior)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Find the graph file matching the chromatic number
    filename = None
    for fname, chrom in CHROMATIC_NUMBERS.items():
        if chrom == args.chromatic:
            filename = fname
            break
    
    if filename is None:
        print(f"Error: No graph found with chromatic number {args.chromatic}")
        print(f"Available chromatic numbers: {sorted(set(CHROMATIC_NUMBERS.values()))}")
        return

    col_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".col")]
    col_files.sort()  

    print("Found dataset graphs:")
    for f in col_files:
        chrom = CHROMATIC_NUMBERS.get(f, "?")
        marker = " <--" if f == filename else ""
        print(f" - {f} (chromatic={chrom}){marker}")

    chromatic_number = args.chromatic

    path = os.path.join(DATA_DIR, filename)
    print(f"\nLoading graph from {path}...\n")

    adj = load_col_file(path)
    N = adj.shape[0]
    
    # K is the max colors available; defaults to number of nodes if not specified
    K = args.max_colors if args.max_colors is not None else N

    if not isinstance(adj, torch.Tensor):
        adj = torch.from_numpy(adj) if isinstance(adj, np.ndarray) else torch.tensor(adj)
    adj = adj.to(device)
    instance = {"adj": adj}

    print(f"Graph loaded: {filename} with {N} nodes, chromatic number={chromatic_number}")

    # Create shared policy with separate forward/backward heads
    shared_policy = GNNPolicy(num_nodes=N, num_colors=K, hidden_dim=args.hidden_dim).to(device)
    forward = GNNPolicyWrapper(shared_policy, mode='forward')
    backward = GNNPolicyWrapper(shared_policy, mode='backward')

    # Select loss function
    LOSS_CONFIG = {"SubTB": {"lambda_": 1.0}, "DB": {}, "TB": {}}
    if args.loss == "TB":
        loss_fn = TrajectoryBalance(forward, backward).to(device)
    elif args.loss == "DB":
        loss_fn = DetailedBalance(forward, backward).to(device)
    elif args.loss == "SubTB":
        loss_params = LOSS_CONFIG.get("SubTB", {})
        loss_fn = SubTrajectoryBalance(forward, backward, **loss_params).to(device)
    
    print(f"Using Loss: {args.loss}")

    # Optimizer uses shared_policy parameters (forward/backward share backbone)
    optimizer = torch.optim.Adam(
        list(shared_policy.parameters()) +
        list(loss_fn.parameters()),
        lr=args.lr  
    )

    env = GraphColoringEnv(instance, num_colors=K, chromatic_number=chromatic_number)

    save_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    
    # Problem name for checkpoint (e.g., myciel3_K4)
    problem_name = filename.replace('.col', '') + f'_K{K}'
    
    print(f"Training Graph Coloring with GNNPolicy + {args.loss} loss (K={K})...\n")
    best_state, best_colors = train(
        env=env,
        forward_policy=forward,
        backward_policy=backward,
        loss_fn=loss_fn,
        optimizer=optimizer,
        steps=args.steps,          
        device=device,
        save_dir=save_dir,
        problem_name=problem_name,
        batch_size=args.batch_size,
        epsilon_start=args.epsilon,
        log_dir=log_dir,
        save_every=args.save_every,
        temperature=args.temperature,
        top_p=args.top_p,
        early_stop_patience=args.patience
    )

    print(f"\n{'='*50}")
    print(f"Training complete!")
    print(f"Best valid coloring found:")
    print(f"  Colors used: {best_colors} (chromatic number: {chromatic_number})")
    print(f"  State: {best_state}")
    if best_colors == chromatic_number:
        print(f"  ✓ Optimal coloring achieved!")
    else:
        print(f"  Gap to optimal: {best_colors - chromatic_number} extra colors")

    try:
        from visualization.plot_graph import visualize_coloring
        visualize_coloring(adj, best_state,
                           title=f"GFlowNet Coloring: {filename} ({best_colors} colors)")
    except ImportError:
        print("(Visualization skipped - matplotlib not available)")


def main_conditional(args):
    """Train conditional GFlowNet on multiple graph instances."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    print("=" * 60)
    print("CONDITIONAL GFLOWNET TRAINING")
    print("=" * 60)
    
    # Load instances
    instances = []
    
    if args.graphs:
        # Load specific graphs
        for graph_name in args.graphs:
            if not graph_name.endswith('.col'):
                graph_name = graph_name + '.col'
            path = os.path.join(DATA_DIR, graph_name)
            if os.path.exists(path):
                adj = load_col_file(path)
                instances.append({
                    'adj': adj,
                    'name': graph_name.replace('.col', ''),
                    'chromatic_number': CHROMATIC_NUMBERS.get(graph_name, None)
                })
                print(f"  Loaded: {graph_name} ({adj.shape[0]} nodes)")
            else:
                print(f"  Warning: {graph_name} not found")
    elif args.all_graphs:
        # Load all graphs from data directory
        for fname in sorted(os.listdir(DATA_DIR)):
            if fname.endswith('.col'):
                path = os.path.join(DATA_DIR, fname)
                adj = load_col_file(path)
                instances.append({
                    'adj': adj,
                    'name': fname.replace('.col', ''),
                    'chromatic_number': CHROMATIC_NUMBERS.get(fname, None)
                })
                print(f"  Loaded: {fname} ({adj.shape[0]} nodes)")
    elif args.random_graphs:
        # Generate random graphs
        print(f"  Generating {args.random_graphs} random graphs...")
        dataset = GraphInstanceDataset.generate_random_graphs(
            num_graphs=args.random_graphs,
            min_nodes=args.min_nodes,
            max_nodes=args.max_nodes,
            edge_prob=args.edge_prob,
            seed=args.seed
        )
        instances = dataset.instances
        for inst in instances[:5]:
            print(f"    {inst['name']}: {inst['adj'].shape[0]} nodes")
        if len(instances) > 5:
            print(f"    ... and {len(instances) - 5} more")
    else:
        # Default: load all myciel graphs
        print("  Loading all myciel graphs (default)...")
        for fname in sorted(os.listdir(DATA_DIR)):
            if fname.endswith('.col'):
                path = os.path.join(DATA_DIR, fname)
                adj = load_col_file(path)
                instances.append({
                    'adj': adj,
                    'name': fname.replace('.col', ''),
                    'chromatic_number': CHROMATIC_NUMBERS.get(fname, None)
                })
                print(f"  Loaded: {fname} ({adj.shape[0]} nodes)")
    
    if not instances:
        print("Error: No instances loaded!")
        return
    
    print(f"\nTotal instances: {len(instances)}")
    
    # Determine K (max colors)
    max_nodes = max(inst['adj'].shape[0] for inst in instances)
    K = args.max_colors if args.max_colors is not None else max_nodes
    print(f"Max colors (K): {K}")
    
    # Create conditional environment
    env = ConditionalGraphColoringEnv(instances, num_colors=K)
    
    # Create shared conditional policy with separate forward/backward heads
    shared_policy = ConditionalGNNPolicy(
        num_colors=K, 
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    ).to(device)
    forward = ConditionalGNNPolicyWrapper(shared_policy, mode='forward')
    backward = ConditionalGNNPolicyWrapper(shared_policy, mode='backward')
    
    # Select loss function
    LOSS_CONFIG = {"SubTB": {"lambda_": 1.0}, "DB": {}, "TB": {}}
    if args.loss == "TB":
        loss_fn = TrajectoryBalance(forward, backward).to(device)
    elif args.loss == "DB":
        loss_fn = DetailedBalance(forward, backward).to(device)
    elif args.loss == "SubTB":
        loss_params = LOSS_CONFIG.get("SubTB", {})
        loss_fn = SubTrajectoryBalance(forward, backward, **loss_params).to(device)
    
    print(f"Loss function: {args.loss}")
    
    # Optimizer uses shared_policy parameters (forward/backward share backbone)
    optimizer = torch.optim.Adam(
        list(shared_policy.parameters()) +
        list(loss_fn.parameters()),
        lr=args.lr
    )
    
    save_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    
    # Build problem name based on graphs used
    if args.random_graphs:
        problem_name = f"conditional_random{args.random_graphs}_K{K}"
    elif args.graphs:
        # Use graph names (e.g., "conditional_myciel3-myciel4-myciel5_K47")
        graph_names = "-".join(sorted([inst['name'] for inst in instances]))
        # Truncate if too long
        if len(graph_names) > 50:
            graph_names = graph_names[:47] + "..."
        problem_name = f"conditional_{graph_names}_K{K}"
    elif args.all_graphs:
        problem_name = f"conditional_all_K{K}"
    else:
        # Default case
        graph_names = "-".join(sorted([inst['name'] for inst in instances]))
        if len(graph_names) > 50:
            graph_names = graph_names[:47] + "..."
        problem_name = f"conditional_{graph_names}_K{K}"
    
    print(f"\nStarting conditional training...\n")
    
    best_per_instance = train_conditional(
        env=env,
        forward_policy=forward,
        backward_policy=backward,
        loss_fn=loss_fn,
        optimizer=optimizer,
        steps=args.steps,
        device=device,
        save_dir=save_dir,
        problem_name=problem_name,
        batch_size=args.batch_size,
        epsilon_start=args.epsilon,
        log_dir=log_dir,
        save_every=args.save_every,
        temperature=args.temperature,
        top_p=args.top_p,
        early_stop_patience=args.patience,
        same_instance_per_batch=args.same_instance_per_batch,
        resume_from=args.resume
    )
    
    return best_per_instance


def main():
    parser = argparse.ArgumentParser(description="Train GFlowNet for Graph Coloring")
    
    # Mode selection
    parser.add_argument("--conditional", action="store_true",
                        help="Use conditional GFlowNet (train on multiple instances)")
    
    # Common arguments
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for optimizer")
    parser.add_argument("--max-colors", type=int, default=None,
                        help="Maximum number of colors (K). Default: max nodes across instances")
    parser.add_argument("--steps", type=int, default=10000,
                        help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Number of trajectories per training step")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Initial epsilon for exploration (decays to 0.01)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (higher = more exploration)")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Top-P (Nucleus) sampling threshold (0.0-1.0)")
    parser.add_argument("--patience", type=int, default=-1,
                        help="Early stopping patience (steps without improvement)")
    parser.add_argument("--save-every", type=int, default=1000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--loss", type=str, default="TB", choices=["TB", "DB", "SubTB"],
                        help="Loss function: TB, DB, or SubTB")
    parser.add_argument("--hidden-dim", type=int, default=64,
                        help="Hidden dimension for policy networks")
    parser.add_argument("--num-layers", type=int, default=3,
                        help="[Conditional] Number of GNN layers")
    
    # Single-instance arguments
    parser.add_argument("--chromatic", type=int, default=4, 
                        help="[Single] Chromatic number to train on")
    
    # Conditional arguments
    parser.add_argument("--graphs", nargs="+", default=None,
                        help="[Conditional] Specific graph files to train on")
    parser.add_argument("--all-graphs", action="store_true",
                        help="[Conditional] Train on all graphs in data directory")
    parser.add_argument("--random-graphs", type=int, default=None,
                        help="[Conditional] Generate N random graphs for training")
    parser.add_argument("--min-nodes", type=int, default=10,
                        help="[Conditional] Min nodes for random graphs")
    parser.add_argument("--max-nodes", type=int, default=50,
                        help="[Conditional] Max nodes for random graphs")
    parser.add_argument("--edge-prob", type=float, default=0.3,
                        help="[Conditional] Edge probability for random graphs")
    parser.add_argument("--seed", type=int, default=42,
                        help="[Conditional] Random seed for graph generation")
    parser.add_argument("--same-instance-per-batch", action="store_true", default=True,
                        help="[Conditional] Use same instance for all trajectories in batch")
    parser.add_argument("--mixed-batch", dest="same_instance_per_batch", action="store_false",
                        help="[Conditional] Mix different instances in each batch")
    parser.add_argument("--resume", type=str, default=None,
                        help="[Conditional] Resume from checkpoint (path or 'latest')")
    
    args = parser.parse_args()
    
    if args.conditional:
        main_conditional(args)
    else:
        main_single_instance(args)


if __name__ == "__main__":
    main()
