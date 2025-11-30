# problems/graph_coloring/main.py

import argparse
import os
import sys
import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from problems.graph_coloring.env import GraphColoringEnv
from problems.graph_coloring.policy import GNNPolicy
from problems.graph_coloring.utils import load_col_file
from problems.graph_coloring.trainer import train
from losses.trajectorybalance import TrajectoryBalance
from losses.detailedbalance import DetailedBalance
from losses.subtrajectorybalance import SubTrajectoryBalance
from visualization.plot_graph import visualize_coloring


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


def main():
    parser = argparse.ArgumentParser(description="Train GFlowNet for Graph Coloring")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for optimizer")
    parser.add_argument("--chromatic", type=int, default=4, 
                        help="Chromatic number to train on (selects corresponding myciel graph)")
    parser.add_argument("--max-colors", type=int, default=None,
                        help="Maximum number of colors the model can use (default: number of nodes)")
    parser.add_argument("--steps", type=int, default=10000,
                        help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Number of trajectories per training step")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Initial epsilon for exploration (decays to 0.01)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (higher = more exploration)")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Top-P (Nucleus) sampling threshold (0.0-1.0, 1.0 = full softmax)")
    parser.add_argument("--patience", type=int, default=5000,
                        help="Early stopping patience (steps without improvement)")
    parser.add_argument("--save-every", type=int, default=1000,
                        help="Save checkpoint every N steps (default: 1000)")
    parser.add_argument("--loss", type=str, default="TB", choices=["TB", "DB", "SubTB"],
                        help="Loss function: TB (Trajectory Balance), DB (Detailed Balance), SubTB (Sub-Trajectory Balance)")
    parser.add_argument("--hidden-dim", type=int, default=64,
                        help="Hidden dimension for policy networks (default: 64)")
    args = parser.parse_args()

    # Loss configuration
    LOSS_CONFIG = {
        "SubTB": {"lambda_": 1.0},
        "DB": {},
        "TB": {}
    }

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

    forward = GNNPolicy(num_nodes=N, num_colors=K, hidden_dim=args.hidden_dim).to(device)
    backward = GNNPolicy(num_nodes=N, num_colors=K, hidden_dim=args.hidden_dim).to(device)

    # Select loss function
    if args.loss == "TB":
        loss_fn = TrajectoryBalance(forward, backward).to(device)
    elif args.loss == "DB":
        loss_fn = DetailedBalance(forward, backward).to(device)
    elif args.loss == "SubTB":
        loss_params = LOSS_CONFIG.get("SubTB", {})
        loss_fn = SubTrajectoryBalance(forward, backward, **loss_params).to(device)
    
    print(f"Using Loss: {args.loss}")

    optimizer = torch.optim.Adam(
        list(forward.parameters()) +
        list(backward.parameters()) +
        list(loss_fn.parameters()),
        lr=args.lr  
    )

    env = GraphColoringEnv(instance, num_colors=K, chromatic_number=chromatic_number)

    save_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    
    # Problem name for checkpoint (e.g., myciel3_K4)
    problem_name = filename.replace('.col', '') + f'_K{K}'
    
    print(f"Training Graph Coloring with GNNPolicy + TB loss (K={K})...\n")
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

    visualize_coloring(adj, best_state,
                       title=f"GFlowNet Coloring: {filename} ({best_colors} colors)")


if __name__ == "__main__":
    main()
