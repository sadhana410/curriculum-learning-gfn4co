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
from losses.trajectorybalance import TrajectoryBalance
from training.trainer import train
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
    parser.add_argument("--chromatic", type=int, default=4, 
                        help="Chromatic number to train on (selects corresponding myciel graph)")
    parser.add_argument("--extra-colors", type=int, default=1,
                        help="Extra colors beyond chromatic number (default: 1)")
    parser.add_argument("--steps", type=int, default=10000,
                        help="Number of training steps")
    args = parser.parse_args()

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
    K = chromatic_number + args.extra_colors

    path = os.path.join(DATA_DIR, filename)
    print(f"\nLoading graph from {path}...\n")

    adj = load_col_file(path)

    if not isinstance(adj, torch.Tensor):
        adj = torch.from_numpy(adj) if isinstance(adj, np.ndarray) else torch.tensor(adj)
    adj = adj.to(device)
    instance = {"adj": adj}

    N = adj.shape[0]

    print(f"Graph loaded: {filename} with {N} nodes, chromatic number={chromatic_number}")

    forward = GNNPolicy(num_nodes=N, num_colors=K).to(device)
    backward = GNNPolicy(num_nodes=N, num_colors=K).to(device)

    loss_fn = TrajectoryBalance(forward, backward).to(device)

    optimizer = torch.optim.Adam(
        list(forward.parameters()) +
        list(backward.parameters()) +
        list(loss_fn.parameters()),
        lr=1e-3
    )

    env = GraphColoringEnv(instance, num_colors=K, chromatic_number=chromatic_number)

    save_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    
    # Problem name for checkpoint (e.g., myciel3_K4)
    problem_name = filename.replace('.col', '') + f'_K{K}'
    
    print(f"Training Graph Coloring with GNNPolicy + TB loss (K={K})...\n")
    final_state = train(
        env=env,
        forward_policy=forward,
        backward_policy=backward,
        loss_fn=loss_fn,
        optimizer=optimizer,
        steps=args.steps,          
        device=device,
        save_dir=save_dir,
        problem_name=problem_name
    )

    print("\nFinal state:", final_state)

    visualize_coloring(adj, final_state,
                       title=f"GFlowNet Coloring: {filename}")


if __name__ == "__main__":
    main()
