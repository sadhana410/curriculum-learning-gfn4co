# problems/graph_coloring/main.py

import argparse
import os
import sys
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from problems.graph_coloring.env import GraphColoringEnv
from problems.graph_coloring.policy import GNNPolicy
from problems.graph_coloring.utils import load_col_file
from losses.trajectorybalance import TrajectoryBalance
from training.trainer import train
from visualization.plot_graph import visualize_coloring


# Data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def main():
    parser = argparse.ArgumentParser(description="Train GFlowNet for Graph Coloring")
    parser.add_argument("--graph", type=str, default="myciel3.col",
                        help="Which graph file to load from data/")
    parser.add_argument("--steps", type=int, default=10000,
                        help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Number of trajectories per training step")
    parser.add_argument("--epsilon", type=float, default=0.5,
                        help="Initial epsilon for exploration (decays to 0.01)")
    args = parser.parse_args()

   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    
    filename = args.graph
    path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(path):
        print(f"Error: Graph file '{filename}' not found in {DATA_DIR}")

    print(f"Loading graph from {path}...\n")
    adj = load_col_file(path)
    if not isinstance(adj, torch.Tensor):
        adj = torch.from_numpy(adj)
    adj = adj.to(device)

    instance = {"adj": adj}
    N = adj.shape[0]
    degrees = adj.sum(dim=1).cpu().numpy() if torch.is_tensor(adj) else adj.sum(axis=1)
    K = int(degrees.max()) + 1
    
    print(f"Graph loaded: {filename} with {N} nodes")
    print()

    env = GraphColoringEnv(instance)

    print(f"Using K = max deg(G)+1 = {K} colors")
    print()

    
    forward = GNNPolicy(num_nodes=N, num_colors=K).to(device)
    backward = GNNPolicy(num_nodes=N, num_colors=K).to(device)

    loss_fn = TrajectoryBalance(forward, backward).to(device)

    optimizer = torch.optim.Adam(
        list(forward.parameters()) +
        list(backward.parameters()) +
        list(loss_fn.parameters()),
        lr=1e-4  # Lower LR for stability with batch training
    )


    print(f"Training Graph Coloring GFlowNet on {filename} ...")
    print(f"Total steps: {args.steps}\n")

    final_state = train(
        env=env,
        forward_policy=forward,
        backward_policy=backward,
        loss_fn=loss_fn,
        optimizer=optimizer,
        steps=args.steps,
        device=device,
        save_dir=None,          
        problem_name=None       
    )

    print("\nFinal sampled terminal state:", final_state)

    visualize_coloring(adj, final_state,
                       title=f"GFlowNet Coloring: {filename}")


if __name__ == "__main__":
    main()
