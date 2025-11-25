# main_graph_colouring.py

import os
import numpy as np
import torch

from envs.graph_colouring_env import GraphColoringEnv
from models.policy_net import GNNPolicy
from losses.trajectorybalance import TrajectoryBalance
from training.trainer import train
from visualization.plot_graph import visualize_coloring
from utils.load_col_graph import load_col_file


DATA_DIR = "data"   # folder containing .col files


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    col_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".col")]
    col_files.sort()  

    print("Found dataset graphs:")
    for f in col_files:
        print(" -", f)

    filename = "myciel4.col"         

    path = os.path.join(DATA_DIR, filename)
    print(f"\nLoading graph from {path}...\n")

    adj = load_col_file(path)

    if not isinstance(adj, torch.Tensor):
        adj = torch.from_numpy(adj) if isinstance(adj, np.ndarray) else torch.tensor(adj)
    adj = adj.to(device)
    instance = {"adj": adj}

    N = adj.shape[0]
    K = 4                

    print(f"Graph loaded: {filename} with {N} nodes")

    forward = GNNPolicy(num_nodes=N, num_colors=K).to(device)
    backward = GNNPolicy(num_nodes=N, num_colors=K).to(device)

    loss_fn = TrajectoryBalance(forward, backward).to(device)

    optimizer = torch.optim.Adam(
        list(forward.parameters()) +
        list(backward.parameters()) +
        list(loss_fn.parameters()),
        lr=1e-3
    )

    env = GraphColoringEnv(instance, num_colors=K)

    print("Training Graph Coloring with GNNPolicy + TB loss...\n")
    final_state = train(
        env=env,
        forward_policy=forward,
        backward_policy=backward,
        loss_fn=loss_fn,
        optimizer=optimizer,
        steps=2000,          
        device=device
    )

    print("\nFinal state:", final_state)

    visualize_coloring(adj, final_state,
                       title=f"GFlowNet Coloring: {filename}")


if __name__ == "__main__":
    main()
