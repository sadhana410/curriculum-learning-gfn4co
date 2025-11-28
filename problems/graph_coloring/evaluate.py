# problems/graph_coloring/evaluate.py

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
from visualization.plot_graph import visualize_coloring

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def count_conflicts(adj, state):
    adj_np = adj.cpu().numpy() if torch.is_tensor(adj) else adj
    colored = state != -1
    same_color = (
        (state[:, None] == state[None, :]) &
        colored[:, None] & colored[None, :]
    )
    return int(np.sum(adj_np * same_color) // 2)


def greedy_rollout(env, policy, adj, device):
    state = env.reset()

    with torch.no_grad():
        while not env.is_terminal(state):
            logits = policy(state, adj, device=device)
            mask = torch.tensor(env.allowed_actions(state), device=device)

            masked = logits.clone()
            masked[mask == 0] = -1e30
            action = masked.argmax().item()

            state, reward, done = env.step(state, action)
            if done:
                break

    return state


def stochastic_rollout(env, policy, adj, device):
    state = env.reset()

    with torch.no_grad():
        while not env.is_terminal(state):
            logits = policy(state, adj, device=device)
            mask = torch.tensor(env.allowed_actions(state), device=device)

            masked = logits.clone()
            masked[mask == 0] = -1e30
            probs = torch.softmax(masked, dim=0)

            action = torch.multinomial(probs, 1).item()
            state, reward, done = env.step(state, action)
            if done:
                break

    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=str, default="myciel3.col")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load graph
    path = os.path.join(DATA_DIR, args.graph)
    adj = load_col_file(path)
    adj = torch.tensor(adj, dtype=torch.float32, device=device)
    N = adj.shape[0]

    # Env computes K = max deg(G)+1
    env = GraphColoringEnv({"adj": adj})
    K = env.K

    print(f"Evaluating {args.graph} with K={K}, N={N}")

    # Policy (weights to be provided)
    policy = GNNPolicy(N, K, hidden_dim=args.hidden_dim).to(device)

    # Greedy rollout
    greedy_state = greedy_rollout(env, policy, adj, device)
    greedy_conf = count_conflicts(adj, greedy_state)
    greedy_colors = len(set(int(c) for c in greedy_state if c != -1))

    print("\nGreedy Evaluation:")
    print(f"  Conflicts: {greedy_conf}")
    print(f"  Colors used: {greedy_colors}/{K}")

    # Best-of-many stochastic
    best_state = None
    best_colors = 999
    best_conf = 999

    for x in range(args.samples):
        s = stochastic_rollout(env, policy, adj, device)
        conf = count_conflicts(adj, s)
        used = len(set(int(c) for c in s if c != -1))

        if (conf < best_conf) or (conf == best_conf and used < best_colors):
            best_state = s.copy()
            best_colors = used
            best_conf = conf

    print(f"\nStochastic Best-of-{args.samples}:")
    print(f"  Conflicts: {best_conf}")
    print(f"  Colors used: {best_colors}/{K}")

    visualize_coloring(adj, best_state, title=f"Evaluation: {args.graph}")


if __name__ == "__main__":
    main()
