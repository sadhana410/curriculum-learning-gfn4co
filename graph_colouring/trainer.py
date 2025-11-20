import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from graph_colouring import GraphColoringEnv
from trajectorybalance import TrajectoryBalance   # minimal TB version


# ---------------------------------------------------------------------
# Visualization helper
# ---------------------------------------------------------------------
def visualize_coloring(adj, coloring, title="Graph Coloring"):
    """
    adj: NxN adjacency matrix
    coloring: final state (list of colors)
    """
    N = adj.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(N))

    # Add edges
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i, j] == 1:
                G.add_edge(i, j)

    # Color nodes
    node_colors = []
    for c in coloring:
        if c == -1:
            node_colors.append("lightgray")
        else:
            node_colors.append(f"C{c}")

    plt.figure(figsize=(5, 5))
    pos = nx.spring_layout(G, seed=42)

    nx.draw(
        G, pos,
        node_color=node_colors,
        with_labels=True,
        node_size=700,
        font_size=14,
        linewidths=1,
        font_weight="bold"
    )
    plt.title(title)
    plt.show()


# ---------------------------------------------------------------------
# Simple MLP policy network
# ---------------------------------------------------------------------
class PolicyNet(nn.Module):
    def __init__(self, input_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------
# Sample one trajectory using forward policy
# ---------------------------------------------------------------------
def sample_trajectory(env, forward_policy, device):
    state = env.reset()
    traj_states = []
    traj_actions = []

    done = False
    while not done:
        traj_states.append(state.copy())   # state BEFORE action

        s_enc = torch.tensor(env.encode_state(state), dtype=torch.float32).to(device)
        logits = forward_policy(s_enc)

        # Mask invalid actions
        mask = torch.tensor(env.allowed_actions(state), dtype=torch.float32).to(device)
        masked_logits = logits + (mask + 1e-8).log()

        probs = torch.softmax(masked_logits, dim=0)
        action = torch.multinomial(probs, 1).item()

        # Environment step
        next_state, reward, done = env.step(state, action)
        traj_actions.append(action)

        state = next_state

    # append the TRUE terminal state
    traj_states.append(state.copy())

    return traj_states, traj_actions, reward


# ---------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------
def main():
    device = torch.device("cpu")

    # Simple 4-node cycle graph
    adj = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
    ])
    instance = {"adj": adj}

    N = adj.shape[0]
    K = 3
    input_dim = N * K
    action_dim = N * K

    forward_policy = PolicyNet(input_dim, action_dim).to(device)
    backward_policy = PolicyNet(input_dim, action_dim).to(device)

    # Minimal Trajectory Balance loss
    loss_fn = TrajectoryBalance(
        forward_policy=forward_policy,
        backward_policy=backward_policy
    )

    optimizer = torch.optim.Adam(
        list(forward_policy.parameters()) +
        list(backward_policy.parameters()) +
        list(loss_fn.parameters()),
        lr=1e-3
    )

    env = GraphColoringEnv(instance, num_colors=K)

    print("Training Graph Coloring with TB loss...")

    last_terminal_state = None

    for step in range(2000):
        traj_states, traj_actions, reward = sample_trajectory(env, forward_policy, device)
        last_terminal_state = traj_states[-1]  # correct terminal state

        # -------------------------------
        # Compute forward log-probability
        # -------------------------------
        logprobs_f = 0
        for i in range(len(traj_actions)):          # iterate over ACTIONS
            s = torch.tensor(env.encode_state(traj_states[i]), dtype=torch.float32)
            logits = forward_policy(s)
            mask = torch.tensor(env.allowed_actions(traj_states[i]), dtype=torch.float32)
            masked = logits + (mask + 1e-8).log()
            probs = torch.softmax(masked, dim=0)
            logprobs_f += torch.log(probs[traj_actions[i]] + 1e-8)

        # -------------------------------
        # Compute backward log-probability
        # -------------------------------
        logprobs_b = 0
        for i in reversed(range(len(traj_actions))):   # reverse over ACTIONS
            s = torch.tensor(env.encode_state(traj_states[i]), dtype=torch.float32)
            logits = backward_policy(s)
            mask = torch.tensor(env.allowed_actions(traj_states[i]), dtype=torch.float32)
            masked = logits + (mask + 1e-8).log()
            probs = torch.softmax(masked, dim=0)
            logprobs_b += torch.log(probs[traj_actions[i]] + 1e-8)

        # log reward
        logreward = torch.log(torch.tensor([reward], dtype=torch.float32) + 1e-8)

        # TB loss
        loss = loss_fn(logprobs_f, logprobs_b, logreward)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            print(f"[{step}] loss={loss.item():.4f}, reward={reward:.4f}")

    print("Training complete.")
    print("Final sampled coloring:", last_terminal_state)

    # Visualize the graph
    visualize_coloring(adj, last_terminal_state, title="Final GFlowNet Coloring")


if __name__ == "__main__":
    main()
