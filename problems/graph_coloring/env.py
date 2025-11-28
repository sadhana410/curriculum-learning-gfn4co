import numpy as np
import torch

import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from envs.base_env import BaseEnv


class GraphColoringEnv(BaseEnv):
    """
    Environment for graph coloring for GFlowNets.
    The agent assigns colors to nodes one by one.
    """

    def __init__(self, instance, num_colors=None):
        self.adj = instance["adj"]
        self.N = self.adj.shape[0]

        # Convert adjacency to numpy for fast operations
        if isinstance(self.adj, torch.Tensor):
            self._adj_np = self.adj.cpu().numpy()
        else:
            self._adj_np = np.array(self.adj)

        # If user passes num_colors then use it.
        # Otherwise  K = max deg(G) + 1
        if num_colors is None:
            degrees = self._adj_np.sum(axis=1)
            max_degree = int(degrees.max())
            self.K = max_degree + 1
        else:
            self.K = int(num_colors)

        super().__init__(instance)


    def reset(self):
        """Start with all nodes uncolored (-1)."""
        self.state = -1 * np.ones(self.N, dtype=int)
        return self.state.copy()

    def allowed_actions(self, state):
        """
        Return a flat mask of size N*K indicating which (node, color)
        assignments are valid. 
        """
        mask = np.zeros((self.N, self.K), dtype=np.float32)

        uncolored = (state == -1)

        for node in np.where(uncolored)[0]:
            neighbors = np.where(self._adj_np[node] == 1)[0]
            neighbor_colors = state[neighbors]
            neighbor_colors = neighbor_colors[neighbor_colors != -1]

            valid_colors = np.ones(self.K, dtype=np.float32)
            valid_colors[neighbor_colors] = 0.0

            mask[node] = valid_colors

        return mask.flatten()  # shape = N*K

    def step(self, state, action):
        """
        Apply action (node*K + color) to new state.
        """
        node = action // self.K
        color = action % self.K

        next_state = state.copy()
        next_state[node] = color

        done = self.is_terminal(next_state)
        reward = self.reward(next_state) if done else 0.0

        return next_state, reward, done

    def is_terminal(self, state):
        """
        Terminal if all nodes are coloured or if it is stuck
        """
        if np.all(state != -1):
            return True

        mask = self.allowed_actions(state)
        return mask.sum() == 0


    def reward(self, state):
        """
        Reward for terminal states
        Among conflict-free colorings, exponentially prefer using fewer colors.
        R(s) = exp( -beta * conflicts ) * exp( gamma * (K - colors_used) )
        """
        # Only give reward at terminal states
        if np.any(state == -1):
            return 0.0

        # Conflicts
        colored_mask = state != -1
        same_color = (
            (state[:, None] == state[None, :]) &
            colored_mask[:, None] &
            colored_mask[None, :]
        )
        conflicts = int(np.sum(self._adj_np * same_color) // 2)

        # Colours used
        colors_used = len(set(int(c) for c in state if c != -1))

        beta = 8.0    #conflict penalty
        gamma = 3.0   #colour-savings bonus

        base = np.exp(-beta * conflicts)
        color_bonus = np.exp(gamma * (self.K - colors_used))

        return float(base * color_bonus)

    def encode_state(self, state):
        """
        Encode state into N*K one-hot flattened vector.
        """
        one_hot = np.zeros((self.N, self.K), dtype=np.float32)
        for i in range(self.N):
            if state[i] != -1:
                one_hot[i, state[i]] = 1.0
        return one_hot.flatten()

    def _conflict(self, state, node, color):
        """check if coloring node with color causes a conflict"""
        for nbr in range(self.N):
            if self._adj_np[node, nbr] == 1 and state[nbr] == color:
                return True
        return False
