import numpy as np
import torch

import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from envs.base_env import BaseEnv

class GraphColoringEnv(BaseEnv):
    def __init__(self, instance, num_colors=3, chromatic_number=None):
        self.adj = instance["adj"]
        self.N = self.adj.shape[0]
        self.K = num_colors
        # Chromatic number is the minimum colors needed (if known)
        self.chromatic_number = chromatic_number if chromatic_number else num_colors
        # Pre-compute numpy adjacency for faster operations
        if isinstance(self.adj, torch.Tensor):
            self._adj_np = self.adj.cpu().numpy()
        else:
            self._adj_np = np.array(self.adj)
        super().__init__(instance)

    def reset(self):
        self.state = -1 * np.ones(self.N, dtype=int)
        return self.state.copy()

    def allowed_actions(self, state):
        """Vectorized allowed actions computation."""
        mask = np.zeros((self.N, self.K), dtype=np.float32)
        
        # Find uncolored nodes
        uncolored = state == -1
        
        for node in np.where(uncolored)[0]:
            # Get neighbors of this node
            neighbors = np.where(self._adj_np[node] == 1)[0]
            # Get colors used by neighbors
            neighbor_colors = state[neighbors]
            neighbor_colors = neighbor_colors[neighbor_colors != -1]
            # All colors except those used by neighbors are valid
            valid_colors = np.ones(self.K, dtype=np.float32)
            valid_colors[neighbor_colors] = 0.0
            mask[node] = valid_colors
        
        return mask.flatten()

    def step(self, state, action):
        node = action // self.K
        color = action % self.K
        next_state = state.copy()
        next_state[node] = color
        done = self.is_terminal(next_state)
        reward = self.reward(next_state) if done else 0.0
        return next_state, reward, done

    def is_terminal(self, state):
        # Terminal if all nodes are colored
        if np.all(state != -1):
            return True
        # Also terminal if no valid actions remain (stuck)
        mask = self.allowed_actions(state)
        return mask.sum() == 0

    def reward(self, state):
        # Vectorized conflict counting
        colored_mask = state != -1
        same_color = (state[:, None] == state[None, :]) & colored_mask[:, None] & colored_mask[None, :]
        conflicts = int(np.sum(self._adj_np * same_color) // 2)
        
        # Count colored nodes and colors used
        colored = np.sum(colored_mask)
        colors_used = len(set(c for c in state if c != -1)) if colored > 0 else 0
        # if colored == self.N and conflicts == 0:
        #     return float(np.exp(-colors_used))
        # else:
        #     return float(0.0)
        if colored == self.N and conflicts == 0:
            # Valid complete coloring - reward fewer colors
            # Scale: exp(N/colors_used) so logreward ≈ N/colors_used (same scale as logprobs)
            # For N=23, colors=5: reward = exp(4.6) ≈ 100, logreward ≈ 4.6
            # For N=23, colors=23: reward = exp(1) ≈ 2.7, logreward ≈ 1
            return float(np.exp(self.N / colors_used))
        elif colored == self.N and conflicts > 0:
            # Complete but invalid - penalize conflicts
            return float(np.exp(-conflicts))
        else:
            # Partial coloring - small reward
            progress = colored / self.N
            return float(np.exp(-10 + progress * 5))  # logreward in [-10, -5] range

    def encode_state(self, state):
        one_hot = np.zeros((self.N, self.K), dtype=np.float32)
        for i in range(self.N):
            if state[i] != -1:
                one_hot[i, state[i]] = 1.0
        return one_hot.flatten()

    def _conflict(self, state, node, color):
        for nbr in range(self.N):
            if self.adj[node, nbr] == 1 and state[nbr] == color:
                return True
        return False
