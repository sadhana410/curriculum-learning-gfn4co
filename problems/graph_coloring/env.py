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
        
        if colored == self.N and conflicts == 0:
            # Valid complete coloring - STRONG exponential reward for fewer colors
            # Using chromatic number (4) vs all colors (5) should be a big difference
            colors_saved = self.K - colors_used
            # exp(2 * colors_saved) makes 4 colors give exp(2)=7.4x more than 5 colors
            return float(np.exp(2.0 * colors_saved))
        elif colored == self.N and conflicts > 0:
            # Complete but invalid - very small reward
            return float(0.01 * np.exp(-conflicts))
        else:
            # Partial coloring - reward based on progress AND color efficiency
            # Bonus for using fewer colors so far (encourages color reuse during exploration)
            progress = colored / self.N
            # Expected colors at this point if optimal: roughly (colored/N) * chromatic_number
            expected_colors = max(1, (colored / self.N) * self.chromatic_number)
            color_efficiency = expected_colors / max(1, colors_used)  # >1 if using fewer than expected
            return float(0.01 * progress * color_efficiency * np.exp(-conflicts))

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
