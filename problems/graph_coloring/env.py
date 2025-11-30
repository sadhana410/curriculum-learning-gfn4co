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

    def step_batch(self, states, actions):
        """
        Vectorized step for a batch of states.
        states: (B, N) numpy array
        actions: (B,) numpy array
        """
        B, N = states.shape
        K = self.K
        
        nodes = actions // K
        colors = actions % K
        
        next_states = states.copy()
        # Advanced indexing for batch update
        # next_states[range(B), nodes] = colors
        # But nodes is (B,), colors is (B,)
        next_states[np.arange(B), nodes] = colors
        
        # Check done (only fully colored check here, stuck check handled by sampler)
        dones = (next_states != -1).all(axis=1)
        
        # Compute rewards for done states
        rewards = np.zeros(B, dtype=np.float32)
        if dones.any():
            done_states = next_states[dones]
            rewards[dones] = self.reward_batch(done_states)
            
        return next_states, rewards, dones

    def is_terminal(self, state):
        # Terminal if all nodes are colored
        if np.all(state != -1):
            return True
        # Also terminal if no valid actions remain (stuck)
        mask = self.allowed_actions(state)
        return mask.sum() == 0

    def reward(self, state):
        # Single state reward
        colored_mask = state != -1
        same_color = (state[:, None] == state[None, :]) & colored_mask[:, None] & colored_mask[None, :]
        conflicts = int(np.sum(self._adj_np * same_color) // 2)
        
        colored = np.sum(colored_mask)
        colors_used = len(set(c for c in state if c != -1)) if colored > 0 else 0
        missing = self.N - colored

        # logR as a smooth function
        alpha = 1.0   # penalty per conflict
        beta  = 0.5   # penalty per color used
        gamma = 0.2   # penalty per uncolored node

        logR = -alpha * conflicts - beta * colors_used - gamma * missing
        
        # Scale by N to match log_pf magnitude
        logR = logR * self.N

        return float(logR)

    def reward_batch(self, states):
        """Vectorized reward for batch of states (B, N)."""
        B, N = states.shape
        
        colored_mask = (states != -1) # (B, N)
        
        # same_color: (B, N, N)
        same_color = (states[:, :, None] == states[:, None, :]) & \
                     colored_mask[:, :, None] & \
                     colored_mask[:, None, :]
                     
        # Conflicts: sum(adj * same_color)
        adj_broadcast = self._adj_np[None, :, :]
        conflicts = np.sum(adj_broadcast * same_color, axis=(1, 2)) // 2 # (B,)
        
        # Colors used: (B,)
        colors_used = np.zeros(B, dtype=np.float32)
        for k in range(self.K):
            has_color = (states == k).any(axis=1)
            colors_used += has_color
            
        colored_counts = colored_mask.sum(axis=1)
        missing = N - colored_counts
        
        # Parameters
        alpha = 1.0
        beta = 0.5
        gamma = 0.2
        
        logR = -alpha * conflicts - beta * colors_used - gamma * missing
        
        # Scale by N
        logR = logR * N
        
        return logR

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
