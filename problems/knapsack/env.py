# problems/knapsack/env.py

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from envs.base_env import BaseEnv


class KnapsackEnv(BaseEnv):
    """
    Knapsack environment for GFlowNet.
    
    State: binary array of length N indicating which items are selected.
            -1 means item not yet decided, 0 means not selected, 1 means selected.
    Action: index of item to decide (0 to 2*N-1)
            action < N means select item (action), action >= N means skip item (action - N)
    """
    
    def __init__(self, instance):
        """
        instance should contain:
            - 'profits': array of item profits/values
            - 'weights': array of item weights
            - 'capacity': knapsack capacity
        """
        self.profits = np.array(instance['profits'], dtype=np.float32)
        self.weights = np.array(instance['weights'], dtype=np.float32)
        self.capacity = instance['capacity']
        self.N = len(self.profits)
        
        # Precompute for reward normalization
        self.max_profit = np.sum(self.profits)
        
        super().__init__(instance)
    
    def reset(self):
        # -1 means undecided, 0 means not selected, 1 means selected
        self.state = -1 * np.ones(self.N, dtype=np.int32)
        return self.state.copy()
    
    def allowed_actions(self, state):
        """
        Actions: 0 to N-1 = select item i, N to 2N-1 = skip item i
        Only undecided items can be acted upon.
        Selecting an item is only valid if it fits in remaining capacity.
        """
        mask = np.zeros(2 * self.N, dtype=np.float32)
        
        # Current weight used
        selected = state == 1
        current_weight = np.sum(self.weights[selected])
        remaining_capacity = self.capacity - current_weight
        
        # For each undecided item
        undecided = state == -1
        
        for i in np.where(undecided)[0]:
            # Can always skip (action = N + i)
            mask[self.N + i] = 1.0
            # Can select only if it fits (action = i)
            if self.weights[i] <= remaining_capacity:
                mask[i] = 1.0
        
        return mask
    
    def step(self, state, action):
        next_state = state.copy()
        
        if action < self.N:
            # Select item
            next_state[action] = 1
        else:
            # Skip item
            next_state[action - self.N] = 0
        
        done = self.is_terminal(next_state)
        reward = self.reward(next_state) if done else 0.0
        return next_state, reward, done
    
    def is_terminal(self, state):
        # Terminal if all items are decided
        if np.all(state != -1):
            return True
        # Also terminal if no valid actions remain
        mask = self.allowed_actions(state)
        return mask.sum() == 0
    
    def reward(self, state):
        """
        Reward based on total profit of selected items.
        Higher profit = higher reward.
        """
        selected = state == 1
        total_profit = np.sum(self.profits[selected])
        total_weight = np.sum(self.weights[selected])
        
        # Check feasibility
        if total_weight > self.capacity:
            # Infeasible - very small reward
            return 0.001
        
        # Reward proportional to profit (normalized and exponentiated for GFN)
        # Using exp to make reward positive and emphasize better solutions
        profit_ratio = total_profit / (self.max_profit + 1e-8)
        return float(np.exp(2.0 * profit_ratio))
    
    def encode_state(self, state):
        """Encode state as feature vector."""
        # One-hot encode: undecided, not selected, selected
        encoding = np.zeros((self.N, 3), dtype=np.float32)
        encoding[state == -1, 0] = 1.0  # undecided
        encoding[state == 0, 1] = 1.0   # not selected
        encoding[state == 1, 2] = 1.0   # selected
        return encoding.flatten()
    
    def get_profit(self, state):
        """Get total profit of current selection."""
        selected = state == 1
        return np.sum(self.profits[selected])
    
    def get_weight(self, state):
        """Get total weight of current selection."""
        selected = state == 1
        return np.sum(self.weights[selected])
