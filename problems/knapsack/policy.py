# models/knapsack_policy.py

import torch
import torch.nn as nn
import numpy as np


class KnapsackPolicy(nn.Module):
    """
    MLP policy for knapsack problem.
    
    Input: state encoding + item features (profits, weights, capacity info)
    Output: logits over 2*N actions (select or skip each item)
    """
    
    def __init__(self, num_items: int, hidden_dim: int = 128):
        super().__init__()
        self.N = num_items
        self.hidden_dim = hidden_dim
        
        # Input: for each item: (undecided, not_selected, selected, profit_norm, weight_norm, fits)
        # Plus global features: (capacity_used_ratio, items_decided_ratio)
        self.item_features = 6
        self.global_features = 2
        input_dim = self.N * self.item_features + self.global_features
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * num_items),  # select or skip each item
        )
        
        # Cache for instance data
        self._profits = None
        self._weights = None
        self._capacity = None
    
    def set_instance(self, profits, weights, capacity):
        """Set the knapsack instance data for feature computation."""
        self._profits = profits
        self._weights = weights
        self._capacity = capacity
        # Normalize
        self._profits_norm = profits / (profits.max() + 1e-8)
        self._weights_norm = weights / (capacity + 1e-8)
    
    def forward(self, state, device="cpu"):
        """
        state: numpy array of length N with values in {-1, 0, 1}
        returns: logits of shape (2*N,)
        """
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.long, device=device)
        else:
            state = state.to(device).long()
        
        # Build feature vector
        features = []
        
        # Per-item features
        for i in range(self.N):
            # One-hot state encoding
            undecided = 1.0 if state[i] == -1 else 0.0
            not_selected = 1.0 if state[i] == 0 else 0.0
            selected = 1.0 if state[i] == 1 else 0.0
            
            # Item properties (normalized)
            profit_norm = self._profits_norm[i]
            weight_norm = self._weights_norm[i]
            
            # Can this item still fit?
            current_weight = (self._weights * (state == 1).float().cpu().numpy()).sum()
            remaining = self._capacity - current_weight
            fits = 1.0 if self._weights[i] <= remaining else 0.0
            
            features.extend([undecided, not_selected, selected, profit_norm, weight_norm, fits])
        
        # Global features
        selected_mask = (state == 1).float()
        current_weight = (torch.tensor(self._weights, device=device) * selected_mask).sum()
        capacity_used = current_weight / (self._capacity + 1e-8)
        items_decided = (state != -1).float().mean()
        
        features.extend([capacity_used.item(), items_decided.item()])
        
        # Forward pass
        x = torch.tensor(features, dtype=torch.float32, device=device)
        logits = self.net(x)
        
        return logits
