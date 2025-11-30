# models/knapsack_policy.py

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        
        # Flow head for DB/SubTB
        self.flow_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
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
    
    def get_features(self, state, device="cpu"):
        """
        Compute features for state.
        Returns: (B, input_dim)
        """
        # Ensure tensor on device
        if isinstance(state, np.ndarray):
            state = torch.as_tensor(state, dtype=torch.long, device=device)
        else:
            state = state.to(device).long()
            
        # Handle batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)  # (1, N)
            
        B, N = state.shape
        
        # Prepare instance data tensors (cached on device if needed)
        if not hasattr(self, '_profits_t') or self._profits_t.device != device:
            self._profits_t = torch.as_tensor(self._profits, dtype=torch.float32, device=device)
            self._weights_t = torch.as_tensor(self._weights, dtype=torch.float32, device=device)
            self._profits_norm_t = torch.as_tensor(self._profits_norm, dtype=torch.float32, device=device)
            self._weights_norm_t = torch.as_tensor(self._weights_norm, dtype=torch.float32, device=device)
        
        # 1. One-hot state encoding: (B, N, 3)
        state_indices = state + 1
        state_onehot = F.one_hot(state_indices, num_classes=3).float() # (B, N, 3)
        
        # 2. Item properties: (B, N, 2)
        props = torch.stack([self._profits_norm_t, self._weights_norm_t], dim=1) # (N, 2)
        props_batch = props.unsqueeze(0).expand(B, -1, -1) # (B, N, 2)
        
        # 3. Fits feature: (B, N, 1)
        selected_mask = (state == 1).float() # (B, N)
        current_weight = (selected_mask * self._weights_t.unsqueeze(0)).sum(dim=1) # (B,)
        remaining = self._capacity - current_weight # (B,)
        
        fits = (self._weights_t.unsqueeze(0) <= remaining.unsqueeze(1)).float() # (B, N)
        fits = fits.unsqueeze(-1) # (B, N, 1)
        
        # Concatenate per-item features: (B, N, 6)
        item_feats = torch.cat([state_onehot, props_batch, fits], dim=2)
        item_feats_flat = item_feats.view(B, -1) # (B, N*6)
        
        # 4. Global features: (B, 2)
        cap_ratio = current_weight / (self._capacity + 1e-8) # (B,)
        decided_mask = (state != -1).float()
        decided_ratio = decided_mask.mean(dim=1) # (B,)
        
        global_feats = torch.stack([cap_ratio, decided_ratio], dim=1) # (B, 2)
        
        # Combine all features: (B, input_dim)
        x = torch.cat([item_feats_flat, global_feats], dim=1)
        return x

    def forward(self, state, device="cpu"):
        """
        state: numpy array or tensor
            shape (N,)      -> returns (2*N,)
            shape (B, N)    -> returns (B, 2*N)
        returns: logits
        """
        x = self.get_features(state, device)
        logits = self.net(x)
        
        if state.dim() == 1 or (isinstance(state, np.ndarray) and state.ndim == 1):
            return logits.squeeze(0)
        return logits

    def predict_flow(self, state, device="cpu"):
        """
        Predict Log Flow F(s).
        Returns: (B, 1) or (1,)
        """
        x = self.get_features(state, device)
        log_flow = self.flow_head(x)
        
        if state.dim() == 1 or (isinstance(state, np.ndarray) and state.ndim == 1):
            return log_flow.squeeze(0)
        return log_flow.squeeze(-1)
