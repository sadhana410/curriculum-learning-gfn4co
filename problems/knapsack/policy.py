# problems/knapsack/policy.py

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


class ConditionalKnapsackPolicy(nn.Module):
    """
    Conditional MLP policy for knapsack problem that generalizes across different instances.
    
    This policy is conditioned on the instance data (profits, weights, capacity) and can handle
    variable-sized instances. It uses a shared backbone with separate forward/backward heads.
    
    The key difference from KnapsackPolicy is that instance data is passed as input rather than
    being cached, allowing the same policy to work with multiple instances.
    
    Input features per item:
        - State encoding: (undecided, not_selected, selected) one-hot
        - Profit normalized by max profit
        - Weight normalized by capacity
        - Fits in remaining capacity (binary)
    Global features:
        - Capacity used ratio
        - Items decided ratio
    
    Output: logits over 2*N actions (select or skip each item)
    """
    
    def __init__(self, hidden_dim: int = 128, num_layers: int = 3, head_hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.head_hidden_dim = head_hidden_dim
        
        # Input: for each item: (undecided, not_selected, selected, profit_norm, weight_norm, fits) = 6
        # Plus global features: (capacity_used_ratio, items_decided_ratio) = 2
        self.item_features = 6
        self.global_features = 2
        
        # ===== SHARED BACKBONE =====
        # First layer handles variable input size
        self.item_encoder = nn.Sequential(
            nn.Linear(self.item_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Global feature encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(self.global_features, hidden_dim // 2),
            nn.ReLU(),
        )
        
        # Attention layers for variable-size aggregation
        self.attention_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attention_layers.append(AttentionLayer(hidden_dim))
        
        # ===== FORWARD POLICY HEAD =====
        self.forward_head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, 2),  # select or skip per item
        )
        
        # ===== BACKWARD POLICY HEAD =====
        self.backward_head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, 2),  # undo select or undo skip per item
        )
        
        # ===== FLOW HEAD (for DB/SubTB) =====
        self.flow_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, head_hidden_dim),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, 1)
        )
        
        # Track which head to use
        self._mode = 'forward'
    
    def get_features(self, state, profits, weights, capacity, device="cpu"):
        """
        Compute features for state given instance data.
        
        Args:
            state: (N,) or (B, N) tensor of item states {-1, 0, 1}
            profits: (N,) tensor of item profits
            weights: (N,) tensor of item weights
            capacity: scalar capacity
            
        Returns:
            item_feats: (B, N, item_features) per-item features
            global_feats: (B, global_features) global features
        """
        # Convert to tensor on device
        if isinstance(state, np.ndarray):
            state = torch.as_tensor(state, dtype=torch.long, device=device)
        else:
            state = state.to(device).long()
            
        if isinstance(profits, np.ndarray):
            profits = torch.as_tensor(profits, dtype=torch.float32, device=device)
        else:
            profits = profits.to(device).float()
            
        if isinstance(weights, np.ndarray):
            weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
        else:
            weights = weights.to(device).float()
        
        # Ensure batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)  # (1, N)
            
        B, N = state.shape
        
        # Normalize profits and weights
        profits_norm = profits / (profits.max() + 1e-8)  # (N,)
        weights_norm = weights / (capacity + 1e-8)  # (N,)
        
        # 1. One-hot state encoding: (B, N, 3)
        state_indices = state + 1  # -1 -> 0, 0 -> 1, 1 -> 2
        state_onehot = F.one_hot(state_indices, num_classes=3).float()  # (B, N, 3)
        
        # 2. Item properties: (N, 2) -> (B, N, 2)
        props = torch.stack([profits_norm, weights_norm], dim=0).T  # (N, 2)
        props_batch = props.unsqueeze(0).expand(B, -1, -1)  # (B, N, 2)
        
        # 3. Fits feature: (B, N, 1)
        selected_mask = (state == 1).float()  # (B, N)
        current_weight = (selected_mask * weights.unsqueeze(0)).sum(dim=1)  # (B,)
        remaining = capacity - current_weight  # (B,)
        
        fits = (weights.unsqueeze(0) <= remaining.unsqueeze(1)).float()  # (B, N)
        fits = fits.unsqueeze(-1)  # (B, N, 1)
        
        # Concatenate per-item features: (B, N, 6)
        item_feats = torch.cat([state_onehot, props_batch, fits], dim=2)
        
        # 4. Global features: (B, 2)
        cap_ratio = current_weight / (capacity + 1e-8)  # (B,)
        decided_mask = (state != -1).float()
        decided_ratio = decided_mask.mean(dim=1)  # (B,)
        
        global_feats = torch.stack([cap_ratio, decided_ratio], dim=1)  # (B, 2)
        
        return item_feats, global_feats
    
    def get_item_embeddings(self, state, profits, weights, capacity, device="cpu"):
        """
        Compute item embeddings using shared backbone.
        
        Returns:
            h: (B, N, H) item embeddings
            global_h: (B, H//2) global embedding
        """
        item_feats, global_feats = self.get_features(state, profits, weights, capacity, device)
        B, N, _ = item_feats.shape
        
        # Encode items
        h = self.item_encoder(item_feats)  # (B, N, H)
        
        # Encode global features
        global_h = self.global_encoder(global_feats)  # (B, H//2)
        
        # Apply attention layers
        for attn_layer in self.attention_layers:
            h = attn_layer(h)  # (B, N, H)
        
        return h, global_h
    
    def set_mode(self, mode: str):
        """Set policy mode: 'forward' or 'backward'."""
        assert mode in ('forward', 'backward'), f"Mode must be 'forward' or 'backward', got {mode}"
        self._mode = mode
    
    def forward_logits(self, state, profits, weights, capacity, device="cpu"):
        """
        Compute forward policy logits.
        
        Returns:
            logits: (2*N,) or (B, 2*N) action logits
        """
        single_input = False
        if isinstance(state, np.ndarray) and state.ndim == 1:
            single_input = True
        elif isinstance(state, torch.Tensor) and state.dim() == 1:
            single_input = True
            
        h, _ = self.get_item_embeddings(state, profits, weights, capacity, device)
        B, N, H = h.shape
        
        # Per-item logits via forward head: (B, N, 2)
        item_logits = self.forward_head(h)
        
        # Reshape to (B, 2*N): first N are select, next N are skip
        logits = item_logits.view(B, N * 2)
        
        if single_input:
            return logits.squeeze(0)  # (2*N,)
        return logits
    
    def backward_logits(self, state, profits, weights, capacity, device="cpu"):
        """
        Compute backward policy logits.
        
        Returns:
            logits: (2*N,) or (B, 2*N) action logits
        """
        single_input = False
        if isinstance(state, np.ndarray) and state.ndim == 1:
            single_input = True
        elif isinstance(state, torch.Tensor) and state.dim() == 1:
            single_input = True
            
        h, _ = self.get_item_embeddings(state, profits, weights, capacity, device)
        B, N, H = h.shape
        
        # Per-item logits via backward head: (B, N, 2)
        item_logits = self.backward_head(h)
        
        # Reshape to (B, 2*N): first N are undo-select, next N are undo-skip
        logits = item_logits.view(B, N * 2)
        
        if single_input:
            return logits.squeeze(0)  # (2*N,)
        return logits
    
    def forward(self, state, profits, weights, capacity, device="cpu"):
        """
        Default forward pass - uses current mode.
        """
        if self._mode == 'forward':
            return self.forward_logits(state, profits, weights, capacity, device)
        else:
            return self.backward_logits(state, profits, weights, capacity, device)
    
    def predict_flow(self, state, profits, weights, capacity, device="cpu"):
        """
        Predict log flow F(s) for DB/SubTB losses.
        
        Returns:
            log_flow: (B,) or scalar
        """
        single_input = False
        if isinstance(state, np.ndarray) and state.ndim == 1:
            single_input = True
        elif isinstance(state, torch.Tensor) and state.dim() == 1:
            single_input = True
            
        h, global_h = self.get_item_embeddings(state, profits, weights, capacity, device)
        
        # Global pooling (mean + max)
        h_mean = h.mean(dim=1)  # (B, H)
        h_max = h.max(dim=1)[0]  # (B, H)
        h_graph = h_mean + h_max  # (B, H)
        
        # Combine with global features
        combined = torch.cat([h_graph, global_h], dim=1)  # (B, H + H//2)
        
        log_flow = self.flow_head(combined).squeeze(-1)  # (B,)
        
        if single_input:
            return log_flow.squeeze(0)
        return log_flow


class ConditionalKnapsackPolicyWrapper(nn.Module):
    """
    Wrapper to expose a specific head (forward or backward) of a ConditionalKnapsackPolicy
    as a standalone policy. This allows using the shared policy with existing
    training code that expects separate forward/backward policy objects.
    """
    
    def __init__(self, shared_policy: ConditionalKnapsackPolicy, mode: str):
        super().__init__()
        self.shared_policy = shared_policy
        self.mode = mode
        assert mode in ('forward', 'backward'), f"Mode must be 'forward' or 'backward'"
    
    def forward(self, state, profits, weights, capacity, device="cpu"):
        if self.mode == 'forward':
            return self.shared_policy.forward_logits(state, profits, weights, capacity, device)
        else:
            return self.shared_policy.backward_logits(state, profits, weights, capacity, device)
    
    def predict_flow(self, state, profits, weights, capacity, device="cpu"):
        return self.shared_policy.predict_flow(state, profits, weights, capacity, device)
    
    def parameters(self, recurse=True):
        return self.shared_policy.parameters(recurse)


class AttentionLayer(nn.Module):
    """
    Self-attention layer for variable-size item sets.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, h):
        """
        Args:
            h: (B, N, H) item embeddings
            
        Returns:
            h_new: (B, N, H) updated embeddings
        """
        # Self-attention with residual
        attn_out, _ = self.attention(h, h, h)
        h = self.layer_norm(h + attn_out)
        
        # FFN with residual
        ffn_out = self.ffn(h)
        h = self.layer_norm2(h + ffn_out)
        
        return h
