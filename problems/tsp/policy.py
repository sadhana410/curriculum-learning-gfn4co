# problems/tsp/policy.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TSPPolicy(nn.Module):
    """
    Attention-based policy for TSP.
    
    Uses a transformer-style architecture that is well-suited for
    sequential decision making in TSP.
    
    Input: state encoding + city coordinates + distance features
    Output: logits over N actions (which city to visit next)
    """
    
    def __init__(self, num_cities: int, hidden_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.N = num_cities
        self.hidden_dim = hidden_dim
        
        # Input features per city:
        # - visited (1)
        # - position in tour normalized (1)
        # - x, y coordinates (2)
        # - is current city (1)
        # - distance to current city (1)
        self.input_dim = 6
        
        # City encoder
        self.city_encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Self-attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
            for _ in range(3)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(3)
        ])
        
        # Output head for action logits
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Flow head for DB/SubTB
        self.flow_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Cache for instance data
        self._coords = None
        self._distance_matrix = None
    
    def set_instance(self, coords, distance_matrix):
        """Set the TSP instance data for feature computation."""
        self._coords = coords
        self._distance_matrix = distance_matrix
        
        # Normalize coordinates
        coords_min = coords.min(axis=0)
        coords_max = coords.max(axis=0)
        self._coords_norm = (coords - coords_min) / (coords_max - coords_min + 1e-8)
        
        # Normalize distances
        self._dist_max = distance_matrix.max()
        self._distance_matrix_norm = distance_matrix / (self._dist_max + 1e-8)
    
    def get_features(self, state, device="cpu"):
        """
        Compute features for state.
        
        Args:
            state: (N,) or (B, N) array
            
        Returns:
            features: (B, N, input_dim) tensor
        """
        if isinstance(state, np.ndarray):
            state = torch.as_tensor(state, dtype=torch.long, device=device)
        else:
            state = state.to(device).long()
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        B, N = state.shape
        
        # Prepare instance data
        if not hasattr(self, '_coords_t') or self._coords_t.device != device:
            self._coords_t = torch.as_tensor(self._coords_norm, dtype=torch.float32, device=device)
            self._dist_t = torch.as_tensor(self._distance_matrix_norm, dtype=torch.float32, device=device)
        
        features = torch.zeros(B, N, self.input_dim, device=device)
        
        # 1. Visited flag
        visited = (state != -1).float()  # (B, N)
        features[:, :, 0] = visited
        
        # 2. Position in tour (normalized)
        position = state.float()
        position = torch.where(state == -1, torch.zeros_like(position), position / (N - 1))
        features[:, :, 1] = position
        
        # 3. Coordinates
        features[:, :, 2:4] = self._coords_t.unsqueeze(0).expand(B, -1, -1)
        
        # 4. Is current city (last visited)
        # Find the city with highest position value
        max_pos = state.max(dim=1, keepdim=True)[0]  # (B, 1)
        is_current = (state == max_pos).float()  # (B, N)
        # Handle initial state where only city 0 is visited
        is_current = is_current * visited
        features[:, :, 4] = is_current
        
        # 5. Distance to current city
        # Get current city index for each batch
        current_city = state.argmax(dim=1)  # (B,) - city with highest position
        dist_to_current = self._dist_t[current_city]  # (B, N)
        features[:, :, 5] = dist_to_current
        
        return features
    
    def forward(self, state, device="cpu"):
        """
        Compute action logits.
        
        Args:
            state: (N,) or (B, N) array
            
        Returns:
            logits: (N,) or (B, N) tensor
        """
        single_input = False
        if isinstance(state, np.ndarray) and state.ndim == 1:
            single_input = True
        elif isinstance(state, torch.Tensor) and state.dim() == 1:
            single_input = True
        
        features = self.get_features(state, device)  # (B, N, input_dim)
        
        # Encode cities
        h = self.city_encoder(features)  # (B, N, H)
        
        # Self-attention layers
        for attn, ln in zip(self.attention_layers, self.layer_norms):
            h_attn, _ = attn(h, h, h)
            h = ln(h + h_attn)
        
        # Output logits
        logits = self.output_head(h).squeeze(-1)  # (B, N)
        
        if single_input:
            return logits.squeeze(0)
        return logits
    
    def predict_flow(self, state, device="cpu"):
        """
        Predict log flow F(s).
        
        Returns:
            log_flow: (B,) or scalar
        """
        single_input = False
        if isinstance(state, np.ndarray) and state.ndim == 1:
            single_input = True
        elif isinstance(state, torch.Tensor) and state.dim() == 1:
            single_input = True
        
        features = self.get_features(state, device)
        h = self.city_encoder(features)
        
        for attn, ln in zip(self.attention_layers, self.layer_norms):
            h_attn, _ = attn(h, h, h)
            h = ln(h + h_attn)
        
        # Global pooling
        h_mean = h.mean(dim=1)  # (B, H)
        h_max = h.max(dim=1)[0]  # (B, H)
        h_graph = h_mean + h_max
        
        log_flow = self.flow_head(h_graph).squeeze(-1)
        
        if single_input:
            return log_flow.squeeze(0)
        return log_flow


class ConditionalTSPPolicy(nn.Module):
    """
    Conditional attention-based policy for TSP that generalizes across instances.
    
    This policy is conditioned on the instance data (coordinates, distances)
    and can handle variable-sized instances.
    """
    
    def __init__(self, hidden_dim: int = 128, num_heads: int = 4, num_layers: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Input features per city:
        # - visited (1)
        # - position in tour normalized (1)
        # - x, y coordinates (2)
        # - is current city (1)
        # - distance to current city (1)
        self.input_dim = 6
        
        # City encoder
        self.city_encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Self-attention layers
        self.attention_layers = nn.ModuleList([
            AttentionBlock(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Forward policy head
        self.forward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Backward policy head
        self.backward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Flow head for DB/SubTB
        self.flow_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self._mode = 'forward'
    
    def get_features(self, state, coords, distance_matrix, device="cpu"):
        """
        Compute features for state given instance data.
        
        Args:
            state: (N,) or (B, N) tensor
            coords: (N, 2) array of coordinates
            distance_matrix: (N, N) distance matrix
            
        Returns:
            features: (B, N, input_dim) tensor
        """
        if isinstance(state, np.ndarray):
            state = torch.as_tensor(state, dtype=torch.long, device=device)
        else:
            state = state.to(device).long()
        
        if isinstance(coords, np.ndarray):
            coords = torch.as_tensor(coords, dtype=torch.float32, device=device)
        else:
            coords = coords.to(device).float()
        
        if isinstance(distance_matrix, np.ndarray):
            distance_matrix = torch.as_tensor(distance_matrix, dtype=torch.float32, device=device)
        else:
            distance_matrix = distance_matrix.to(device).float()
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        B, N = state.shape
        
        # Normalize coordinates
        coords_min = coords.min(dim=0)[0]
        coords_max = coords.max(dim=0)[0]
        coords_norm = (coords - coords_min) / (coords_max - coords_min + 1e-8)
        
        # Normalize distances
        dist_max = distance_matrix.max()
        dist_norm = distance_matrix / (dist_max + 1e-8)
        
        features = torch.zeros(B, N, self.input_dim, device=device)
        
        # 1. Visited flag
        visited = (state != -1).float()
        features[:, :, 0] = visited
        
        # 2. Position in tour (normalized)
        position = state.float()
        position = torch.where(state == -1, torch.zeros_like(position), position / (N - 1))
        features[:, :, 1] = position
        
        # 3. Coordinates
        features[:, :, 2:4] = coords_norm.unsqueeze(0).expand(B, -1, -1)
        
        # 4. Is current city
        max_pos = state.max(dim=1, keepdim=True)[0]
        is_current = (state == max_pos).float() * visited
        features[:, :, 4] = is_current
        
        # 5. Distance to current city
        current_city = state.argmax(dim=1)
        dist_to_current = dist_norm[current_city]
        features[:, :, 5] = dist_to_current
        
        return features
    
    def get_embeddings(self, state, coords, distance_matrix, device="cpu"):
        """Get city embeddings using shared backbone."""
        features = self.get_features(state, coords, distance_matrix, device)
        
        h = self.city_encoder(features)
        
        for attn_block in self.attention_layers:
            h = attn_block(h)
        
        return h
    
    def set_mode(self, mode: str):
        """Set policy mode: 'forward' or 'backward'."""
        assert mode in ('forward', 'backward')
        self._mode = mode
    
    def forward_logits(self, state, coords, distance_matrix, device="cpu"):
        """Compute forward policy logits."""
        single_input = False
        if isinstance(state, np.ndarray) and state.ndim == 1:
            single_input = True
        elif isinstance(state, torch.Tensor) and state.dim() == 1:
            single_input = True
        
        h = self.get_embeddings(state, coords, distance_matrix, device)
        logits = self.forward_head(h).squeeze(-1)
        
        if single_input:
            return logits.squeeze(0)
        return logits
    
    def backward_logits(self, state, coords, distance_matrix, device="cpu"):
        """Compute backward policy logits."""
        single_input = False
        if isinstance(state, np.ndarray) and state.ndim == 1:
            single_input = True
        elif isinstance(state, torch.Tensor) and state.dim() == 1:
            single_input = True
        
        h = self.get_embeddings(state, coords, distance_matrix, device)
        logits = self.backward_head(h).squeeze(-1)
        
        if single_input:
            return logits.squeeze(0)
        return logits
    
    def forward(self, state, coords, distance_matrix, device="cpu"):
        """Default forward pass using current mode."""
        if self._mode == 'forward':
            return self.forward_logits(state, coords, distance_matrix, device)
        else:
            return self.backward_logits(state, coords, distance_matrix, device)
    
    def predict_flow(self, state, coords, distance_matrix, device="cpu"):
        """Predict log flow F(s)."""
        single_input = False
        if isinstance(state, np.ndarray) and state.ndim == 1:
            single_input = True
        elif isinstance(state, torch.Tensor) and state.dim() == 1:
            single_input = True
        
        h = self.get_embeddings(state, coords, distance_matrix, device)
        
        h_mean = h.mean(dim=1)
        h_max = h.max(dim=1)[0]
        h_graph = h_mean + h_max
        
        log_flow = self.flow_head(h_graph).squeeze(-1)
        
        if single_input:
            return log_flow.squeeze(0)
        return log_flow


class ConditionalTSPPolicyWrapper(nn.Module):
    """
    Wrapper to expose a specific head of ConditionalTSPPolicy.
    """
    
    def __init__(self, shared_policy: ConditionalTSPPolicy, mode: str):
        super().__init__()
        self.shared_policy = shared_policy
        self.mode = mode
        assert mode in ('forward', 'backward')
    
    def forward(self, state, coords, distance_matrix, device="cpu"):
        if self.mode == 'forward':
            return self.shared_policy.forward_logits(state, coords, distance_matrix, device)
        else:
            return self.shared_policy.backward_logits(state, coords, distance_matrix, device)
    
    def predict_flow(self, state, coords, distance_matrix, device="cpu"):
        return self.shared_policy.predict_flow(state, coords, distance_matrix, device)
    
    def parameters(self, recurse=True):
        return self.shared_policy.parameters(recurse)


class AttentionBlock(nn.Module):
    """Self-attention block with residual connection."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, h):
        """
        Args:
            h: (B, N, H) node embeddings
            
        Returns:
            h_new: (B, N, H) updated embeddings
        """
        # Self-attention with residual
        h_attn, _ = self.attention(h, h, h)
        h = self.layer_norm1(h + h_attn)
        
        # FFN with residual
        h_ffn = self.ffn(h)
        h = self.layer_norm2(h + h_ffn)
        
        return h
