# models/policy_net.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GNNPolicy(nn.Module):
    """
    GNN policy for graph colouring (fixed graph size) with shared backbone
    and separate forward/backward heads.

    Inputs:  state (length N, values in {-1, 0, ..., K-1}),
               adj   (NxN adjacency matrix)
    Output:  logits over N*K actions (node * K + color)
    
    The backbone (embedding + GNN layers) is shared between forward and backward
    policies. Small separate heads produce the final logits for each direction.
    """

    def __init__(self, num_nodes: int, num_colors: int, hidden_dim: int = 64, head_hidden_dim: int = 32):
        super().__init__()
        self.N = num_nodes
        self.K = num_colors
        self.hidden_dim = hidden_dim
        self.head_hidden_dim = head_hidden_dim
        
        # Cache for adjacency matrix
        self._adj_cache = None

        # ===== SHARED BACKBONE =====
        self.embedding = nn.Embedding(num_colors + 1, hidden_dim)
        self.W_self = nn.Linear(hidden_dim, hidden_dim)
        self.W_neigh = nn.Linear(hidden_dim, hidden_dim)

        # ===== FORWARD POLICY HEAD =====
        self.forward_head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, num_colors)
        )
        
        # ===== BACKWARD POLICY HEAD =====
        self.backward_head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, num_colors)
        )
        
        # ===== FLOW HEAD (for DB/SubTB) =====
        self.flow_head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, 1)
        )
        
        # Track which head to use (for compatibility with existing code)
        self._mode = 'forward'

    def _get_adj(self, adj, device):
        """Cache adjacency matrix on device."""
        if self._adj_cache is None or self._adj_cache.device != torch.device(device):
            if isinstance(adj, np.ndarray):
                self._adj_cache = torch.tensor(adj, dtype=torch.float32, device=device)
            else:
                self._adj_cache = adj.to(device).float()
        return self._adj_cache

    def get_node_embeddings(self, state, adj, device="cpu"):
        """
        Compute node embeddings using shared backbone.
        Returns: (B, N, H)
        """
        # Convert to tensor on device
        if isinstance(state, np.ndarray):
            state = torch.as_tensor(state, dtype=torch.long, device=device)
        else:
            state = state.to(device).long()

        # Ensure we have batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)  # (1, N)

        B, N = state.shape
        assert N == self.N, f"Expected N={self.N}, got {N}"

        # Replace -1 with K for embedding lookup
        color_ids = torch.where(state == -1, self.K, state)  # (B, N)

        # Node embeddings: (B, N, H)
        x = self.embedding(color_ids)

        # Adjacency
        A = self._get_adj(adj, device)          # (N, N)
        A_batch = A.unsqueeze(0).expand(B, -1, -1)  # (B, N, N)

        # Message passing
        h_self = self.W_self(x)                 # (B, N, H)
        neigh_agg = torch.bmm(A_batch, x)       # (B, N, H)
        h_neigh = self.W_neigh(neigh_agg)       # (B, N, H)
        h = torch.relu(h_self + h_neigh)        # (B, N, H)
        
        return h
    
    def set_mode(self, mode: str):
        """Set policy mode: 'forward' or 'backward'."""
        assert mode in ('forward', 'backward'), f"Mode must be 'forward' or 'backward', got {mode}"
        self._mode = mode
    
    def forward_logits(self, state, adj, device="cpu"):
        """
        Compute forward policy logits.
        Returns: (N*K,) or (B, N*K)
        """
        single_input = isinstance(state, np.ndarray) and state.ndim == 1
        if isinstance(state, torch.Tensor) and state.dim() == 1:
            single_input = True
            
        h = self.get_node_embeddings(state, adj, device)
        B = h.shape[0]
        
        # Per-node logits via forward head: (B, N, K)
        node_logits = self.forward_head(h)
        logits = node_logits.view(B, -1)  # (B, N*K)

        if single_input:
            return logits.squeeze(0)  # (N*K,)
        return logits
    
    def backward_logits(self, state, adj, device="cpu"):
        """
        Compute backward policy logits.
        Returns: (N*K,) or (B, N*K)
        """
        single_input = isinstance(state, np.ndarray) and state.ndim == 1
        if isinstance(state, torch.Tensor) and state.dim() == 1:
            single_input = True
            
        h = self.get_node_embeddings(state, adj, device)
        B = h.shape[0]
        
        # Per-node logits via backward head: (B, N, K)
        node_logits = self.backward_head(h)
        logits = node_logits.view(B, -1)  # (B, N*K)

        if single_input:
            return logits.squeeze(0)  # (N*K,)
        return logits

    def forward(self, state, adj, device="cpu"):
        """
        Default forward pass - uses current mode.
        For backward compatibility with existing training code.
        
        state: numpy array or tensor
            shape (N,)      -> returns (N*K,)
            shape (B, N)    -> returns (B, N*K)
        adj:   NxN adjacency matrix
        """
        if self._mode == 'forward':
            return self.forward_logits(state, adj, device)
        else:
            return self.backward_logits(state, adj, device)

    def predict_flow(self, state, adj, device="cpu"):
        """
        Predict Log Flow F(s).
        Returns: (B,) or scalar
        """
        single_input = isinstance(state, np.ndarray) and state.ndim == 1
        if isinstance(state, torch.Tensor) and state.dim() == 1:
            single_input = True
            
        h = self.get_node_embeddings(state, adj, device)
        
        # Global pooling (max over nodes)
        h_graph = h.max(dim=1)[0]  # (B, H)
        
        log_flow = self.flow_head(h_graph)  # (B, 1)
        
        if single_input:
            return log_flow.squeeze()
        return log_flow.squeeze(-1)  # (B,)


class GNNPolicyWrapper(nn.Module):
    """
    Wrapper to expose a specific head (forward or backward) of a SharedGNNPolicy
    as a standalone policy. This allows using the shared policy with existing
    training code that expects separate forward/backward policy objects.
    """
    
    def __init__(self, shared_policy: GNNPolicy, mode: str):
        super().__init__()
        self.shared_policy = shared_policy
        self.mode = mode
        assert mode in ('forward', 'backward'), f"Mode must be 'forward' or 'backward'"
    
    def forward(self, state, adj, device="cpu"):
        if self.mode == 'forward':
            return self.shared_policy.forward_logits(state, adj, device)
        else:
            return self.shared_policy.backward_logits(state, adj, device)
    
    def predict_flow(self, state, adj, device="cpu"):
        return self.shared_policy.predict_flow(state, adj, device)
    
    def parameters(self, recurse=True):
        # Return only the head parameters + shared backbone
        # This ensures proper gradient flow
        return self.shared_policy.parameters(recurse)
    
    @property
    def N(self):
        return self.shared_policy.N
    
    @property
    def K(self):
        return self.shared_policy.K


class ConditionalGNNPolicy(nn.Module):
    """
    Conditional GNN policy for graph coloring that generalizes across different graphs
    with shared backbone and separate forward/backward heads.
    
    This policy is conditioned on the graph structure (adjacency matrix) and can handle
    variable-sized graphs. It uses message passing GNN layers that are size-agnostic.
    
    The backbone (embedding + GNN layers) is shared between forward and backward
    policies. Small separate heads produce the final logits for each direction.
    
    Inputs:
        state: (B, N) tensor of node colors {-1, 0, ..., K-1}
        adj: (B, N, N) batch of adjacency matrices OR (N, N) single adjacency
    Output:
        logits: (B, N*K) action logits
    """
    
    def __init__(self, num_colors: int, hidden_dim: int = 64, num_layers: int = 3, head_hidden_dim: int = 32):
        super().__init__()
        self.K = num_colors
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.head_hidden_dim = head_hidden_dim
        
        # ===== SHARED BACKBONE =====
        # Color embedding: K colors + 1 for uncolored (-1)
        self.color_embedding = nn.Embedding(num_colors + 1, hidden_dim)
        
        # GNN layers (size-agnostic message passing)
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(GNNLayer(hidden_dim, hidden_dim))
        
        # ===== FORWARD POLICY HEAD =====
        self.forward_head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, num_colors)
        )
        
        # ===== BACKWARD POLICY HEAD =====
        self.backward_head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, num_colors)
        )
        
        # ===== FLOW HEAD (for DB/SubTB) =====
        self.flow_head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, 1)
        )
        
        # Track which head to use (for compatibility with existing code)
        self._mode = 'forward'
    
    def get_node_embeddings(self, state, adj, device="cpu"):
        """
        Compute node embeddings using shared GNN backbone.
        
        Args:
            state: (N,) or (B, N) tensor of node colors
            adj: (N, N) or (B, N, N) adjacency matrix
            
        Returns:
            h: (B, N, H) node embeddings
        """
        # Convert to tensor
        if isinstance(state, np.ndarray):
            state = torch.as_tensor(state, dtype=torch.long, device=device)
        else:
            state = state.to(device).long()
            
        if isinstance(adj, np.ndarray):
            adj = torch.as_tensor(adj, dtype=torch.float32, device=device)
        else:
            adj = adj.to(device).float()
        
        # Ensure batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)  # (1, N)
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)  # (1, N, N)
            
        B, N = state.shape
        
        # Expand adj if needed (single adj for all batch)
        if adj.shape[0] == 1 and B > 1:
            adj = adj.expand(B, -1, -1)
        
        # Replace -1 with K for embedding lookup
        color_ids = torch.where(state == -1, self.K, state)  # (B, N)
        
        # Initial node embeddings from colors
        h = self.color_embedding(color_ids)  # (B, N, H)
        
        # Message passing layers
        for gnn_layer in self.gnn_layers:
            h = gnn_layer(h, adj)  # (B, N, H)
        
        return h
    
    def set_mode(self, mode: str):
        """Set policy mode: 'forward' or 'backward'."""
        assert mode in ('forward', 'backward'), f"Mode must be 'forward' or 'backward', got {mode}"
        self._mode = mode
    
    def forward_logits(self, state, adj, device="cpu"):
        """
        Compute forward policy logits.
        
        Args:
            state: (N,) or (B, N) tensor of node colors
            adj: (N, N) or (B, N, N) adjacency matrix
            
        Returns:
            logits: (N*K,) or (B, N*K) action logits
        """
        single_input = False
        if isinstance(state, np.ndarray) and state.ndim == 1:
            single_input = True
        elif isinstance(state, torch.Tensor) and state.dim() == 1:
            single_input = True
            
        h = self.get_node_embeddings(state, adj, device)  # (B, N, H)
        B, N, H = h.shape
        
        # Per-node logits via forward head
        node_logits = self.forward_head(h)  # (B, N, K)
        
        # Flatten to action space
        logits = node_logits.view(B, N * self.K)  # (B, N*K)
        
        if single_input:
            return logits.squeeze(0)  # (N*K,)
        return logits
    
    def backward_logits(self, state, adj, device="cpu"):
        """
        Compute backward policy logits.
        
        Args:
            state: (N,) or (B, N) tensor of node colors
            adj: (N, N) or (B, N, N) adjacency matrix
            
        Returns:
            logits: (N*K,) or (B, N*K) action logits
        """
        single_input = False
        if isinstance(state, np.ndarray) and state.ndim == 1:
            single_input = True
        elif isinstance(state, torch.Tensor) and state.dim() == 1:
            single_input = True
            
        h = self.get_node_embeddings(state, adj, device)  # (B, N, H)
        B, N, H = h.shape
        
        # Per-node logits via backward head
        node_logits = self.backward_head(h)  # (B, N, K)
        
        # Flatten to action space
        logits = node_logits.view(B, N * self.K)  # (B, N*K)
        
        if single_input:
            return logits.squeeze(0)  # (N*K,)
        return logits
    
    def forward(self, state, adj, device="cpu"):
        """
        Default forward pass - uses current mode.
        For backward compatibility with existing training code.
        
        Args:
            state: (N,) or (B, N) tensor of node colors
            adj: (N, N) or (B, N, N) adjacency matrix
            
        Returns:
            logits: (N*K,) or (B, N*K) action logits
        """
        if self._mode == 'forward':
            return self.forward_logits(state, adj, device)
        else:
            return self.backward_logits(state, adj, device)
    
    def predict_flow(self, state, adj, device="cpu"):
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
            
        h = self.get_node_embeddings(state, adj, device)  # (B, N, H)
        
        # Global pooling (mean + max)
        h_mean = h.mean(dim=1)  # (B, H)
        h_max = h.max(dim=1)[0]  # (B, H)
        h_graph = h_mean + h_max  # (B, H)
        
        log_flow = self.flow_head(h_graph).squeeze(-1)  # (B,)
        
        if single_input:
            return log_flow.squeeze(0)
        return log_flow


class ConditionalGNNPolicyWrapper(nn.Module):
    """
    Wrapper to expose a specific head (forward or backward) of a ConditionalGNNPolicy
    as a standalone policy. This allows using the shared policy with existing
    training code that expects separate forward/backward policy objects.
    """
    
    def __init__(self, shared_policy: ConditionalGNNPolicy, mode: str):
        super().__init__()
        self.shared_policy = shared_policy
        self.mode = mode
        assert mode in ('forward', 'backward'), f"Mode must be 'forward' or 'backward'"
    
    def forward(self, state, adj, device="cpu"):
        if self.mode == 'forward':
            return self.shared_policy.forward_logits(state, adj, device)
        else:
            return self.shared_policy.backward_logits(state, adj, device)
    
    def predict_flow(self, state, adj, device="cpu"):
        return self.shared_policy.predict_flow(state, adj, device)
    
    def parameters(self, recurse=True):
        return self.shared_policy.parameters(recurse)
    
    @property
    def K(self):
        return self.shared_policy.K


class GNNLayer(nn.Module):
    """
    Single GNN message passing layer.
    Uses a simple aggregation: h' = ReLU(W_self * h + W_neigh * Adj @ h)
    """
    
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W_self = nn.Linear(in_dim, out_dim)
        self.W_neigh = nn.Linear(in_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)
    
    def forward(self, h, adj):
        """
        Args:
            h: (B, N, H) node features
            adj: (B, N, N) adjacency matrix
            
        Returns:
            h_new: (B, N, H) updated node features
        """
        # Self transformation
        h_self = self.W_self(h)  # (B, N, H)
        
        # Neighbor aggregation
        neigh_agg = torch.bmm(adj, h)  # (B, N, H)
        h_neigh = self.W_neigh(neigh_agg)  # (B, N, H)
        
        # Combine with residual connection
        h_new = F.relu(h_self + h_neigh)
        h_new = self.layer_norm(h_new)
        
        # Residual connection (if dimensions match)
        if h.shape[-1] == h_new.shape[-1]:
            h_new = h_new + h
        
        return h_new
