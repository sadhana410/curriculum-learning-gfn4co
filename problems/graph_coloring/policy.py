# models/policy_net.py

import torch
import torch.nn as nn
import numpy as np

class GNNPolicy(nn.Module):
    """
    GNN policy for graph colouring.

    Inputs:  state (length N, values in {-1, 0, ..., K-1}),
               adj   (NxN adjacency matrix)
    Output:  logits over N*K actions (node* K + color)
    """

    def __init__(self, num_nodes: int, num_colors: int, hidden_dim: int = 64):
        super().__init__()
        self.N = num_nodes
        self.K = num_colors
        self.hidden_dim = hidden_dim
        
        # Cache for adjacency matrix
        self._adj_cache = None

        self.embedding = nn.Embedding(num_colors + 1, hidden_dim)

        self.W_self = nn.Linear(hidden_dim, hidden_dim)
        self.W_neigh = nn.Linear(hidden_dim, hidden_dim)

        self.out = nn.Linear(hidden_dim, num_colors)
        
        # Flow head for DB/SubTB
        self.flow_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

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
        Compute node embeddings.
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

    def forward(self, state, adj, device="cpu"):
        """
        state: numpy array or tensor
            shape (N,)      -> returns (N*K,)
            shape (B, N)    -> returns (B, N*K)
        adj:   NxN adjacency matrix
        """
        h = self.get_node_embeddings(state, adj, device)
        B = h.shape[0]
        
        # Per-node logits: (B, N, K)
        node_logits = self.out(h)

        # Flatten node/color -> action: (B, N*K)
        logits = node_logits.view(B, -1)

        if state.dim() == 1 or (isinstance(state, np.ndarray) and state.ndim == 1):
            return logits.squeeze(0)            # (N*K,)
        return logits                           # (B, N*K)

    def predict_flow(self, state, adj, device="cpu"):
        """
        Predict Log Flow F(s).
        Returns: (B, 1) or (1,)
        """
        h = self.get_node_embeddings(state, adj, device)
        
        # Global pooling (max over nodes)
        h_graph = h.max(dim=1)[0]  # (B, H)
        
        log_flow = self.flow_head(h_graph)  # (B, 1)
        
        if state.dim() == 1 or (isinstance(state, np.ndarray) and state.ndim == 1):
            return log_flow.squeeze(0)
        return log_flow.squeeze(-1)  # (B,)
