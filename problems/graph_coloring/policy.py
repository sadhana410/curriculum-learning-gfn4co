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

    def _get_adj(self, adj, device):
        """Cache adjacency matrix on device."""
        if self._adj_cache is None or self._adj_cache.device != torch.device(device):
            if isinstance(adj, np.ndarray):
                self._adj_cache = torch.tensor(adj, dtype=torch.float32, device=device)
            else:
                self._adj_cache = adj.to(device).float()
        return self._adj_cache

    def forward(self, state, adj, device="cpu"):
        """
        state: numpy array or 1D tensor of length N, values in {-1,0,...,K-1}
        adj:   numpy array NxN (0/1)
        returns: logits of shape (N*K,)
        """
        if isinstance(state, np.ndarray):
            state = torch.as_tensor(state, dtype=torch.long, device=device)
        else:
            state = state.to(device).long()

        # Replace -1 with K for embedding lookup
        color_ids = torch.where(state == -1, self.K, state)

        x = self.embedding(color_ids) 

        A = self._get_adj(adj, device)

        #h = ReLU( W_self x + W_neigh (A x) )
        h_self = self.W_self(x)
        neigh_agg = A @ x
        h_neigh = self.W_neigh(neigh_agg)

        h = torch.relu(h_self + h_neigh)        

        node_logits = self.out(h)              

        #action logits (node * K + color)
        return node_logits.view(-1)             # (N*K,)
