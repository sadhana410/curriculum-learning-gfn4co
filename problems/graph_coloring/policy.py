import torch
import torch.nn as nn
import numpy as np


class GNNPolicy(nn.Module):
    """
    Simple 1-layer GNN policy for Graph Coloring.

    Inputs:
        state: length-N array, each entry in {-1, 0, ..., K-1}
        adj: NxN adjacency matrix (numpy or torch)
    Output:
        logits of shape (N*K,)
    """

    def __init__(self, num_nodes: int, num_colors: int, hidden_dim: int = 64):
        super().__init__()
        self.N = num_nodes
        self.K = num_colors
        self.hidden_dim = hidden_dim

        # Cache adjacency on device
        self._adj_cache = None

        # We need K+1 embeddings because -1 is mapped to index K
        self.embedding = nn.Embedding(self.K + 1, hidden_dim)

        # Message passing: x -> Wx_self + W(Ax)
        self.W_self = nn.Linear(hidden_dim, hidden_dim)
        self.W_neigh = nn.Linear(hidden_dim, hidden_dim)

        # Output logits for each node: shape (N, K)
        self.out = nn.Linear(hidden_dim, self.K)


    def _get_adj(self, adj, device):
        """Ensure adjacency is a torch tensor (cached per device)."""
        if self._adj_cache is None or self._adj_cache.device != torch.device(device):
            if isinstance(adj, np.ndarray):
                A = torch.tensor(adj, dtype=torch.float32, device=device)
            else:
                A = adj.to(device).float()
            self._adj_cache = A
        return self._adj_cache


    def forward(self, state, adj, device="cpu"):
        """
        Forward pass returning logits of size (N*K).
        """
        # Convert state to tensor on device
        if isinstance(state, np.ndarray):
            state = torch.as_tensor(state, dtype=torch.long, device=device)
        else:
            state = state.to(device).long()

        # Replace -1 (uncolored) with index K
        color_ids = torch.where(state == -1, self.K, state)

        # Node embeddings: (N, hidden_dim)
        x = self.embedding(color_ids)

        A = self._get_adj(adj, device)

        # GNN update
        # h = ReLU( W_self x + W_neigh (A x) )
        h_self = self.W_self(x)
        neigh_agg = A @ x
        h_neigh = self.W_neigh(neigh_agg)
        h = torch.relu(h_self + h_neigh)

        # Output node logits shape (N, K)
        node_logits = self.out(h)

        # Flatten to shape (N*K,)
        return node_logits.reshape(-1)
