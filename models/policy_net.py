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

        self.embedding = nn.Embedding(num_colors + 1, hidden_dim)

        self.W_self = nn.Linear(hidden_dim, hidden_dim)
        self.W_neigh = nn.Linear(hidden_dim, hidden_dim)

        self.out = nn.Linear(hidden_dim, num_colors)

    def forward(self, state, adj, device="cpu"):
        """
        state: numpy array or 1D tensor of length N, values in {-1,0,...,K-1}
        adj:   numpy array NxN (0/1)
        returns: logits of shape (N*K,)
        """
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.long, device=device)
        else:
            state = state.to(device).long()

        color_ids = state.clone()
        color_ids[color_ids == -1] = self.K

        x = self.embedding(color_ids) 

        if isinstance(adj, np.ndarray):
            A = torch.tensor(adj, dtype=torch.float32, device=device)
        else:
            A = adj.to(device).float()

        #h = ReLU( W_self x + W_neigh (A x) )
        h_self = self.W_self(x)
        neigh_agg = torch.matmul(A, x)         
        h_neigh = self.W_neigh(neigh_agg)

        h = torch.relu(h_self + h_neigh)        

        node_logits = self.out(h)              

        #action logits (node * K + color)
        return node_logits.view(-1)             # (N*K,)
