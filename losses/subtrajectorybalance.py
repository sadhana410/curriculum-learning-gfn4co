import torch
import torch.nn as nn

class SubTrajectoryBalance(nn.Module):
    """
    Sub-Trajectory Balance Loss.
    Minimizes differences between forward and backward paths for all sub-trajectories.
    """
    def __init__(self, forward_policy, backward_policy, lambda_=1.0):
        super().__init__()
        self.forward_policy = forward_policy
        self.backward_policy = backward_policy
        self.lambda_ = lambda_

    def forward(self, log_pf, log_pb, log_flows, log_reward, step_mask):
        """
        log_pf: (T, B)
        log_pb: (T, B)
        log_flows: (T+1, B)
        log_reward: (B,)
        step_mask: (T, B)
        """
        T, B = log_pf.shape
        device = log_pf.device
        
        # Prepare flows with terminal reward substitution
        # lengths: (B,)
        lengths = step_mask.sum(dim=0).long()
        
        # Substitute terminal flow with reward
        # We construct a full flow tensor including R
        targets_flows = log_flows.clone()
        batch_indices = torch.arange(B, device=device)
        # Last valid state index is lengths[b] (0-based in states, so index is length)
        # log_flows has T+1 entries. states 0..T.
        # Trajectory of length L means states s_0...s_L.
        # log_flows[L] corresponds to F(s_L).
        targets_flows[lengths, batch_indices] = log_reward
        
        # Compute Cumulative Sums
        # Prepend 0 to match state indexing (0 to T)
        # cum_log_pf[t] = sum_{k=0}^{t-1} log pf_k. cum_log_pf[0] = 0.
        cum_log_pf = torch.zeros(T+1, B, device=device)
        cum_log_pf[1:] = torch.cumsum(log_pf, dim=0)
        
        cum_log_pb = torch.zeros(T+1, B, device=device)
        cum_log_pb[1:] = torch.cumsum(log_pb, dim=0)
        
        # A(t) = log F(s_t) - cum_log_pf[t] + cum_log_pb[t]
        # We want A(i) - A(j) = 0 for i < j
        # Actually SubTB equation:
        # log F(si) + sum P_F = log F(sj) + sum P_B
        # log F(si) + (C_F[j] - C_F[i]) = log F(sj) + (C_B[j] - C_B[i])
        # log F(si) - C_F[i] + C_B[i] = log F(sj) - C_F[j] + C_B[j]
        # So yes, we minimize diffs of A(t).
        
        A = targets_flows - cum_log_pf + cum_log_pb # (T+1, B)
        
        loss = 0.0
        count = 0
        
        # Compute all pairs (i, j) with i < j
        # We can vectorize over T using broadcasting
        
        # Mask for valid states: t <= length
        # valid_states: (T+1, B)
        # step_mask is (T, B). We need (T+1, B).
        valid_states = torch.zeros(T+1, B, dtype=torch.bool, device=device)
        valid_states[0] = True # Start state always valid
        # If step t exists (mask[t]), then state t+1 exists.
        valid_states[1:] = step_mask
        # Wait, step_mask[t] means transition s_t -> s_{t+1} exists.
        # So if step_mask[t] is true, s_{t+1} is valid.
        
        # However, we pad with -1 states/invalid steps.
        # We only care about i < j <= length.
        
        # Vectorized pairwise
        # shape (T+1, T+1, B)
        diffs = A.unsqueeze(1) - A.unsqueeze(0) # A[i] - A[j]
        diffs_sq = diffs ** 2
        
        # Weighting lambda^(j-i-1)
        # indices i (0), j (1)
        indices = torch.arange(T+1, device=device)
        i_idx = indices.unsqueeze(1).unsqueeze(2) # (T+1, 1, 1)
        j_idx = indices.unsqueeze(0).unsqueeze(2) # (1, T+1, 1)
        
        dist = j_idx - i_idx # j - i
        weights = torch.pow(self.lambda_, dist - 1)
        
        # Mask: i < j and j <= length
        # i < j
        pair_mask = (j_idx > i_idx)
        
        # Valid lengths
        len_mask = (j_idx <= lengths.view(1, 1, B))
        
        full_mask = pair_mask & len_mask
        
        # Loss
        weighted_diffs = diffs_sq * weights
        # Apply mask
        loss_terms = weighted_diffs * full_mask.float()
        
        loss = loss_terms.sum() / full_mask.sum()
        
        return loss
