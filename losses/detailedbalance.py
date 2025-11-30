import torch
import torch.nn as nn

class DetailedBalance(nn.Module):
    """
    Detailed Balance Loss for GFlowNets.
    L(s, s') = (log F(s) + log P_F(s'|s) - log F(s') - log P_B(s|s'))^2
    """
    def __init__(self, forward_policy, backward_policy):
        super().__init__()
        self.forward_policy = forward_policy
        self.backward_policy = backward_policy

    def forward(self, log_pf, log_pb, log_flows, log_reward, step_mask):
        """
        log_pf: (T, B) log P_F(s_{t+1} | s_t)
        log_pb: (T, B) log P_B(s_t | s_{t+1})
        log_flows: (T+1, B) log F(s_t) for t=0...T
        log_reward: (B,) log R(s_T)
        step_mask: (T, B) True if step t exists
        """
        T, B = log_pf.shape
        
        # F(s_t) and F(s_{t+1})
        log_F_s = log_flows[:-1]      # (T, B)
        log_F_next = log_flows[1:]    # (T, B)
        
        # For terminal states, F(s_T) should be fixed to R(s_T)
        # We enforce this by replacing log_F_next at the last valid step with log_reward
        # However, log_flows includes F(s_T) as predicted by network.
        # DB enforces F(s_T) = R(s_T). We can add a constraint or just use R for the last transition term.
        # A common way is to treat F(s_T) predictions as auxiliary and supervise them,
        # OR just use R in the transition equation for the last step.
        # Here we use R for the last step.
        
        # Create a target log_F_next tensor
        # If step t is the last step, target is log_reward
        # Else target is log_flows[t+1]
        
        # We can compute diffs for all steps using predicted flows
        diff = log_F_s + log_pf - log_F_next - log_pb
        
        # Correct the last step:
        # For the last valid step of each trajectory, log_F_next is predicted F(s_T).
        # We want it to match log_reward.
        # We can add a term (log_F_next_last - log_reward)^2
        # OR substitute R into the transition.
        # Let's substitute.
        
        # Identify last step indices
        # step_mask is (T, B). Sum dim 0 gives lengths.
        lengths = step_mask.sum(dim=0).long() # (B,)
        last_step_indices = lengths - 1
        
        # Create a mask for the last step
        last_step_mask = torch.zeros((T, B), dtype=torch.bool, device=log_pf.device)
        last_step_mask.scatter_(0, last_step_indices.unsqueeze(0), True)
        
        # Where last step, diff should use log_reward instead of log_F_next
        # Current diff: log_F_s + log_pf - log_F_next - log_pb
        # Target diff:  log_F_s + log_pf - log_reward - log_pb
        # Correction:   add log_F_next - log_reward
        
        # Wait, easier:
        # target_F_next = log_F_next.clone()
        # target_F_next[last_step_mask] = log_reward
        # But we can't modify in place with grads easily if we reuse log_flows?
        # Actually we can.
        
        targets_next = log_F_next.clone()
        # We need to expand log_reward to (B,) then scatter?
        # log_reward is (B,).
        
        # For each b, targets_next[lengths[b]-1, b] = log_reward[b]
        # This replaces the predicted flow at terminal state with actual reward
        batch_indices = torch.arange(B, device=log_pf.device)
        targets_next[last_step_indices, batch_indices] = log_reward
        
        # Recompute diff with targets
        # We also mask out invalid steps
        
        diff = log_F_s + log_pf - targets_next - log_pb
        diff = diff * step_mask
        
        loss = (diff ** 2).sum() / step_mask.sum()
        return loss
