import torch
import torch.nn as nn

class TrajectoryBalance(nn.Module):
    def __init__(self, forward_policy, backward_policy):
        super().__init__()
        self.forward_policy = forward_policy
        self.backward_policy = backward_policy
        self.logZ = nn.Parameter(torch.tensor(0.0))

    def forward(self, logprobs_f, logprobs_b, logreward):
        lhs = self.logZ + logprobs_f
        rhs = logprobs_b + logreward
        diff = lhs - rhs
        # Huber loss: quadratic for small errors, linear for large errors
        # This prevents extreme loss spikes while preserving gradients
        delta = 2.0
        abs_diff = torch.abs(diff)
        loss = torch.where(
            abs_diff <= delta,
            0.5 * diff.pow(2),
            delta * (abs_diff - 0.5 * delta)
        )
        return loss
