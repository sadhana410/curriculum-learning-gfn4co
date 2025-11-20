import torch
import torch.nn as nn

class TrajectoryBalance(nn.Module):
    def __init__(self, forward_policy, backward_policy):
        super().__init__()
        self.forward_policy = forward_policy
        self.backward_policy = backward_policy
        self.logZ = nn.Parameter(torch.tensor(0.0))

    def forward(self, logprobs_f, logprobs_b, logreward):
        return (self.logZ + logprobs_f - logprobs_b - logreward).pow(2)
