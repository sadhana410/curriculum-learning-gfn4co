# trajectorybalance.py
import torch
import torch.nn as nn

class TrajectoryBalance(nn.Module):
    """
    Minimal Trajectory Balance loss compatible with our simple training script.
    Does NOT use torchtyping or Batch class.
    """

    def __init__(self, forward_policy=None, backward_policy=None):
        super().__init__()
        assert forward_policy is not None
        assert backward_policy is not None

        self.forward_policy = forward_policy
        self.backward_policy = backward_policy

        # logZ is learnable
        self.logZ = nn.Parameter(torch.tensor(0.0))

    def forward(self, logprobs_f, logprobs_b, logreward):
        """
        TB loss = (logZ + sum_f - sum_b - logR)^2
        logprobs_f : scalar tensor (sum of forward logprobs)
        logprobs_b : scalar tensor (sum of backward logprobs)
        logreward  : scalar tensor (log(R))
        """
        return (self.logZ + logprobs_f - logprobs_b - logreward).pow(2)
