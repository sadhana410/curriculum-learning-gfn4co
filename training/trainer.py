# training/trainer.py

import os
import torch
import torch.nn.functional as F
import numpy as np
from training.sampler import sample_trajectory

# Pre-allocate tensors for speed
_inf_tensor = None

def get_inf_tensor(device):
    global _inf_tensor
    if _inf_tensor is None or _inf_tensor.device != device:
        _inf_tensor = torch.tensor(float('-inf'), device=device)
    return _inf_tensor

def compute_logprobs_fast(states, actions, forward_policy, backward_policy, env, device):
    """Compute forward and backward log probabilities in one pass."""
    N, K = env.N, env.K
    inf_val = get_inf_tensor(device)
    
    logprob_f = torch.tensor(0.0, device=device)
    logprob_b = torch.tensor(0.0, device=device)
    
    # Pre-compute node indices for backward mask
    node_indices = torch.arange(N, device=device)
    
    for i, action in enumerate(actions):
        state_before = states[i]
        state_after = states[i + 1]
        
        # Forward: P(action | state_before)
        logits_f = forward_policy(state_before, env.adj, device=device)
        mask_f = torch.from_numpy(env.allowed_actions(state_before)).to(device)
        logits_f = torch.where(mask_f > 0, logits_f, inf_val)
        logprob_f = logprob_f + F.log_softmax(logits_f, dim=0)[action]
        
        # Backward: P(action | state_after)
        logits_b = backward_policy(state_after, env.adj, device=device)
        colored_mask = torch.from_numpy(state_after != -1).to(device)
        colors = torch.from_numpy(np.maximum(state_after, 0)).to(device)
        backward_mask = torch.zeros(N * K, device=device)
        action_indices = node_indices * K + colors
        backward_mask.scatter_(0, action_indices[colored_mask], 1.0)
        logits_b = torch.where(backward_mask > 0, logits_b, inf_val)
        logprob_b = logprob_b + F.log_softmax(logits_b, dim=0)[action]
    
    return logprob_f, logprob_b

def train(env, forward_policy, backward_policy, loss_fn, optimizer,
          steps=2000, device="cpu", save_dir="checkpoints", problem_name=None):

    last_terminal_state = None
    reward_history = []
    best_reward = 0.0
    
    # Create checkpoint directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Use gradient accumulation for effective larger batch
    accum_steps = 4
    optimizer.zero_grad()

    for step in range(steps):
        # Decay epsilon over time - higher exploration for harder problems
        epsilon = max(0.05, 0.5 * (1 - step / steps))
        
        traj_states, traj_actions, reward = sample_trajectory(env, forward_policy, device, epsilon=epsilon)
        last_terminal_state = traj_states[-1]
        reward_history.append(reward)
        
        if reward > best_reward:
            best_reward = reward

        # Compute log probabilities (combined for speed)
        logprobs_f, logprobs_b = compute_logprobs_fast(
            traj_states, traj_actions, forward_policy, backward_policy, env, device
        )

        # TB loss
        logreward = torch.log(torch.tensor(reward + 1e-8, device=device))
        loss = loss_fn(logprobs_f, logprobs_b, logreward) / accum_steps

        loss.backward()
        
        # Update every accum_steps
        if (step + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        if step % 50 == 0:
            avg_reward = np.mean(reward_history[-50:]) if len(reward_history) >= 50 else np.mean(reward_history)
            colored = np.sum(last_terminal_state != -1)
            # Count unique colors used (excluding -1 for uncolored)
            colors_used = len(set(c for c in last_terminal_state if c != -1))
            print(f"[{step}] loss={loss.item()*accum_steps:.4f}, reward={reward:.4f}, avg_r={avg_reward:.4f}, best={best_reward:.4f}, colored={colored}/{env.N}, colors={colors_used}/{env.K}, eps={epsilon:.2f}", flush=True)
        
        # Save checkpoint every 500 steps
        if (step + 1) % 500 == 0:
            checkpoint = {
                'step': step + 1,
                'forward_policy': forward_policy.state_dict(),
                'backward_policy': backward_policy.state_dict(),
                'loss_fn': loss_fn.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_reward': best_reward,
                'problem_name': problem_name,
            }
            # Save with problem name prefix if provided
            if problem_name:
                ckpt_name = f'{problem_name}_step_{step+1}.pt'
            else:
                ckpt_name = f'checkpoint_step_{step+1}.pt'
            torch.save(checkpoint, os.path.join(save_dir, ckpt_name))
            print(f"  -> Saved checkpoint: {ckpt_name}", flush=True)

    return last_terminal_state
