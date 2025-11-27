# training/trainer.py

import os
import json
import time
from datetime import datetime
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
          steps=2000, device="cpu", save_dir="checkpoints", problem_name=None,
          batch_size=16, epsilon_start=0.3, log_dir="logs"):

    last_terminal_state = None
    reward_history = []
    loss_history = []  # Track loss for smoothed reporting
    best_reward = 0.0
    
    # Create checkpoint and log directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"{problem_name}_{timestamp}" if problem_name else f"train_{timestamp}"
    log_path = os.path.join(log_dir, f"{log_name}.jsonl")
    
    # Log training config
    config = {
        "type": "config",
        "problem_name": problem_name,
        "steps": steps,
        "batch_size": batch_size,
        "epsilon_start": epsilon_start,
        "num_nodes": env.N,
        "num_colors": env.K,
        "device": str(device),
        "timestamp": timestamp,
    }
    with open(log_path, "w") as f:
        f.write(json.dumps(config) + "\n")
    
    print(f"Logging to: {log_path}")
    start_time = time.time()

    for step in range(steps):
        # Epsilon decays from epsilon_start to 0.01
        epsilon = max(0.01, epsilon_start * (1 - step / steps))
        
        # Sample multiple trajectories and accumulate losses
        batch_losses = []
        batch_rewards = []
        
        for _ in range(batch_size):
            traj_states, traj_actions, reward = sample_trajectory(env, forward_policy, device, epsilon=epsilon)
            last_terminal_state = traj_states[-1]
            batch_rewards.append(reward)
            
            if reward > best_reward:
                best_reward = reward

            # Compute log probabilities
            logprobs_f, logprobs_b = compute_logprobs_fast(
                traj_states, traj_actions, forward_policy, backward_policy, env, device
            )

            # TB loss for this trajectory
            logreward = torch.log(torch.tensor(reward + 1e-8, device=device))
            traj_loss = loss_fn(logprobs_f, logprobs_b, logreward)
            batch_losses.append(traj_loss)
        
        # Cumulative loss: mean over all trajectories in batch
        cumulative_loss = torch.stack(batch_losses).mean()
        reward_history.extend(batch_rewards)
        loss_history.append(cumulative_loss.item())
        
        optimizer.zero_grad()
        cumulative_loss.backward()
        
        # Gradient clipping to stabilize training
        torch.nn.utils.clip_grad_norm_(forward_policy.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(backward_policy.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_([loss_fn.logZ], max_norm=0.5)
        optimizer.step()

        if step % 50 == 0:
            avg_reward = np.mean(reward_history[-50*batch_size:]) if len(reward_history) >= 50*batch_size else np.mean(reward_history)
            avg_loss = np.mean(loss_history[-50:]) if len(loss_history) >= 50 else np.mean(loss_history)
            colored = np.sum(last_terminal_state != -1)
            # Count unique colors used (excluding -1 for uncolored)
            colors_used = len(set(c for c in last_terminal_state if c != -1))
            batch_avg_reward = np.mean(batch_rewards)
            elapsed = time.time() - start_time
            
            print(f"[{step}] loss={cumulative_loss.item():.4f} (avg={avg_loss:.4f}), batch_r={batch_avg_reward:.4f}, avg_r={avg_reward:.4f}, best={best_reward:.4f}, colored={colored}/{env.N}, colors={colors_used}/{env.K}, eps={epsilon:.3f}", flush=True)
            
            # Log to file
            log_entry = {
                "type": "step",
                "step": step,
                "loss": cumulative_loss.item(),
                "avg_loss": avg_loss,
                "batch_reward": batch_avg_reward,
                "avg_reward": avg_reward,
                "best_reward": best_reward,
                "colored": int(colored),
                "colors_used": colors_used,
                "epsilon": epsilon,
                "logZ": loss_fn.logZ.item(),
                "elapsed_seconds": elapsed,
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        
        
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
