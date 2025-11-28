# training/trainer.py

import os
import json
import time
from datetime import datetime
import torch
import torch.nn.functional as F
import numpy as np
from training.sampler import sample_trajectory

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
    
    node_indices = torch.arange(N, device=device)
    
    for i, action in enumerate(actions):
        state_before = states[i]
        state_after = states[i + 1]
        
        # forward
        logits_f = forward_policy(state_before, env.adj, device=device)
        mask_f = torch.from_numpy(env.allowed_actions(state_before)).to(device)
        logits_f = torch.where(mask_f > 0, logits_f, inf_val)
        logprob_f = logprob_f + F.log_softmax(logits_f, dim=0)[action]
        
        # backward
        logits_b = backward_policy(state_after, env.adj, device=device)
        
        colored_mask = torch.from_numpy(state_after != -1).to(device)
        colors = torch.from_numpy(np.maximum(state_after, 0)).to(device)
        backward_mask = torch.zeros(N * K, device=device)
        action_indices = node_indices * K + colors
        backward_mask.scatter_(0, action_indices[colored_mask], 1.0)
        
        logits_b = torch.where(backward_mask > 0, logits_b, inf_val)
        logprob_b = logprob_b + F.log_softmax(logits_b, dim=0)[action]
    
    return logprob_f, logprob_b



#training
def train(env, forward_policy, backward_policy, loss_fn, optimizer,
          steps=2000, device="cpu", save_dir="checkpoints", problem_name=None,
          batch_size=16, epsilon_start=0.3, log_dir="logs"):

    last_terminal_state = None
    reward_history = []
    loss_history = []  # Track loss for smoothed reporting
    best_reward = 0.0
    global_best_state = None

    batch_size = 8            
    accum_steps = 4           # gradient accumulation

    # Create checkpoint directory
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    optimizer.zero_grad()

    for step in range(steps):

        # Decayed epsilon schedule
        epsilon = max(0.05, 0.5 * (1 - step / steps))

        #batch size = 8
        traj_batch = []
        for _ in range(batch_size):
            traj_states, traj_actions, reward = sample_trajectory(
                env, forward_policy, device, epsilon=epsilon
            )
            traj_batch.append((traj_states, traj_actions, reward))

        # Track last terminal state 
        # last_terminal_state = traj_batch[-1][0][-1]
        # reward_history.append(traj_batch[-1][2])

        # Track best reward of all trajectories
        # for (x, y, r) in traj_batch:
        #     best_reward = max(best_reward, r)

        batch_rewards = [r for (_,_,r) in traj_batch]
        best_idx = int(np.argmax(batch_rewards))
        best_traj_states, _, best_r = traj_batch[best_idx]

        last_terminal_state = best_traj_states[-1]   # the best terminal state
        reward_history.append(best_r)

        # Update global best reward
        if best_r > best_reward:
            best_reward = best_r
            global_best_state = best_traj_states[-1].copy()

        # trajectory balance loss
        total_loss = 0.0

        for (traj_states, traj_actions, reward) in traj_batch:
            logf, logb = compute_logprobs_fast(
                traj_states, traj_actions,
                forward_policy, backward_policy,
                env, device
            )
            logreward = torch.log(torch.tensor(reward + 1e-8, device=device))

            total_loss = total_loss + loss_fn(logf, logb, logreward)

        # Average across batch and accumulation
        loss = (total_loss / batch_size) / accum_steps

        loss.backward()

        if (step + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()


        #loss and reward
        if step % 50 == 0:
            avg_reward = np.mean(reward_history[-50*batch_size:]) if len(reward_history) >= 50*batch_size else np.mean(reward_history)
            avg_loss = np.mean(loss_history[-50:]) if len(loss_history) >= 50 else np.mean(loss_history)
            colored = np.sum(last_terminal_state != -1)
            colors_used = len(set(c for c in last_terminal_state if c != -1))

            print(
                f"[{step}] loss={loss.item()*accum_steps:.4f}, "
                f"best={best_reward:.4f}, avg={avg_reward:.4f}, "
                f"colored={colored}/{env.N}, colors={colors_used}/{env.K}, "
                f"eps={epsilon:.2f}",
                flush=True
            )


        # save checkpoint
        if save_dir is not None and (step + 1) % 500 == 0:
            checkpoint = {
                'step': step + 1,
                'forward_policy': forward_policy.state_dict(),
                'backward_policy': backward_policy.state_dict(),
                'loss_fn': loss_fn.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_reward': best_reward,
                'problem_name': problem_name,
            }
            ckpt_name = f"{problem_name}_step_{step+1}.pt" if problem_name else f"checkpoint_step_{step+1}.pt"
            torch.save(checkpoint, os.path.join(save_dir, ckpt_name))
            print(f"  -> Saved checkpoint: {ckpt_name}", flush=True)

    return global_best_state