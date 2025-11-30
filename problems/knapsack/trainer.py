# problems/knapsack/trainer.py

import os
import json
import time
from datetime import datetime
import torch
import torch.nn.functional as F
import numpy as np

from problems.knapsack.sampler import sample_trajectories_batched, collate_trajectories, get_batched_mask_knapsack


# Pre-allocate tensors for speed
_inf_tensor = None

def get_inf_tensor(device):
    global _inf_tensor
    if _inf_tensor is None or _inf_tensor.device != device:
        _inf_tensor = torch.tensor(float('-inf'), device=device)
    return _inf_tensor


def build_backward_masks_knapsack(states, N, device):
    """
    Compute backward allowed actions mask.
    states: (B, N)
    Returns: (B, 2*N) boolean tensor
    """
    B = states.shape[0]
    mask = torch.zeros(B, 2*N, dtype=torch.bool, device=device)
    
    # Selected items (1) allow undo select (action 0..N-1)
    mask[:, :N] = (states == 1)
    
    # Skipped items (0) allow undo skip (action N..2N-1)
    mask[:, N:] = (states == 0)
    
    return mask


def compute_logprobs_batched(states, actions, step_mask,
                             forward_policy, backward_policy, env, device):
    """
    states:    (T_max+1, B, N) long
    actions:   (T_max,   B)    long
    step_mask: (T_max,   B)    bool
    Returns:
      logprob_f: (B,)
      logprob_b: (B,)
    """
    T_max, B, N = states.shape[0] - 1, states.shape[1], states.shape[2]
    
    T_B = T_max * B
    
    states_before = states[:-1].reshape(T_B, N)
    states_after  = states[1:].reshape(T_B, N)
    flat_actions  = actions.reshape(T_B)
    flat_mask     = step_mask.reshape(T_B)
    
    valid_idx = flat_mask.nonzero(as_tuple=False).squeeze(-1)
    if valid_idx.numel() == 0:
        return torch.zeros(B, device=device), torch.zeros(B, device=device)
        
    states_before_valid = states_before[valid_idx]
    states_after_valid  = states_after[valid_idx]
    actions_valid       = flat_actions[valid_idx]
    
    # Batched policy calls
    logits_f = forward_policy(states_before_valid, device=device)
    logits_b = backward_policy(states_after_valid, device=device)
    
    # Masks
    mask_f = get_batched_mask_knapsack(states_before_valid, env, device).bool()
    mask_b = build_backward_masks_knapsack(states_after_valid, N, device)
    
    inf_val = get_inf_tensor(device)
    logits_f = torch.where(mask_f, logits_f, inf_val)
    logits_b = torch.where(mask_b, logits_b, inf_val)
    
    logp_f_all = F.log_softmax(logits_f, dim=-1)
    logp_b_all = F.log_softmax(logits_b, dim=-1)
    
    step_logp_f_valid = logp_f_all[
        torch.arange(actions_valid.numel(), device=device),
        actions_valid
    ]
    step_logp_b_valid = logp_b_all[
        torch.arange(actions_valid.numel(), device=device),
        actions_valid
    ]
    
    # Scatter back
    step_logp_f_flat = torch.zeros(T_B, device=device)
    step_logp_b_flat = torch.zeros(T_B, device=device)
    
    step_logp_f_flat[valid_idx] = step_logp_f_valid
    step_logp_b_flat[valid_idx] = step_logp_b_valid
    
    # Reshape to (T, B)
    step_logp_f = step_logp_f_flat.view(T_max, B)
    step_logp_b = step_logp_b_flat.view(T_max, B)
    
    # Mask invalid steps
    step_logp_f = step_logp_f * step_mask
    step_logp_b = step_logp_b * step_mask
    
    return step_logp_f, step_logp_b


def train(env, forward_policy, backward_policy, loss_fn, optimizer,
          steps=2000, device="cpu", save_dir=None, problem_name=None,
          batch_size=16, epsilon_start=0.5, log_dir=None, save_every=500,
          early_stop_patience=500, optimal_profit=None, top_p=1.0, temperature=1.0):
    """
    Training loop for knapsack GFN with batch training.
    """
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"{problem_name}_{timestamp}" if problem_name else f"train_{timestamp}"
    log_path = os.path.join(log_dir, f"{log_name}.jsonl")
    
    config = {
        "type": "config",
        "problem_name": problem_name,
        "steps": steps,
        "batch_size": batch_size,
        "epsilon_start": epsilon_start,
        "top_p": top_p,
        "temperature": temperature,
        "num_items": env.N,
        "capacity": env.capacity,
        "optimal_profit": optimal_profit,
        "device": str(device),
        "timestamp": timestamp,
    }
    with open(log_path, "w") as f:
        f.write(json.dumps(config) + "\n")
    
    print(f"Logging to: {log_path}")
    start_time = time.time()
    
    reward_history = []
    profit_history = []
    loss_history = []
    best_profit = 0.0
    best_reward = float('-inf')
    best_state = None
    
    # Early stopping tracking
    steps_without_improvement = 0
    prev_best_profit = 0.0
    
    for step in range(steps):
        epsilon = max(0.01, epsilon_start * (1 - step / steps))
        
        # Sample batch of trajectories in parallel
        batch_trajectories = sample_trajectories_batched(
            env, forward_policy, device, batch_size, 
            epsilon=epsilon, temperature=temperature, top_p=top_p
        )
        
        batch_rewards = []
        batch_profits = []
        
        for traj_states, traj_actions, reward in batch_trajectories:
            final_state = traj_states[-1]
            profit = env.get_profit(final_state)
            weight = env.get_weight(final_state)
            
            batch_rewards.append(reward)
            batch_profits.append(profit)
            
            if reward > best_reward:
                best_reward = reward

            if profit > best_profit and weight <= env.capacity:
                best_profit = profit
                best_state = final_state.copy()
        
        # Collate and compute losses
        states, actions, step_mask, rewards_tensor = collate_trajectories(
            batch_trajectories, env, device
        )
        
        logprobs_f_batch, logprobs_b_batch = compute_logprobs_batched(
            states, actions, step_mask,
            forward_policy, backward_policy, env, device
        )
        
        logreward_batch = rewards_tensor
        
        # Detect loss type
        loss_type = "TB"
        if hasattr(loss_fn, 'lambda_'):
            loss_type = "SubTB"
        elif loss_fn.__class__.__name__ == "DetailedBalance":
            loss_type = "DB"
            
        if loss_type == "TB":
            log_pf_sum = (logprobs_f_batch * step_mask).sum(dim=0)
            log_pb_sum = (logprobs_b_batch * step_mask).sum(dim=0)
            traj_losses = loss_fn(log_pf_sum, log_pb_sum, logreward_batch)
            cumulative_loss = traj_losses.mean()
        else:
            # DB or SubTB
            T_plus_1, B, N = states.shape
            flat_states = states.reshape(-1, N)
            
            # Predict flows
            flat_log_flows = forward_policy.predict_flow(flat_states, device)
            log_flows = flat_log_flows.view(T_plus_1, B)
            
            cumulative_loss = loss_fn(logprobs_f_batch, logprobs_b_batch, log_flows, logreward_batch, step_mask)
        
        reward_history.extend(batch_rewards)
        profit_history.extend(batch_profits)
        loss_history.append(cumulative_loss.item())
        
        optimizer.zero_grad()
        cumulative_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(forward_policy.parameters(), max_norm=0.5)
        if hasattr(forward_policy, 'flow_head'):
             torch.nn.utils.clip_grad_norm_(forward_policy.flow_head.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(backward_policy.parameters(), max_norm=0.5)
        if hasattr(loss_fn, 'logZ'):
            torch.nn.utils.clip_grad_norm_([loss_fn.logZ], max_norm=0.5)
        optimizer.step()
        
        if step % 50 == 0:
            avg_profit = np.mean(profit_history[-50*batch_size:]) if len(profit_history) >= 50*batch_size else np.mean(profit_history)
            avg_reward = np.mean(reward_history[-50*batch_size:]) if len(reward_history) >= 50*batch_size else np.mean(reward_history)
            avg_loss = np.mean(loss_history[-50:]) if len(loss_history) >= 50 else np.mean(loss_history)
            batch_avg_profit = np.mean(batch_profits)
            elapsed = time.time() - start_time
            
            # Calculate ETA
            if step > 0:
                steps_remaining = steps - step
                time_per_step = elapsed / step
                eta_seconds = steps_remaining * time_per_step
                eta_min, eta_sec = divmod(int(eta_seconds), 60)
                eta_hour, eta_min = divmod(eta_min, 60)
                eta_str = f"{eta_hour}h{eta_min:02d}m" if eta_hour > 0 else f"{eta_min}m{eta_sec:02d}s"
            else:
                eta_str = "--"
            
            opt_str = f"/{optimal_profit:.0f}" if optimal_profit else ""
            gap = ((optimal_profit - best_profit) / optimal_profit * 100) if optimal_profit and optimal_profit > 0 else 0
            print(f"[{step}] loss={cumulative_loss.item():.4f}, log_reward={avg_reward:.2f}, profit={batch_avg_profit:.1f}, best={best_profit:.0f}{opt_str}, gap={gap:.1f}%, no_improv={steps_without_improvement}, eps={epsilon:.3f}, ETA={eta_str}", flush=True)
            
            # Log to file
            log_entry = {
                "type": "step",
                "step": step,
                "loss": float(cumulative_loss.item()),
                "avg_loss": float(avg_loss),
                "avg_reward": float(avg_reward),
                "best_profit": float(best_profit),
                "optimal_profit": float(optimal_profit) if optimal_profit else None,
                "gap_percent": float(gap),
                "steps_without_improvement": steps_without_improvement,
                "epsilon": float(epsilon),
                "logZ": float(loss_fn.logZ.item()) if hasattr(loss_fn, 'logZ') else None,
                "elapsed_seconds": float(elapsed),
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        
        if (step + 1) % save_every == 0:
            checkpoint = {
                'step': step + 1,
                'forward_policy': forward_policy.state_dict(),
                'backward_policy': backward_policy.state_dict(),
                'loss_fn': loss_fn.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_profit': best_profit,
                'best_state': best_state,
                'problem_name': problem_name,
            }
            if problem_name:
                ckpt_name = f'{problem_name}_step_{step+1}.pt'
            else:
                ckpt_name = f'checkpoint_step_{step+1}.pt'
            torch.save(checkpoint, os.path.join(save_dir, ckpt_name))
            print(f"  -> Saved checkpoint: {ckpt_name}", flush=True)
        
        # Early stopping check
        if best_profit > prev_best_profit:
            prev_best_profit = best_profit
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1
        
        # Stop if optimal profit reached
        if optimal_profit and best_profit >= optimal_profit:
            print(f"\n✓ Optimal solution found! Profit: {best_profit:.0f} (optimal: {optimal_profit:.0f})")
            print(f"  Stopping at step {step + 1}")
            break
        
        # Stop if no improvement for patience steps
        if early_stop_patience and steps_without_improvement >= early_stop_patience:
            gap = ((optimal_profit - best_profit) / optimal_profit * 100) if optimal_profit else 0
            print(f"\n⚠ Early stopping: no improvement for {early_stop_patience} steps")
            print(f"  Best profit: {best_profit:.0f}, Optimal: {optimal_profit}, Gap: {gap:.2f}%")
            break
    
    return best_state, best_profit
