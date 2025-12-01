# problems/tsp/trainer.py

import os
import json
import time
from datetime import datetime
import torch
import torch.nn.functional as F
import numpy as np

from problems.tsp.sampler import (
    sample_trajectories_batched, collate_trajectories, get_batched_mask_tsp,
    sample_trajectories_conditional, sample_trajectories_conditional_batched,
    collate_trajectories_conditional, get_batched_mask_tsp_conditional
)


_inf_tensor = None

def get_inf_tensor(device):
    global _inf_tensor
    if _inf_tensor is None or _inf_tensor.device != device:
        _inf_tensor = torch.tensor(float('-inf'), device=device)
    return _inf_tensor


def build_backward_masks_tsp(states, N, device):
    """
    Compute backward allowed actions mask for TSP.
    
    In TSP, backward action undoes the last city visit.
    Only the last visited city (highest position value) can be unvisited.
    
    Args:
        states: (B, N) tensor
        
    Returns:
        mask: (B, N) boolean tensor
    """
    B = states.shape[0]
    mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    
    # Find the last visited city (highest position value)
    # Exclude city 0 which is always the start
    states_masked = states.clone()
    states_masked[:, 0] = -2  # Don't allow unvisiting city 0
    
    max_pos = states_masked.max(dim=1, keepdim=True)[0]  # (B, 1)
    
    # Only the city with max position can be unvisited (if > 0)
    is_last = (states == max_pos) & (max_pos > 0)
    mask = is_last
    
    return mask


def compute_logprobs_batched(states, actions, step_mask,
                             forward_policy, backward_policy, env, device):
    """
    Compute log probabilities for TSP trajectories.
    """
    T_max, B, N = states.shape[0] - 1, states.shape[1], states.shape[2]
    T_B = T_max * B
    
    states_before = states[:-1].reshape(T_B, N)
    states_after = states[1:].reshape(T_B, N)
    flat_actions = actions.reshape(T_B)
    flat_mask = step_mask.reshape(T_B)
    
    valid_idx = flat_mask.nonzero(as_tuple=False).squeeze(-1)
    if valid_idx.numel() == 0:
        return torch.zeros(B, device=device), torch.zeros(B, device=device)
    
    states_before_valid = states_before[valid_idx]
    states_after_valid = states_after[valid_idx]
    actions_valid = flat_actions[valid_idx]
    
    # Batched policy calls
    logits_f = forward_policy(states_before_valid, device=device)
    logits_b = backward_policy(states_after_valid, device=device)
    
    # Masks
    mask_f = get_batched_mask_tsp(states_before_valid, env, device).bool()
    mask_b = build_backward_masks_tsp(states_after_valid, N, device)
    
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
    
    step_logp_f = step_logp_f_flat.view(T_max, B)
    step_logp_b = step_logp_b_flat.view(T_max, B)
    
    step_logp_f = step_logp_f * step_mask
    step_logp_b = step_logp_b * step_mask
    
    return step_logp_f, step_logp_b


def train(env, forward_policy, backward_policy, loss_fn, optimizer,
          steps=2000, device="cpu", save_dir=None, problem_name=None,
          batch_size=16, epsilon_start=0.5, log_dir=None, save_every=500,
          early_stop_patience=500, optimal_length=None, top_p=1.0, temperature=1.0):
    """
    Training loop for TSP GFlowNet.
    """
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
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
        "num_cities": env.N,
        "optimal_length": float(optimal_length) if optimal_length is not None else None,
        "device": str(device),
        "timestamp": timestamp,
    }
    with open(log_path, "w") as f:
        f.write(json.dumps(config) + "\n")
    
    print(f"Logging to: {log_path}")
    start_time = time.time()
    
    reward_history = []
    length_history = []
    loss_history = []
    best_length = float('inf')
    best_reward = float('-inf')
    best_state = None
    
    steps_without_improvement = 0
    prev_best_length = float('inf')
    
    for step in range(steps):
        epsilon = max(0.01, epsilon_start * (1 - step / steps))
        
        batch_trajectories = sample_trajectories_batched(
            env, forward_policy, device, batch_size,
            epsilon=epsilon, temperature=temperature, top_p=top_p
        )
        
        batch_rewards = []
        batch_lengths = []
        
        for traj_states, traj_actions, reward in batch_trajectories:
            final_state = traj_states[-1]
            length = env.get_tour_length(final_state)
            
            batch_rewards.append(reward)
            batch_lengths.append(length)
            
            if reward > best_reward:
                best_reward = reward
            
            if length < best_length:
                best_length = length
                best_state = final_state.copy()
        
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
            T_plus_1, B, N = states.shape
            flat_states = states.reshape(-1, N)
            flat_log_flows = forward_policy.predict_flow(flat_states, device)
            log_flows = flat_log_flows.view(T_plus_1, B)
            cumulative_loss = loss_fn(logprobs_f_batch, logprobs_b_batch, log_flows, logreward_batch, step_mask)
        
        reward_history.extend(batch_rewards)
        length_history.extend(batch_lengths)
        loss_history.append(cumulative_loss.item())
        
        optimizer.zero_grad()
        cumulative_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(forward_policy.parameters(), max_norm=0.5)
        if hasattr(forward_policy, 'flow_head'):
            torch.nn.utils.clip_grad_norm_(forward_policy.flow_head.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(backward_policy.parameters(), max_norm=0.5)
        if hasattr(loss_fn, 'logZ'):
            torch.nn.utils.clip_grad_norm_([loss_fn.logZ], max_norm=0.5)
        optimizer.step()
        
        if step % 50 == 0:
            avg_length = np.mean(length_history[-50*batch_size:]) if len(length_history) >= 50*batch_size else np.mean(length_history)
            avg_reward = np.mean(reward_history[-50*batch_size:]) if len(reward_history) >= 50*batch_size else np.mean(reward_history)
            avg_loss = np.mean(loss_history[-50:]) if len(loss_history) >= 50 else np.mean(loss_history)
            batch_avg_length = np.mean(batch_lengths)
            elapsed = time.time() - start_time
            
            if step > 0:
                steps_remaining = steps - step
                time_per_step = elapsed / step
                eta_seconds = steps_remaining * time_per_step
                eta_min, eta_sec = divmod(int(eta_seconds), 60)
                eta_hour, eta_min = divmod(eta_min, 60)
                eta_str = f"{eta_hour}h{eta_min:02d}m" if eta_hour > 0 else f"{eta_min}m{eta_sec:02d}s"
            else:
                eta_str = "--"
            
            opt_str = f"/{optimal_length:.1f}" if optimal_length else ""
            gap = ((best_length - optimal_length) / optimal_length * 100) if optimal_length and optimal_length > 0 else 0
            gap_str = f"{max(0, gap):.1f}%"  # Avoid -0.0% display
            print(f"[{step}] loss={cumulative_loss.item():.4f}, log_reward={avg_reward:.2f}, length={batch_avg_length:.1f}, best={best_length:.1f}{opt_str}, gap={gap_str}, no_improv={steps_without_improvement}, eps={epsilon:.3f}, ETA={eta_str}", flush=True)
            
            log_entry = {
                "type": "step",
                "step": step,
                "loss": float(cumulative_loss.item()),
                "avg_loss": float(avg_loss),
                "avg_reward": float(avg_reward),
                "avg_length": float(avg_length),
                "best_length": float(best_length),
                "optimal_length": float(optimal_length) if optimal_length else None,
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
                'best_length': best_length,
                'best_state': best_state,
                'problem_name': problem_name,
            }
            if problem_name:
                ckpt_name = f'{problem_name}_step_{step+1}.pt'
            else:
                ckpt_name = f'checkpoint_step_{step+1}.pt'
            torch.save(checkpoint, os.path.join(save_dir, ckpt_name))
            print(f"  -> Saved checkpoint: {ckpt_name}", flush=True)
        
        if best_length < prev_best_length:
            prev_best_length = best_length
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1
        
        # Check if optimal solution found (within 0.01% tolerance)
        if optimal_length and best_length <= optimal_length * 1.0001:
            gap = max(0, (best_length - optimal_length) / optimal_length * 100)
            print(f"\n✓ Optimal solution found at step {step}!")
            print(f"  Best length: {best_length:.4f}, Optimal: {optimal_length:.4f}, Gap: {gap:.4f}%")
            # Save final checkpoint
            checkpoint = {
                'step': step + 1,
                'forward_policy': forward_policy.state_dict(),
                'backward_policy': backward_policy.state_dict(),
                'loss_fn': loss_fn.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_length': best_length,
                'best_state': best_state,
                'problem_name': problem_name,
                'optimal_found': True,
            }
            ckpt_name = f'{problem_name}_optimal.pt' if problem_name else 'checkpoint_optimal.pt'
            torch.save(checkpoint, os.path.join(save_dir, ckpt_name))
            print(f"  -> Saved optimal checkpoint: {ckpt_name}")
            break
        
        if early_stop_patience and steps_without_improvement >= early_stop_patience:
            gap = max(0, ((best_length - optimal_length) / optimal_length * 100)) if optimal_length else 0
            print(f"\n⚠ Early stopping: no improvement for {early_stop_patience} steps")
            print(f"  Best length: {best_length:.1f}, Optimal: {optimal_length}, Gap: {gap:.2f}%")
            break
    
    return best_state, best_length


# ============================================================================
# Conditional Training Functions
# ============================================================================

def build_backward_masks_tsp_conditional(states, N, device):
    """Compute backward allowed actions mask for conditional TSP."""
    return build_backward_masks_tsp(states, N, device)


def compute_logprobs_conditional(states, actions, step_mask, coords, distance_matrix,
                                  forward_policy, backward_policy, device):
    """
    Compute log probabilities for conditional TSP trajectories.
    """
    T_max, B, N = states.shape[0] - 1, states.shape[1], states.shape[2]
    T_B = T_max * B
    
    states_before = states[:-1].reshape(T_B, N)
    states_after = states[1:].reshape(T_B, N)
    flat_actions = actions.reshape(T_B)
    flat_mask = step_mask.reshape(T_B)
    
    valid_idx = flat_mask.nonzero(as_tuple=False).squeeze(-1)
    if valid_idx.numel() == 0:
        return torch.zeros(T_max, B, device=device), torch.zeros(T_max, B, device=device)
    
    states_before_valid = states_before[valid_idx]
    states_after_valid = states_after[valid_idx]
    actions_valid = flat_actions[valid_idx]
    
    logits_f = forward_policy(states_before_valid, coords, distance_matrix, device=device)
    logits_b = backward_policy(states_after_valid, coords, distance_matrix, device=device)
    
    mask_f = get_batched_mask_tsp_conditional(states_before_valid, device).bool()
    mask_b = build_backward_masks_tsp_conditional(states_after_valid, N, device)
    
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
    
    step_logp_f_flat = torch.zeros(T_B, device=device)
    step_logp_b_flat = torch.zeros(T_B, device=device)
    
    step_logp_f_flat[valid_idx] = step_logp_f_valid
    step_logp_b_flat[valid_idx] = step_logp_b_valid
    
    step_logp_f = step_logp_f_flat.view(T_max, B)
    step_logp_b = step_logp_b_flat.view(T_max, B)
    
    step_logp_f = step_logp_f * step_mask
    step_logp_b = step_logp_b * step_mask
    
    return step_logp_f, step_logp_b


def find_latest_checkpoint(save_dir, problem_name=None):
    """Find the latest checkpoint in save_dir."""
    if not os.path.exists(save_dir):
        return None
    
    checkpoints = []
    for f in os.listdir(save_dir):
        if not f.endswith('.pt'):
            continue
        if problem_name and not f.startswith(problem_name):
            continue
        checkpoints.append(os.path.join(save_dir, f))
    
    if not checkpoints:
        return None
    
    def get_step(path):
        name = os.path.basename(path)
        if 'step_' in name:
            try:
                return int(name.split('step_')[1].split('.')[0])
            except:
                pass
        return 0
    
    checkpoints.sort(key=get_step, reverse=True)
    return checkpoints[0]


def train_conditional(env, forward_policy, backward_policy, loss_fn, optimizer,
                      steps=2000, device="cpu", save_dir="checkpoints", problem_name=None,
                      batch_size=16, epsilon_start=0.3, log_dir="logs", save_every=500,
                      early_stop_patience=500, top_p=1.0, temperature=1.0,
                      same_instance_per_batch=True, resume_from=None):
    """
    Train Conditional GFlowNet for TSP on multiple instances.
    """
    reward_history = []
    loss_history = []
    best_reward = float('-inf')
    start_step = 0
    
    best_per_instance = {}
    for i in range(env.num_instances):
        inst = env.get_instance(i)
        best_per_instance[i] = {
            'length': float('inf'),
            'state': None,
            'name': inst['name'],
            'optimal': inst.get('optimal_length', None)
        }
    
    steps_without_improvement = 0
    prev_best_avg_gap = float('inf')
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    if resume_from:
        checkpoint_path = None
        
        if resume_from == 'latest':
            checkpoint_path = find_latest_checkpoint(save_dir, problem_name)
        elif os.path.exists(resume_from):
            checkpoint_path = resume_from
        else:
            potential_path = os.path.join(save_dir, resume_from)
            if os.path.exists(potential_path):
                checkpoint_path = potential_path
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            
            forward_policy.load_state_dict(checkpoint['forward_policy'])
            backward_policy.load_state_dict(checkpoint['backward_policy'])
            loss_fn.load_state_dict(checkpoint['loss_fn'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            
            start_step = checkpoint.get('step', 0)
            best_reward = checkpoint.get('best_reward', float('-inf'))
            
            if 'best_per_instance' in checkpoint:
                saved_best = checkpoint['best_per_instance']
                for idx in best_per_instance:
                    if idx in saved_best:
                        best_per_instance[idx]['length'] = saved_best[idx].get('length', float('inf'))
                        best_per_instance[idx]['state'] = saved_best[idx].get('state')
            
            print(f"  Resumed at step {start_step}")
            print(f"  Best reward: {best_reward:.2f}")
            print()
        else:
            print(f"Warning: Checkpoint not found: {resume_from}")
            print("Starting from scratch...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if resume_from and start_step > 0:
        log_name = f"{problem_name}_resumed_{timestamp}" if problem_name else f"train_conditional_resumed_{timestamp}"
    else:
        log_name = f"{problem_name}_{timestamp}" if problem_name else f"train_conditional_{timestamp}"
    log_path = os.path.join(log_dir, f"{log_name}.jsonl")
    
    loss_type = "TB"
    if hasattr(loss_fn, 'lambda_'):
        loss_type = "SubTB"
    elif loss_fn.__class__.__name__ == "DetailedBalance":
        loss_type = "DB"
    
    config = {
        "type": "config",
        "mode": "conditional",
        "problem_name": problem_name,
        "steps": steps,
        "start_step": start_step,
        "batch_size": batch_size,
        "epsilon_start": epsilon_start,
        "top_p": top_p,
        "temperature": temperature,
        "loss_type": loss_type,
        "num_instances": env.num_instances,
        "same_instance_per_batch": same_instance_per_batch,
        "device": str(device),
        "timestamp": timestamp,
        "resumed_from": resume_from if resume_from else None,
    }
    with open(log_path, "w") as f:
        f.write(json.dumps(config) + "\n")
    
    print(f"Conditional GFlowNet Training (TSP)")
    print(f"  Instances: {env.num_instances}")
    print(f"  Loss: {loss_type}")
    if start_step > 0:
        print(f"  Resuming from step: {start_step}")
    print(f"  Total steps: {steps}")
    print(f"  Logging to: {log_path}")
    print()
    
    start_time = time.time()
    
    for step in range(start_step, steps):
        epsilon = max(0.01, epsilon_start * (1 - step / steps))
        
        if same_instance_per_batch:
            batch_trajectories = sample_trajectories_conditional_batched(
                env, forward_policy, device, batch_size,
                epsilon=epsilon, temperature=temperature, top_p=top_p,
                same_instance=True
            )
        else:
            batch_trajectories = sample_trajectories_conditional(
                env, forward_policy, device, batch_size,
                epsilon=epsilon, temperature=temperature, top_p=top_p
            )
        
        batch_rewards = []
        batch_lengths = []
        
        for states_list, actions_list, reward, instance_idx, instance in batch_trajectories:
            batch_rewards.append(reward)
            
            if reward > best_reward:
                best_reward = reward
            
            final_state = states_list[-1]
            length = env.get_tour_length(final_state, instance_idx)
            batch_lengths.append(length)
            
            if length < best_per_instance[instance_idx]['length']:
                best_per_instance[instance_idx]['length'] = length
                best_per_instance[instance_idx]['state'] = final_state.copy()
        
        collated = collate_trajectories_conditional(batch_trajectories, device)
        
        total_loss = torch.tensor(0.0, device=device)
        total_trajs = 0
        
        for N, data in collated.items():
            states = data['states']
            actions = data['actions']
            step_mask = data['step_mask']
            rewards_tensor = data['rewards']
            coords = data['coords']
            distance_matrix = data['distance_matrix']
            B_group = states.shape[1]
            
            step_log_pf, step_log_pb = compute_logprobs_conditional(
                states, actions, step_mask, coords, distance_matrix,
                forward_policy, backward_policy, device
            )
            
            logreward_batch = rewards_tensor
            
            if loss_type == "TB":
                log_pf_sum = (step_log_pf * step_mask).sum(dim=0)
                log_pb_sum = (step_log_pb * step_mask).sum(dim=0)
                traj_losses = loss_fn(log_pf_sum, log_pb_sum, logreward_batch)
                total_loss = total_loss + traj_losses.sum()
            else:
                T_plus_1, B, _ = states.shape
                flat_states = states.reshape(-1, N)
                flat_log_flows = forward_policy.predict_flow(flat_states, coords, distance_matrix, device)
                log_flows = flat_log_flows.view(T_plus_1, B)
                
                group_loss = loss_fn(step_log_pf, step_log_pb, log_flows, logreward_batch, step_mask)
                total_loss = total_loss + group_loss * B_group
            
            total_trajs += B_group
        
        cumulative_loss = total_loss / max(total_trajs, 1)
        
        reward_history.extend(batch_rewards)
        loss_history.append(cumulative_loss.item())
        
        optimizer.zero_grad()
        cumulative_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(forward_policy.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(backward_policy.parameters(), max_norm=0.5)
        if hasattr(loss_fn, 'logZ'):
            torch.nn.utils.clip_grad_norm_([loss_fn.logZ], max_norm=0.5)
        
        optimizer.step()
        
        if step % 50 == 0:
            avg_loss = np.mean(loss_history[-50:]) if len(loss_history) >= 50 else np.mean(loss_history)
            avg_reward = np.mean(reward_history[-50*batch_size:]) if len(reward_history) >= 50*batch_size else np.mean(reward_history)
            avg_length = np.mean(batch_lengths)
            elapsed = time.time() - start_time
            
            gaps = []
            for idx, info in best_per_instance.items():
                if info['optimal'] and info['optimal'] > 0:
                    gap = (info['length'] - info['optimal']) / info['optimal'] * 100
                    gaps.append(gap)
            avg_gap = np.mean(gaps) if gaps else 0.0
            
            if step > start_step:
                steps_remaining = steps - step
                time_per_step = elapsed / (step - start_step)
                eta_seconds = steps_remaining * time_per_step
                eta_min, eta_sec = divmod(int(eta_seconds), 60)
                eta_hour, eta_min = divmod(eta_min, 60)
                eta_str = f"{eta_hour}h{eta_min:02d}m" if eta_hour > 0 else f"{eta_min}m{eta_sec:02d}s"
            else:
                eta_str = "--"
            
            print(f"[{step}] loss={cumulative_loss.item():.4f}, log_reward={avg_reward:.2f}, "
                  f"length={avg_length:.1f}, avg_gap={avg_gap:.1f}%, no_improv={steps_without_improvement}, "
                  f"eps={epsilon:.3f}, ETA={eta_str}", flush=True)
            
            log_entry = {
                "type": "step",
                "step": step,
                "loss": float(cumulative_loss.item()),
                "avg_loss": float(avg_loss),
                "avg_reward": float(avg_reward),
                "avg_gap_percent": float(avg_gap),
                "steps_without_improvement": steps_without_improvement,
                "epsilon": float(epsilon),
                "logZ": float(loss_fn.logZ.item()) if hasattr(loss_fn, 'logZ') else None,
                "elapsed_seconds": float(elapsed),
                "best_per_instance": {str(k): float(v['length']) for k, v in best_per_instance.items()},
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
                'best_reward': best_reward,
                'best_per_instance': best_per_instance,
                'problem_name': problem_name,
            }
            if problem_name:
                ckpt_name = f'{problem_name}_step_{step+1}.pt'
            else:
                ckpt_name = f'conditional_checkpoint_step_{step+1}.pt'
            torch.save(checkpoint, os.path.join(save_dir, ckpt_name))
            print(f"  -> Saved checkpoint: {ckpt_name}", flush=True)
        
        gaps = []
        for idx, info in best_per_instance.items():
            if info['optimal'] and info['optimal'] > 0:
                gap = (info['length'] - info['optimal']) / info['optimal'] * 100
                gaps.append(gap)
        avg_gap = np.mean(gaps) if gaps else float('inf')
        
        if avg_gap < prev_best_avg_gap:
            prev_best_avg_gap = avg_gap
            steps_without_improvement = 0
            
            best_checkpoint = {
                'step': step + 1,
                'forward_policy': forward_policy.state_dict(),
                'backward_policy': backward_policy.state_dict(),
                'loss_fn': loss_fn.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_reward': best_reward,
                'best_per_instance': best_per_instance,
                'problem_name': problem_name,
                'avg_gap': avg_gap,
            }
            if problem_name:
                best_ckpt_name = f'{problem_name}_best.pt'
            else:
                best_ckpt_name = 'conditional_best.pt'
            torch.save(best_checkpoint, os.path.join(save_dir, best_ckpt_name))
        else:
            steps_without_improvement += 1
        
        if early_stop_patience and steps_without_improvement >= early_stop_patience:
            print(f"\n⚠ Early stopping: no improvement for {early_stop_patience} steps")
            break
    
    final_checkpoint = {
        'step': step + 1,
        'forward_policy': forward_policy.state_dict(),
        'backward_policy': backward_policy.state_dict(),
        'loss_fn': loss_fn.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_reward': best_reward,
        'best_per_instance': best_per_instance,
        'problem_name': problem_name,
    }
    if problem_name:
        final_ckpt_name = f'{problem_name}_final.pt'
    else:
        final_ckpt_name = 'conditional_final.pt'
    torch.save(final_checkpoint, os.path.join(save_dir, final_ckpt_name))
    print(f"  -> Saved final checkpoint: {final_ckpt_name}", flush=True)
    
    print(f"\n{'='*60}")
    print("Training Complete - Best Results per Instance:")
    print(f"{'='*60}")
    for idx, info in best_per_instance.items():
        opt_str = f"/{info['optimal']:.1f}" if info['optimal'] else ""
        gap_str = f" (gap: {(info['length'] - info['optimal']) / info['optimal'] * 100:.1f}%)" if info['optimal'] else ""
        print(f"  {info['name']}: {info['length']:.1f}{opt_str}{gap_str}")
    
    return best_per_instance
