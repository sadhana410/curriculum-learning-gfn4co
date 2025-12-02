# problems/graph_coloring/trainer.py

import os
import json
import time
from datetime import datetime
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

from problems.graph_coloring.sampler import (
    sample_trajectories_batched, collate_trajectories, get_batched_mask,
    sample_trajectories_conditional, sample_trajectories_conditional_batched,
    collate_trajectories_conditional
)


# Pre-allocate tensors for speed
_inf_tensor = None

def get_inf_tensor(device):
    global _inf_tensor
    if _inf_tensor is None or _inf_tensor.device != device:
        _inf_tensor = torch.tensor(float('-inf'), device=device)
    return _inf_tensor


def plot_sample_distribution(sample_dist, step, chromatic, problem_name, save_path, 
                              history=None, instance_name=None, init_dist=None, init_step=0,
                              best_dist=None, best_step=None):
    """
    Plot and save sample distribution during training.
    
    Args:
        sample_dist: Dict mapping colors -> count (current step) - used as fallback for right plot
        step: Current training step
        chromatic: Chromatic number (optimal)
        problem_name: Name for the plot title
        save_path: Path to save the plot
        history: Optional list of (step, mean, std) tuples for tracking progress (unused)
        instance_name: Optional instance name for conditional training
        init_dist: Optional dict mapping colors -> count for initial distribution
        init_step: Step number for the initial distribution (default 0, but can be different for resumed stages)
        best_dist: Optional dict mapping colors -> count for best distribution (shown on right)
        best_step: Step number when best distribution was achieved
    """
    if not sample_dist:
        return
    
    # Helper function to plot a single histogram
    def plot_histogram(ax, dist, title):
        min_c = min(dist.keys())
        max_c = max(dist.keys())
        
        # Extend range to include chromatic number if needed
        if chromatic:
            min_c = min(min_c, chromatic)
            max_c = max(max_c, chromatic)
        
        x = list(range(min_c, max_c + 1))
        y = [dist.get(c, 0) for c in x]
        total = sum(y)
        y_pct = [100 * v / total if total > 0 else 0 for v in y]
        
        # Color bars
        bar_colors = []
        for c in x:
            if chromatic and c == chromatic:
                bar_colors.append('green')
            elif chromatic and c < chromatic:
                bar_colors.append('darkgreen')
            else:
                bar_colors.append('steelblue')
        
        bars = ax.bar(x, y_pct, color=bar_colors, edgecolor='black', alpha=0.8)
        
        # Add count labels
        for bar, count, pct in zip(bars, y, y_pct):
            if pct > 3:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{count}', ha='center', va='bottom', fontsize=8)
        
        # Statistics
        colors_list = []
        for c, count in dist.items():
            colors_list.extend([c] * count)
        mean_c = np.mean(colors_list) if colors_list else 0
        std_c = np.std(colors_list) if colors_list else 0
        
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Number of Colors', fontsize=10)
        ax.set_ylabel('Percentage (%)', fontsize=10)
        
        stats_text = f"μ={mean_c:.2f}, σ={std_c:.2f}\nn={total}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Always show chromatic number line
        if chromatic:
            ax.axvline(x=chromatic, color='red', linestyle='--', linewidth=2, label=f'χ={chromatic}')
            ax.legend(loc='upper left', fontsize=8)
        
        ax.set_xticks(x)
        ax.grid(axis='y', alpha=0.3)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Left plot: Initial distribution (at init_step)
    ax1 = axes[0]
    if init_dist:
        title0 = f"Step {init_step} (Initial)"
        if instance_name:
            title0 = f"{instance_name} - Step {init_step}"
        plot_histogram(ax1, init_dist, title0)
    else:
        ax1.text(0.5, 0.5, 'No initial data', ha='center', va='center', 
                transform=ax1.transAxes, fontsize=12)
        ax1.set_title(f'Step {init_step} (Initial)', fontsize=11, fontweight='bold')
    
    # Right plot: Best distribution (or current if no best available)
    ax2 = axes[1]
    if best_dist is not None and best_step is not None:
        title_best = f"Step {best_step} (Best)"
        if instance_name:
            title_best = f"{instance_name} - Step {best_step} (Best)"
        plot_histogram(ax2, best_dist, title_best)
    else:
        title_curr = f"Step {step}"
        if instance_name:
            title_curr = f"{instance_name} - Step {step}"
        plot_histogram(ax2, sample_dist, title_curr)
    
    plt.suptitle(problem_name, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


# def compute_logprobs(states, actions, forward_policy, backward_policy, env, device):
#     """Compute forward and backward log probabilities for a trajectory."""
#     N, K = env.N, env.K
#     inf_val = get_inf_tensor(device)
    
#     logprob_f = torch.tensor(0.0, device=device)
#     logprob_b = torch.tensor(0.0, device=device)
    
#     # Pre-compute node indices for backward mask
#     node_indices = torch.arange(N, device=device)
    
#     for i, action in enumerate(actions):
#         state_before = states[i]
#         state_after = states[i + 1]
        
#         # Forward: P(action | state_before)
#         logits_f = forward_policy(state_before, env.adj, device=device)
#         mask_f = torch.from_numpy(env.allowed_actions(state_before)).to(device)
#         logits_f = torch.where(mask_f > 0, logits_f, inf_val)
#         logprob_f = logprob_f + F.log_softmax(logits_f, dim=0)[action]
        
#         # Backward: P(action | state_after)
#         logits_b = backward_policy(state_after, env.adj, device=device)
#         colored_mask = torch.from_numpy(state_after != -1).to(device)
#         colors = torch.from_numpy(np.maximum(state_after, 0)).to(device)
#         backward_mask = torch.zeros(N * K, device=device)
#         action_indices = node_indices * K + colors
#         backward_mask.scatter_(0, action_indices[colored_mask], 1.0)
#         logits_b = torch.where(backward_mask > 0, logits_b, inf_val)
#         logprob_b = logprob_b + F.log_softmax(logits_b, dim=0)[action]
    
#     return logprob_f, logprob_b


def build_backward_masks(states_flat, N, K, device):
    """
    states_flat: (B, N) long, entries in {-1, 0, ..., K-1}
    Returns:
      masks: (B, N*K) bool, True where backward action is allowed
    """
    B = states_flat.shape[0]

    masks = torch.zeros(B, N * K, dtype=torch.bool, device=device)

    node_idx = torch.arange(N, device=device).view(1, N).expand(B, N)  # (B, N)
    colors = torch.clamp(states_flat, min=0)                            # (B, N)
    colored_mask = (states_flat != -1)                                  # (B, N)

    action_idx = node_idx * K + colors                                  # (B, N)
    batch_idx = torch.arange(B, device=device).view(-1, 1).expand(B, N) # (B, N)

    batch_idx_flat = batch_idx[colored_mask]
    action_idx_flat = action_idx[colored_mask]

    masks[batch_idx_flat, action_idx_flat] = True
    return masks  # (B, N*K)


def build_forward_masks(states_flat, env, device):
    """
    states_flat: (B, N) long
    Returns:
      masks: (B, N*K) bool
    """
    B = states_flat.shape[0]
    masks_list = []
    for i in range(B):
        s_np = states_flat[i].cpu().numpy()
        m_np = env.allowed_actions(s_np)  # (N*K,) array {0,1}
        m_t = torch.from_numpy(m_np.astype(np.bool_)).to(device)
        masks_list.append(m_t)
    masks = torch.stack(masks_list, dim=0)  # (B, N*K)
    return masks


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
    N_env, K = env.N, env.K
    assert N == N_env, "states last dim must match env.N"

    T_B = T_max * B

    states_before = states[:-1].reshape(T_B, N)  # (T*B, N)
    states_after  = states[1:].reshape(T_B, N)   # (T*B, N)
    flat_actions  = actions.reshape(T_B)         # (T*B,)
    flat_mask     = step_mask.reshape(T_B)       # (T*B,)

    # Only real (non-padded) steps
    valid_idx = flat_mask.nonzero(as_tuple=False).squeeze(-1)  # (M,)
    if valid_idx.numel() == 0:
        return torch.zeros(B, device=device), torch.zeros(B, device=device)

    states_before_valid = states_before[valid_idx]   # (M, N)
    states_after_valid  = states_after[valid_idx]    # (M, N)
    actions_valid       = flat_actions[valid_idx]    # (M,)

    # Batched policy calls (thanks to updated GNNPolicy)
    logits_f = forward_policy(states_before_valid, env.adj, device=device)  # (M, N*K)
    logits_b = backward_policy(states_after_valid,  env.adj, device=device) # (M, N*K)

    # Masks - using vectorized get_batched_mask for forward
    mask_f = get_batched_mask(states_before_valid, env, device).bool()    # (M, N*K)
    mask_b = build_backward_masks(states_after_valid, N, K, device)       # (M, N*K)

    inf_val = get_inf_tensor(device)
    logits_f = torch.where(mask_f, logits_f, inf_val)
    logits_b = torch.where(mask_b, logits_b, inf_val)

    logp_f_all = F.log_softmax(logits_f, dim=-1)  # (M, N*K)
    logp_b_all = F.log_softmax(logits_b, dim=-1)  # (M, N*K)

    step_logp_f_valid = logp_f_all[
        torch.arange(actions_valid.numel(), device=device),
        actions_valid
    ]
    step_logp_b_valid = logp_b_all[
        torch.arange(actions_valid.numel(), device=device),
        actions_valid
    ]

    # Scatter back into full (T*B,) vectors
    step_logp_f_flat = torch.zeros(T_B, device=device)
    step_logp_b_flat = torch.zeros(T_B, device=device)

    step_logp_f_flat[valid_idx] = step_logp_f_valid
    step_logp_b_flat[valid_idx] = step_logp_b_valid

    # Reshape to (T, B)
    step_logp_f = step_logp_f_flat.view(T_max, B)   # (T, B)
    step_logp_b = step_logp_b_flat.view(T_max, B)   # (T, B)

    # Mask out invalid steps (they are 0.0 already but good to be explicit if needed)
    step_logp_f = step_logp_f * step_mask
    step_logp_b = step_logp_b * step_mask

    return step_logp_f, step_logp_b


def train(env, forward_policy, backward_policy, loss_fn, optimizer,
          steps=2000, device="cpu", save_dir="checkpoints", problem_name=None,
          batch_size=16, epsilon_start=0.3, log_dir="logs", save_every=500,
          early_stop_patience=500, top_p=1.0, temperature=1.0):
    """
    Train GFlowNet for graph coloring.
    """
    last_terminal_state = None
    reward_history = []
    loss_history = []
    best_reward = float('-inf')
    best_colors = env.K
    best_state = None
    
    # Early stopping tracking based on distribution mean
    steps_without_improvement = 0
    prev_mean_colors = float('inf')
    
    # Track distribution history for plotting
    distribution_history = []  # List of (step, mean, std)
    
    # Create checkpoint and log directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"{problem_name}_{timestamp}" if problem_name else f"train_{timestamp}"
    log_path = os.path.join(log_dir, f"{log_name}.jsonl")
    
    # Setup distribution plot path in separate folder
    dist_dir = os.path.join(os.path.dirname(log_dir), "distribution")
    os.makedirs(dist_dir, exist_ok=True)
    plot_path = os.path.join(dist_dir, f"{log_name}_distribution.png")
    
    # Detect loss type
    loss_type = "TB"
    if hasattr(loss_fn, 'lambda_'):
        loss_type = "SubTB"
    elif loss_fn.__class__.__name__ == "DetailedBalance":
        loss_type = "DB"
    
    # Log training config
    config = {
        "type": "config",
        "problem_name": problem_name,
        "steps": steps,
        "batch_size": batch_size,
        "epsilon_start": epsilon_start,
        "top_p": top_p,
        "temperature": temperature,
        "loss_type": loss_type,
        "num_nodes": env.N,
        "num_colors": env.K,
        "device": str(device),
        "timestamp": timestamp,
    }
    with open(log_path, "w") as f:
        f.write(json.dumps(config) + "\n")
    
    print(f"Logging to: {log_path}")
    start_time = time.time()
    
    # Initial step 0 sampling for baseline comparison
    with torch.no_grad():
        init_trajectories = sample_trajectories_batched(
            env, forward_policy, device, batch_size,
            epsilon=epsilon_start, temperature=temperature, top_p=top_p
        )
        init_sample_dist = {}
        for traj_states, traj_actions, reward in init_trajectories:
            last_state = traj_states[-1]
            colored = np.sum(last_state != -1)
            if colored == env.N:
                colors_in_state = len(set(c for c in last_state if c != -1))
                conflicts = 0
                adj = env._adj_np if hasattr(env, '_adj_np') else env.adj
                for u in range(env.N):
                    for v in range(u + 1, env.N):
                        if adj[u, v] == 1 and last_state[u] == last_state[v]:
                            conflicts += 1
                if conflicts == 0:
                    init_sample_dist[colors_in_state] = init_sample_dist.get(colors_in_state, 0) + 1
        
        # Record step 0 in history and plot
        if init_sample_dist:
            colors_list = []
            for c, count in init_sample_dist.items():
                colors_list.extend([c] * count)
            if colors_list:
                mean_colors = np.mean(colors_list)
                std_colors = np.std(colors_list)
                distribution_history.append((0, mean_colors, std_colors))
                chromatic = getattr(env, 'chromatic_number', None)
                plot_sample_distribution(
                    init_sample_dist, 0, chromatic, problem_name or "Training",
                    plot_path, history=distribution_history
                )
                print(f"[0] Initial distribution: {mean_colors:.2f}±{std_colors:.2f} colors, range=[{min(init_sample_dist.keys())},{max(init_sample_dist.keys())}], valid={sum(init_sample_dist.values())}/{batch_size}")

    for step in range(steps):
        # Epsilon decays from epsilon_start to 0.01
        epsilon = max(0.01, epsilon_start * (1 - step / steps))
        
        # Sample batch of trajectories in parallel
        batch_trajectories = sample_trajectories_batched(
            env, forward_policy, device, batch_size, 
            epsilon=epsilon, temperature=temperature, top_p=top_p
        )
        
        batch_rewards = []
        
        # Track sample distribution for this batch
        batch_sample_dist = {}  # colors -> count

        # First pass: logging / best-color tracking (same logic as before)
        for traj_states, traj_actions, reward in batch_trajectories:
            last_terminal_state = traj_states[-1]
            batch_rewards.append(reward)

            if reward > best_reward:
                best_reward = reward

            # Track best (lowest) colors for valid complete coloring
            colored = np.sum(last_terminal_state != -1)
            if colored == env.N:
                colors_in_state = len(set(c for c in last_terminal_state if c != -1))
                # Check if valid (no conflicts)
                conflicts = 0
                adj = env._adj_np if hasattr(env, '_adj_np') else env.adj
                for u in range(env.N):
                    for v in range(u + 1, env.N):
                        if adj[u, v] == 1 and last_terminal_state[u] == last_terminal_state[v]:
                            conflicts += 1
                
                # Track sample distribution (only valid colorings)
                if conflicts == 0:
                    batch_sample_dist[colors_in_state] = batch_sample_dist.get(colors_in_state, 0) + 1
                
                if conflicts == 0 and colors_in_state < best_colors:
                    best_colors = colors_in_state
                    best_state = last_terminal_state.copy()

        # Collate trajectories to padded tensors
        states, actions, step_mask, rewards_tensor = collate_trajectories(
            batch_trajectories, env, device
        )

        # Batched log-probabilities (per step)
        step_log_pf, step_log_pb = compute_logprobs_batched(
            states, actions, step_mask,
            forward_policy, backward_policy, env, device
        )

        logreward_batch = rewards_tensor  # (B,) - env returns logR now

        # Compute Loss
        if loss_type == "TB":
            # Sum over steps
            log_pf_sum = (step_log_pf * step_mask).sum(dim=0) # (B,)
            log_pb_sum = (step_log_pb * step_mask).sum(dim=0) # (B,)
            traj_losses = loss_fn(log_pf_sum, log_pb_sum, logreward_batch)
            cumulative_loss = traj_losses.mean()
            
        else: # DB or SubTB
            # Need state flows F(s)
            # states is (T+1, B, N)
            T_plus_1, B, N = states.shape
            flat_states = states.reshape(-1, N)
            
            # Predict flows for all states
            flat_log_flows = forward_policy.predict_flow(flat_states, env.adj, device) # ((T+1)*B,)
            log_flows = flat_log_flows.view(T_plus_1, B)
            
            if loss_type == "DB":
                cumulative_loss = loss_fn(step_log_pf, step_log_pb, log_flows, logreward_batch, step_mask)
            else: # SubTB
                cumulative_loss = loss_fn(step_log_pf, step_log_pb, log_flows, logreward_batch, step_mask)

        reward_history.extend(batch_rewards)
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
            avg_loss = np.mean(loss_history[-50:]) if len(loss_history) >= 50 else np.mean(loss_history)
            avg_reward = np.mean(reward_history[-50*batch_size:]) if len(reward_history) >= 50*batch_size else np.mean(reward_history)
            colored = np.sum(last_terminal_state != -1)
            colors_used = len(set(c for c in last_terminal_state if c != -1))
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
            
            chromatic = getattr(env, 'chromatic_number', None)
            chromatic_str = f"/{chromatic}" if chromatic else ""
            gap = best_colors - chromatic if chromatic else 0
            print(f"[{step}] loss={cumulative_loss.item():.4f}, log_reward={avg_reward:.2f}, colored={colored}/{env.N}, colors={colors_used}/{env.K}, best={best_colors}{chromatic_str}, gap={gap}, no_improv={steps_without_improvement}, eps={epsilon:.3f}, ETA={eta_str}", flush=True)
            
            # Print sample distribution (mean ± std) and update plot
            if batch_sample_dist:
                colors_list = []
                for c, count in batch_sample_dist.items():
                    colors_list.extend([c] * count)
                if colors_list:
                    mean_colors = np.mean(colors_list)
                    std_colors = np.std(colors_list)
                    min_colors = min(batch_sample_dist.keys())
                    max_colors = max(batch_sample_dist.keys())
                    total_valid = sum(batch_sample_dist.values())
                    print(f"    samples: {mean_colors:.2f}±{std_colors:.2f} colors, range=[{min_colors},{max_colors}], valid={total_valid}/{batch_size}", flush=True)
                    
                    # Track history and update plot
                    distribution_history.append((step, mean_colors, std_colors))
                    plot_sample_distribution(
                        batch_sample_dist, step, chromatic, problem_name or "Training",
                        plot_path, history=distribution_history, init_dist=init_sample_dist
                    )
            
            # Log to file
            log_entry = {
                "type": "step",
                "step": step,
                "loss": cumulative_loss.item(),
                "avg_loss": avg_loss,
                "avg_reward": avg_reward,
                "best_colors": best_colors,
                "chromatic_number": chromatic,
                "gap": gap,
                "steps_without_improvement": steps_without_improvement,
                "colored": int(colored),
                "colors_used": colors_used,
                "epsilon": epsilon,
                "logZ": loss_fn.logZ.item() if hasattr(loss_fn, 'logZ') else None,
                "elapsed_seconds": elapsed,
                "batch_sample_dist": batch_sample_dist,
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        
        # Save checkpoint periodically
        if (step + 1) % save_every == 0:
            checkpoint = {
                'step': step + 1,
                'forward_policy': forward_policy.state_dict(),
                'backward_policy': backward_policy.state_dict(),
                'loss_fn': loss_fn.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_reward': best_reward,
                'best_colors': best_colors,
                'problem_name': problem_name,
            }
            if problem_name:
                ckpt_name = f'{problem_name}_step_{step+1}.pt'
            else:
                ckpt_name = f'checkpoint_step_{step+1}.pt'
            torch.save(checkpoint, os.path.join(save_dir, ckpt_name))
            print(f"  -> Saved checkpoint: {ckpt_name}", flush=True)
        
        # Early stopping check based on distribution mean improvement
        chromatic = getattr(env, 'chromatic_number', None)
        
        # Compute current mean colors from batch distribution
        current_mean_colors = float('inf')
        if batch_sample_dist:
            colors_list = []
            for c, count in batch_sample_dist.items():
                colors_list.extend([c] * count)
            if colors_list:
                current_mean_colors = np.mean(colors_list)
        
        # Check if distribution is improving (mean colors decreasing)
        if current_mean_colors < prev_mean_colors - 0.01:  # Small threshold to avoid noise
            prev_mean_colors = current_mean_colors
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1
        
        # Stop if no improvement for patience steps (disabled if patience <= 0)
        if early_stop_patience > 0 and steps_without_improvement >= early_stop_patience:
            print(f"\n⚠ Early stopping: distribution not improving for {early_stop_patience} steps")
            print(f"  Mean colors: {current_mean_colors:.2f}, Best: {best_colors}, Chromatic: {chromatic}")
            break

    return best_state if best_state is not None else last_terminal_state, best_colors


# ============================================================================
# Conditional GFlowNet Training
# ============================================================================

def build_backward_masks_conditional(states_flat, N, K, device):
    """
    Build backward masks for conditional GFlowNet.
    Same as build_backward_masks but works with variable N.
    """
    B = states_flat.shape[0]
    masks = torch.zeros(B, N * K, dtype=torch.bool, device=device)
    
    node_idx = torch.arange(N, device=device).view(1, N).expand(B, N)
    colors = torch.clamp(states_flat, min=0)
    colored_mask = (states_flat != -1)
    
    action_idx = node_idx * K + colors
    batch_idx = torch.arange(B, device=device).view(-1, 1).expand(B, N)
    
    batch_idx_flat = batch_idx[colored_mask]
    action_idx_flat = action_idx[colored_mask]
    
    masks[batch_idx_flat, action_idx_flat] = True
    return masks


def get_batched_mask_conditional(states, adj, K, device):
    """
    Compute allowed actions mask for conditional GFlowNet.
    
    Args:
        states: (B, N) tensor of node colors
        adj: (N, N) or (B, N, N) adjacency matrix
        K: number of colors
        device: torch device
        
    Returns:
        mask: (B, N*K) boolean tensor
    """
    if isinstance(states, np.ndarray):
        states_t = torch.from_numpy(states).to(device=device, dtype=torch.long)
    else:
        states_t = states.to(dtype=torch.long)
    
    B, N = states_t.shape
    
    if isinstance(adj, np.ndarray):
        adj_t = torch.from_numpy(adj).to(device=device, dtype=torch.float32)
    else:
        adj_t = adj.to(device).float()
    
    # Handle batch dimension for adj
    if adj_t.dim() == 2:
        adj_t = adj_t.unsqueeze(0).expand(B, -1, -1)
    
    # Identify uncolored nodes
    uncolored_mask = (states_t == -1)
    
    # Compute neighbors' colors
    colors_onehot = torch.zeros(B, N, K, device=device)
    colored_mask = (states_t != -1)
    batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, N)
    node_idx = torch.arange(N, device=device).view(1, N).expand(B, N)
    
    valid_indices = colored_mask
    if valid_indices.any():
        b_idx = batch_idx[valid_indices]
        n_idx = node_idx[valid_indices]
        c_idx = states_t[valid_indices]
        colors_onehot[b_idx, n_idx, c_idx] = 1.0
    
    # Propagate to neighbors
    forbidden = torch.bmm(adj_t, colors_onehot)
    
    # Valid if not forbidden AND node is uncolored
    valid_colors = (forbidden == 0)
    node_valid = uncolored_mask.unsqueeze(-1).expand(-1, -1, K)
    final_mask = node_valid & valid_colors
    
    return final_mask.reshape(B, -1)


def compute_logprobs_conditional(states, actions, step_mask, adj,
                                 forward_policy, backward_policy, K, device):
    """
    Compute log probabilities for conditional GFlowNet.
    
    Args:
        states: (T_max+1, B, N) tensor
        actions: (T_max, B) tensor
        step_mask: (T_max, B) boolean tensor
        adj: (N, N) or (B, N, N) adjacency matrix
        forward_policy: ConditionalGNNPolicy
        backward_policy: ConditionalGNNPolicy
        K: number of colors
        device: torch device
        
    Returns:
        step_log_pf: (T_max, B) tensor
        step_log_pb: (T_max, B) tensor
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
    
    # Batched policy calls
    logits_f = forward_policy(states_before_valid, adj, device=device)
    logits_b = backward_policy(states_after_valid, adj, device=device)
    
    # Masks
    mask_f = get_batched_mask_conditional(states_before_valid, adj, K, device).bool()
    mask_b = build_backward_masks_conditional(states_after_valid, N, K, device)
    
    inf_val = get_inf_tensor(device)
    logits_f = torch.where(mask_f, logits_f, inf_val)
    logits_b = torch.where(mask_b, logits_b, inf_val)
    
    logp_f_all = F.log_softmax(logits_f, dim=-1)
    logp_b_all = F.log_softmax(logits_b, dim=-1)
    
    step_logp_f_valid = logp_f_all[torch.arange(actions_valid.numel(), device=device), actions_valid]
    step_logp_b_valid = logp_b_all[torch.arange(actions_valid.numel(), device=device), actions_valid]
    
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


def find_latest_checkpoint(save_dir, problem_name=None):
    """Find the latest checkpoint in save_dir."""
    if not os.path.exists(save_dir):
        return None
    
    checkpoints = []
    for f in os.listdir(save_dir):
        if not f.endswith('.pt'):
            continue
        # Filter by problem_name if provided
        if problem_name and not f.startswith(problem_name):
            continue
        checkpoints.append(os.path.join(save_dir, f))
    
    if not checkpoints:
        return None
    
    # Sort by step number
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
                      same_instance_per_batch=True, resume_from=None,
                      prev_stage_distributions=None, current_stage_instance=None):
    """
    Train Conditional GFlowNet for graph coloring on multiple graph instances.
    
    The policy learns to generalize across different graph structures.
    
    Args:
        env: ConditionalGraphColoringEnv with multiple instances
        forward_policy: ConditionalGNNPolicy
        backward_policy: ConditionalGNNPolicy
        loss_fn: Loss function (TB, DB, or SubTB)
        optimizer: Optimizer
        steps: Number of training steps
        device: torch device
        save_dir: Directory to save checkpoints
        problem_name: Name for logging
        batch_size: Number of trajectories per step
        epsilon_start: Initial exploration rate
        log_dir: Directory for logs
        save_every: Checkpoint frequency
        early_stop_patience: Steps without improvement before stopping
        top_p: Nucleus sampling threshold
        temperature: Sampling temperature
        same_instance_per_batch: If True, all trajectories in a batch use same instance
        resume_from: Path to checkpoint to resume from (or 'latest' to find latest)
        prev_stage_distributions: Dict mapping instance names to their best distributions 
                                  from previous stages (for plotting comparison)
        current_stage_instance: Name of the current stage's instance (for best checkpoint criterion).
                               If None, uses all instances for the criterion.
    """
    reward_history = []
    loss_history = []
    best_reward = float('-inf')
    start_step = 0
    
    # Track best coloring per instance
    best_per_instance = {}
    for i in range(env.num_instances):
        inst = env.get_instance(i)
        best_per_instance[i] = {
            'colors': inst['N'],  # worst case
            'state': None,
            'name': inst['name']
        }
    
    # Early stopping based on distribution mean improvement
    steps_without_improvement = 0
    prev_mean_colors_avg = float('inf')
    
    # Track distribution history per instance for plotting
    distribution_history = {}  # instance_idx -> list of (step, mean, std)
    for i in range(env.num_instances):
        distribution_history[i] = []
    
    # Track best distribution per instance for plotting
    best_dist_per_instance = {}  # instance_idx -> {'dist': {colors: count}, 'step': step, 'mean': mean}
    for i in range(env.num_instances):
        best_dist_per_instance[i] = {'dist': None, 'step': None, 'mean': float('inf')}
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Handle resume from checkpoint
    if resume_from:
        checkpoint_path = None
        
        if resume_from == 'latest':
            # Find latest checkpoint
            checkpoint_path = find_latest_checkpoint(save_dir, problem_name)
        elif os.path.exists(resume_from):
            checkpoint_path = resume_from
        else:
            # Try to find in save_dir
            potential_path = os.path.join(save_dir, resume_from)
            if os.path.exists(potential_path):
                checkpoint_path = potential_path
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            
            # Load model states
            forward_policy.load_state_dict(checkpoint['forward_policy'])
            backward_policy.load_state_dict(checkpoint['backward_policy'])
            loss_fn.load_state_dict(checkpoint['loss_fn'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            
            # Restore training state
            start_step = checkpoint.get('step', 0)
            best_reward = checkpoint.get('best_reward', float('-inf'))
            
            # Restore best_per_instance if available
            if 'best_per_instance' in checkpoint:
                saved_best = checkpoint['best_per_instance']
                for idx in best_per_instance:
                    if idx in saved_best:
                        best_per_instance[idx]['colors'] = saved_best[idx]['colors']
                        best_per_instance[idx]['state'] = saved_best[idx].get('state')
            
            prev_best_avg = np.mean([v['colors'] for v in best_per_instance.values()])
            
            print(f"  Resumed at step {start_step}")
            print(f"  Best reward: {best_reward:.2f}")
            print(f"  Avg best colors: {prev_best_avg:.2f}")
            print()
        else:
            print(f"Warning: Checkpoint not found: {resume_from}")
            print("Starting from scratch...")
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if resume_from and start_step > 0:
        log_name = f"{problem_name}_resumed_{timestamp}" if problem_name else f"train_conditional_resumed_{timestamp}"
    else:
        log_name = f"{problem_name}_{timestamp}" if problem_name else f"train_conditional_{timestamp}"
    log_path = os.path.join(log_dir, f"{log_name}.jsonl")
    
    # Setup distribution plot paths in separate folder (one per instance)
    dist_dir = os.path.join(os.path.dirname(log_dir), "distribution")
    os.makedirs(dist_dir, exist_ok=True)
    plot_paths = {}
    for i in range(env.num_instances):
        inst_name = best_per_instance[i]['name']
        plot_paths[i] = os.path.join(dist_dir, f"{log_name}_{inst_name}_distribution.png")
    
    # Detect loss type
    loss_type = "TB"
    if hasattr(loss_fn, 'lambda_'):
        loss_type = "SubTB"
    elif loss_fn.__class__.__name__ == "DetailedBalance":
        loss_type = "DB"
    
    # Log config
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
        "num_colors": env.K,
        "same_instance_per_batch": same_instance_per_batch,
        "device": str(device),
        "timestamp": timestamp,
        "resumed_from": resume_from if resume_from else None,
    }
    with open(log_path, "w") as f:
        f.write(json.dumps(config) + "\n")
    
    # steps parameter is the number of steps to train FOR THIS STAGE
    # When resuming, we train from start_step to start_step + steps
    end_step = start_step + steps
    
    print(f"Conditional GFlowNet Training")
    print(f"  Instances: {env.num_instances}")
    print(f"  Colors: {env.K}")
    print(f"  Loss: {loss_type}")
    if start_step > 0:
        print(f"  Resuming from step: {start_step}")
    print(f"  Steps this stage: {steps}")
    print(f"  End step: {end_step}")
    print(f"  Logging to: {log_path}")
    print()
    
    start_time = time.time()
    
    # Store initial distributions for plotting comparison
    init_sample_dist = {}  # instance_idx -> {colors: count}
    init_step_per_instance = {}  # instance_idx -> step number for initial distribution
    
    # For instances with previous stage distributions, use those as initial
    if prev_stage_distributions:
        for inst_idx in range(env.num_instances):
            inst_name = best_per_instance[inst_idx]['name']
            if inst_name in prev_stage_distributions:
                prev_info = prev_stage_distributions[inst_name]
                init_sample_dist[inst_idx] = prev_info['distribution']
                init_step_per_instance[inst_idx] = prev_info.get('step', 0)
    
    # Sample initial distribution at the start of this stage (for baseline comparison)
    # Sample for EACH instance that doesn't have a previous stage distribution
    with torch.no_grad():
        for inst_idx in range(env.num_instances):
            # Skip instances that already have previous stage distribution
            if inst_idx in init_sample_dist:
                continue
            
            # Sample a batch specifically for this instance
            init_trajectories = sample_trajectories_conditional_batched(
                env, forward_policy, device, batch_size,
                epsilon=epsilon_start, temperature=temperature, top_p=top_p,
                same_instance=True, instance_idx=inst_idx
            )
            
            init_sample_dist[inst_idx] = {}
            init_step_per_instance[inst_idx] = start_step
            
            for states_list, actions_list, reward, instance_idx, adj in init_trajectories:
                final_state = states_list[-1]
                N = len(final_state)
                colored = np.sum(final_state != -1)
                if colored == N:
                    colors_used = len(set(c for c in final_state if c != -1))
                    conflicts = 0
                    for u in range(N):
                        for v in range(u + 1, N):
                            if adj[u, v] == 1 and final_state[u] == final_state[v]:
                                conflicts += 1
                    if conflicts == 0:
                        init_sample_dist[inst_idx][colors_used] = init_sample_dist[inst_idx].get(colors_used, 0) + 1
        
        # Record initial distribution in history and plot for each instance
        init_label = f"[{start_step}] Initial distribution:"
        print(init_label)
        for inst_idx, dist in sorted(init_sample_dist.items()):
            if dist:
                colors_list = []
                for c, count in dist.items():
                    colors_list.extend([c] * count)
                if colors_list:
                    mean_colors = np.mean(colors_list)
                    std_colors = np.std(colors_list)
                    inst_init_step = init_step_per_instance.get(inst_idx, start_step)
                    distribution_history[inst_idx].append((inst_init_step, mean_colors, std_colors))
                    inst_name = best_per_instance[inst_idx]['name']
                    chromatic = env.get_instance(inst_idx).get('chromatic_number', None)
                    chromatic_str = chromatic if chromatic else '?'
                    # Mark if this is from previous stage
                    prev_marker = " [prev stage]" if inst_idx in init_step_per_instance and init_step_per_instance[inst_idx] != start_step else ""
                    print(f"    {inst_name} (χ={chromatic_str}): {mean_colors:.2f}±{std_colors:.2f} colors, range=[{min(dist.keys())},{max(dist.keys())}], valid={sum(dist.values())}/{batch_size}{prev_marker}")
                    plot_sample_distribution(
                        dist, inst_init_step, chromatic, f"{problem_name or 'Training'} - {inst_name}",
                        plot_paths[inst_idx], history=distribution_history[inst_idx],
                        instance_name=inst_name
                    )
    
    # Initialize step in case loop doesn't run
    step = start_step
    
    for step in range(start_step, end_step):
        # Epsilon decays based on progress within this stage
        stage_progress = (step - start_step) / steps
        epsilon = max(0.01, epsilon_start * (1 - stage_progress))
        
        # Sample trajectories
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
        
        # Track sample distribution per instance for this batch
        batch_sample_dist = {}  # instance_idx -> {colors: count}
        
        # Track best colorings
        for states_list, actions_list, reward, instance_idx, adj in batch_trajectories:
            batch_rewards.append(reward)
            
            if reward > best_reward:
                best_reward = reward
            
            # Check if valid coloring
            final_state = states_list[-1]
            N = len(final_state)
            colored = np.sum(final_state != -1)
            
            if colored == N:
                colors_used = len(set(c for c in final_state if c != -1))
                # Check conflicts
                conflicts = 0
                for u in range(N):
                    for v in range(u + 1, N):
                        if adj[u, v] == 1 and final_state[u] == final_state[v]:
                            conflicts += 1
                
                # Track sample distribution (only valid colorings)
                if conflicts == 0:
                    if instance_idx not in batch_sample_dist:
                        batch_sample_dist[instance_idx] = {}
                    batch_sample_dist[instance_idx][colors_used] = batch_sample_dist[instance_idx].get(colors_used, 0) + 1
                
                if conflicts == 0 and colors_used < best_per_instance[instance_idx]['colors']:
                    best_per_instance[instance_idx]['colors'] = colors_used
                    best_per_instance[instance_idx]['state'] = final_state.copy()
        
        # Collate trajectories (grouped by graph size)
        collated = collate_trajectories_conditional(batch_trajectories, env.K, device)
        
        # Compute loss for each size group
        total_loss = torch.tensor(0.0, device=device)
        total_trajs = 0
        
        for N, data in collated.items():
            states = data['states']
            actions = data['actions']
            step_mask = data['step_mask']
            rewards_tensor = data['rewards']
            adj = data['adj']
            B_group = states.shape[1]
            
            # Compute log probabilities
            step_log_pf, step_log_pb = compute_logprobs_conditional(
                states, actions, step_mask, adj,
                forward_policy, backward_policy, env.K, device
            )
            
            logreward_batch = rewards_tensor
            
            if loss_type == "TB":
                log_pf_sum = (step_log_pf * step_mask).sum(dim=0)
                log_pb_sum = (step_log_pb * step_mask).sum(dim=0)
                traj_losses = loss_fn(log_pf_sum, log_pb_sum, logreward_batch)
                total_loss = total_loss + traj_losses.sum()
            else:
                # DB or SubTB
                T_plus_1, B, _ = states.shape
                flat_states = states.reshape(-1, N)
                flat_log_flows = forward_policy.predict_flow(flat_states, adj, device)
                log_flows = flat_log_flows.view(T_plus_1, B)
                
                if loss_type == "DB":
                    group_loss = loss_fn(step_log_pf, step_log_pb, log_flows, logreward_batch, step_mask)
                else:
                    group_loss = loss_fn(step_log_pf, step_log_pb, log_flows, logreward_batch, step_mask)
                total_loss = total_loss + group_loss * B_group
            
            total_trajs += B_group
        
        # Average loss
        cumulative_loss = total_loss / max(total_trajs, 1)
        
        reward_history.extend(batch_rewards)
        loss_history.append(cumulative_loss.item())
        
        optimizer.zero_grad()
        cumulative_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(forward_policy.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(backward_policy.parameters(), max_norm=0.5)
        if hasattr(loss_fn, 'logZ'):
            torch.nn.utils.clip_grad_norm_([loss_fn.logZ], max_norm=0.5)
        
        optimizer.step()
        
        # Logging
        if step % 50 == 0:
            avg_loss = np.mean(loss_history[-50:]) if len(loss_history) >= 50 else np.mean(loss_history)
            avg_reward = np.mean(reward_history[-50*batch_size:]) if len(reward_history) >= 50*batch_size else np.mean(reward_history)
            elapsed = time.time() - start_time
            
            # Calculate average mean colors from current batch distribution
            avg_colors = float('inf')
            if batch_sample_dist:
                mean_colors_list = []
                for inst_idx, dist in batch_sample_dist.items():
                    if dist:
                        colors_list = []
                        for c, count in dist.items():
                            colors_list.extend([c] * count)
                        if colors_list:
                            mean_colors_list.append(np.mean(colors_list))
                if mean_colors_list:
                    avg_colors = np.mean(mean_colors_list)
            
            # ETA based on steps remaining in this stage
            if step > start_step:
                steps_remaining = end_step - step
                steps_done = step - start_step
                time_per_step = elapsed / steps_done
                eta_seconds = steps_remaining * time_per_step
                eta_min, eta_sec = divmod(int(eta_seconds), 60)
                eta_hour, eta_min = divmod(eta_min, 60)
                eta_str = f"{eta_hour}h{eta_min:02d}m" if eta_hour > 0 else f"{eta_min}m{eta_sec:02d}s"
            else:
                eta_str = "--"
            
            print(f"[{step}] loss={cumulative_loss.item():.4f}, log_reward={avg_reward:.2f}, "
                  f"avg_colors={avg_colors:.2f}, no_improv={steps_without_improvement}, "
                  f"eps={epsilon:.3f}, ETA={eta_str}", flush=True)
            
            # Print sample distribution for this batch (mean ± std) and update plots
            if batch_sample_dist:
                for inst_idx, dist in sorted(batch_sample_dist.items()):
                    inst_name = best_per_instance[inst_idx]['name']
                    chromatic = env.get_instance(inst_idx).get('chromatic_number', None)
                    total_valid = sum(dist.values())
                    # Compute mean and std of colors
                    colors_list = []
                    for c, count in dist.items():
                        colors_list.extend([c] * count)
                    if colors_list:
                        mean_colors = np.mean(colors_list)
                        std_colors = np.std(colors_list)
                        min_colors = min(dist.keys())
                        max_colors = max(dist.keys())
                        chromatic_str = chromatic if chromatic else '?'
                        print(f"    {inst_name} (χ={chromatic_str}): {mean_colors:.2f}±{std_colors:.2f} colors, range=[{min_colors},{max_colors}], valid={total_valid}/{batch_size}", flush=True)
                        
                        # Track history and update plot for this instance
                        distribution_history[inst_idx].append((step, mean_colors, std_colors))
                        
                        # Update best distribution for this instance if improved
                        if mean_colors < best_dist_per_instance[inst_idx]['mean']:
                            best_dist_per_instance[inst_idx] = {
                                'dist': dict(dist),  # Copy the distribution
                                'step': step,
                                'mean': mean_colors
                            }
                        
                        # Get initial distribution for this instance (if available)
                        inst_init_dist = init_sample_dist.get(inst_idx, None)
                        inst_init_step = init_step_per_instance.get(inst_idx, start_step)
                        plot_sample_distribution(
                            dist, step, chromatic, f"{problem_name or 'Training'} - {inst_name}",
                            plot_paths[inst_idx], history=distribution_history[inst_idx],
                            instance_name=inst_name, init_dist=inst_init_dist, init_step=inst_init_step,
                            best_dist=best_dist_per_instance[inst_idx]['dist'],
                            best_step=best_dist_per_instance[inst_idx]['step']
                        )
            
            # Log to file
            log_entry = {
                "type": "step",
                "step": step,
                "loss": cumulative_loss.item(),
                "avg_loss": avg_loss,
                "avg_reward": avg_reward,
                "avg_colors": avg_colors,
                "steps_without_improvement": steps_without_improvement,
                "epsilon": epsilon,
                "logZ": loss_fn.logZ.item() if hasattr(loss_fn, 'logZ') else None,
                "elapsed_seconds": elapsed,
                "best_per_instance": {str(k): v['colors'] for k, v in best_per_instance.items()},
                "batch_sample_dist": {str(k): v for k, v in batch_sample_dist.items()},
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        
        # Save checkpoint
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
                'num_colors': env.K,
            }
            if problem_name:
                ckpt_name = f'{problem_name}_step_{step+1}.pt'
            else:
                ckpt_name = f'conditional_checkpoint_step_{step+1}.pt'
            torch.save(checkpoint, os.path.join(save_dir, ckpt_name))
            print(f"  -> Saved checkpoint: {ckpt_name}", flush=True)
        
        # Early stopping check based on distribution mean improvement
        # Only consider current stage instance if specified, otherwise all instances
        current_mean_colors_list = []
        for inst_idx, dist in batch_sample_dist.items():
            # If current_stage_instance is set, only use that instance for the criterion
            if current_stage_instance is not None:
                inst_name = best_per_instance[inst_idx]['name']
                if inst_name != current_stage_instance:
                    continue
            if dist:
                colors_list = []
                for c, count in dist.items():
                    colors_list.extend([c] * count)
                if colors_list:
                    current_mean_colors_list.append(np.mean(colors_list))
        
        current_mean_colors_avg = np.mean(current_mean_colors_list) if current_mean_colors_list else float('inf')
        
        # Use small epsilon to avoid floating point issues, save when truly better
        if current_mean_colors_avg < prev_mean_colors_avg - 1e-6:
            prev_mean_colors_avg = current_mean_colors_avg
            steps_without_improvement = 0
            
            # Save best checkpoint when distribution improves
            avg_best_colors = np.mean([v['colors'] for v in best_per_instance.values()])
            # Save distributions per instance for use in subsequent stages
            best_distributions = {}
            for inst_idx, dist in batch_sample_dist.items():
                inst_name = best_per_instance[inst_idx]['name']
                best_distributions[inst_name] = {
                    'distribution': dist,
                    'step': step + 1
                }
            best_checkpoint = {
                'step': step + 1,
                'forward_policy': forward_policy.state_dict(),
                'backward_policy': backward_policy.state_dict(),
                'loss_fn': loss_fn.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_reward': best_reward,
                'best_per_instance': best_per_instance,
                'best_distributions': best_distributions,
                'problem_name': problem_name,
                'num_colors': env.K,
                'avg_best_colors': avg_best_colors,
                'mean_colors_avg': current_mean_colors_avg,
            }
            if problem_name:
                best_ckpt_name = f'{problem_name}_best.pt'
            else:
                best_ckpt_name = 'conditional_best.pt'
            torch.save(best_checkpoint, os.path.join(save_dir, best_ckpt_name))
            criterion_info = f" [{current_stage_instance}]" if current_stage_instance else ""
            print(f"  -> Saved best checkpoint: {best_ckpt_name} (avg_colors={current_mean_colors_avg:.2f}{criterion_info})", flush=True)
        else:
            steps_without_improvement += 1
        
        # Stop if no improvement for patience steps (disabled if patience <= 0)
        if early_stop_patience > 0 and steps_without_improvement >= early_stop_patience:
            avg_best_colors = np.mean([v['colors'] for v in best_per_instance.values()])
            print(f"\n⚠ Early stopping: distribution not improving for {early_stop_patience} steps")
            print(f"  Mean colors: {current_mean_colors_avg:.2f}, Best avg: {avg_best_colors:.2f}")
            break
    
    # Save final checkpoint
    final_checkpoint = {
        'step': step + 1,
        'forward_policy': forward_policy.state_dict(),
        'backward_policy': backward_policy.state_dict(),
        'loss_fn': loss_fn.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_reward': best_reward,
        'best_per_instance': best_per_instance,
        'problem_name': problem_name,
        'num_colors': env.K,
    }
    if problem_name:
        final_ckpt_name = f'{problem_name}_final.pt'
    else:
        final_ckpt_name = 'conditional_final.pt'
    torch.save(final_checkpoint, os.path.join(save_dir, final_ckpt_name))
    print(f"  -> Saved final checkpoint: {final_ckpt_name}", flush=True)
    
    # Final summary
    print(f"\n{'='*60}")
    print("Training Complete - Best Results per Instance:")
    print(f"{'='*60}")
    for idx, info in best_per_instance.items():
        inst = env.get_instance(idx)
        chromatic = inst.get('chromatic_number', '?')
        print(f"  {info['name']}: {info['colors']} colors (chromatic: {chromatic})")
    
    # Build initial distribution info for logging
    init_dist_info = {}
    for inst_idx, dist in init_sample_dist.items():
        inst_name = best_per_instance[inst_idx]['name']
        if dist:
            colors_list = []
            for c, count in dist.items():
                colors_list.extend([c] * count)
            init_dist_info[inst_name] = {
                'distribution': {str(k): v for k, v in dist.items()},  # JSON keys must be strings
                'mean': float(np.mean(colors_list)) if colors_list else None,
                'std': float(np.std(colors_list)) if colors_list else None,
                'step': init_step_per_instance.get(inst_idx, start_step),
            }
    
    return best_per_instance, init_dist_info
