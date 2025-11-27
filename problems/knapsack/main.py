# problems/knapsack/main.py

import argparse
import os
import sys
import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from problems.knapsack.env import KnapsackEnv
from problems.knapsack.policy import KnapsackPolicy
from problems.knapsack.utils import load_knapsack_instance, list_knapsack_instances, get_instance_info
from losses.trajectorybalance import TrajectoryBalance


# Data directory relative to this file
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def sample_trajectory(env, forward_policy, device, epsilon=0.1):
    """Sample a trajectory with epsilon-greedy exploration."""
    import random
    
    state = env.reset()
    traj_states, traj_actions = [], []

    done = False
    while not done:
        traj_states.append(state.copy())

        logits = forward_policy(state, device=device)
        mask = torch.tensor(env.allowed_actions(state), dtype=torch.float32, device=device)
        
        if mask.sum() == 0:
            done = True
            reward = env.reward(state)
            break
        
        if random.random() < epsilon:
            valid_actions = torch.where(mask > 0)[0]
            action = valid_actions[torch.randint(len(valid_actions), (1,))].item()
        else:
            masked_logits = logits.clone()
            masked_logits[mask == 0] = float('-inf')
            probs = torch.softmax(masked_logits, dim=0)
            action = torch.multinomial(probs, 1).item()

        next_state, reward, done = env.step(state, action)
        traj_actions.append(action)
        state = next_state

    traj_states.append(state.copy())
    return traj_states, traj_actions, reward


def compute_logprobs(traj_states, traj_actions, forward_policy, backward_policy, env, device):
    """Compute forward and backward log probabilities for a trajectory."""
    import torch.nn.functional as F
    
    logprob_f = torch.tensor(0.0, device=device)
    logprob_b = torch.tensor(0.0, device=device)
    inf_tensor = torch.tensor(float('-inf'), device=device)
    
    for i, action in enumerate(traj_actions):
        state_before = traj_states[i]
        state_after = traj_states[i + 1]
        
        # Forward
        logits_f = forward_policy(state_before, device=device)
        mask_f = torch.tensor(env.allowed_actions(state_before), dtype=torch.float32, device=device)
        logits_f = torch.where(mask_f > 0, logits_f, inf_tensor)
        logprob_f = logprob_f + F.log_softmax(logits_f, dim=0)[action]
        
        # Backward
        logits_b = backward_policy(state_after, device=device)
        backward_mask = torch.zeros(2 * env.N, device=device)
        decided = state_after != -1
        for j in np.where(decided)[0]:
            if state_after[j] == 1:
                backward_mask[j] = 1.0  # undo select
            else:
                backward_mask[env.N + j] = 1.0  # undo skip
        logits_b = torch.where(backward_mask > 0, logits_b, inf_tensor)
        logprob_b = logprob_b + F.log_softmax(logits_b, dim=0)[action]
    
    return logprob_f, logprob_b


def train(env, forward_policy, backward_policy, loss_fn, optimizer,
          steps=2000, device="cpu", save_dir=None, problem_name=None,
          batch_size=16, epsilon_start=0.5, log_dir=None):
    """Training loop for knapsack GFN with batch training."""
    import json
    import time
    from datetime import datetime
    
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
        "num_items": env.N,
        "capacity": env.capacity,
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
    best_state = None
    
    for step in range(steps):
        epsilon = max(0.01, epsilon_start * (1 - step / steps))
        
        # Sample batch of trajectories
        batch_losses = []
        batch_rewards = []
        batch_profits = []
        
        for _ in range(batch_size):
            traj_states, traj_actions, reward = sample_trajectory(
                env, forward_policy, device, epsilon=epsilon
            )
            
            final_state = traj_states[-1]
            profit = env.get_profit(final_state)
            weight = env.get_weight(final_state)
            
            batch_rewards.append(reward)
            batch_profits.append(profit)
            
            if profit > best_profit and weight <= env.capacity:
                best_profit = profit
                best_state = final_state.copy()
            
            # Compute log probabilities
            logprob_f, logprob_b = compute_logprobs(
                traj_states, traj_actions, forward_policy, backward_policy, env, device
            )
            
            # TB loss
            logreward = torch.log(torch.tensor(reward + 1e-8, device=device))
            traj_loss = loss_fn(logprob_f, logprob_b, logreward)
            batch_losses.append(traj_loss)
        
        # Cumulative loss: mean over batch
        cumulative_loss = torch.stack(batch_losses).mean()
        reward_history.extend(batch_rewards)
        profit_history.extend(batch_profits)
        loss_history.append(cumulative_loss.item())
        
        optimizer.zero_grad()
        cumulative_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(forward_policy.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(backward_policy.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_([loss_fn.logZ], max_norm=0.5)
        optimizer.step()
        
        if step % 50 == 0:
            avg_reward = np.mean(reward_history[-50*batch_size:]) if len(reward_history) >= 50*batch_size else np.mean(reward_history)
            avg_profit = np.mean(profit_history[-50*batch_size:]) if len(profit_history) >= 50*batch_size else np.mean(profit_history)
            avg_loss = np.mean(loss_history[-50:]) if len(loss_history) >= 50 else np.mean(loss_history)
            batch_avg_profit = np.mean(batch_profits)
            elapsed = time.time() - start_time
            
            print(f"[{step}] loss={cumulative_loss.item():.4f} (avg={avg_loss:.4f}), "
                  f"batch_profit={batch_avg_profit:.1f}, avg_profit={avg_profit:.1f}, best={best_profit:.0f}, "
                  f"eps={epsilon:.3f}", flush=True)
            
            # Log to file
            log_entry = {
                "type": "step",
                "step": step,
                "loss": float(cumulative_loss.item()),
                "avg_loss": float(avg_loss),
                "batch_profit": float(batch_avg_profit),
                "avg_profit": float(avg_profit),
                "best_profit": float(best_profit),
                "avg_reward": float(avg_reward),
                "epsilon": float(epsilon),
                "logZ": float(loss_fn.logZ.item()),
                "elapsed_seconds": float(elapsed),
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        
        if (step + 1) % 500 == 0:
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
    
    return best_state, best_profit


def main():
    parser = argparse.ArgumentParser(description="Train GFlowNet for Knapsack Problem")
    parser.add_argument("--problem", type=str, default=None,
                        help="Problem name (e.g., p01, p02). Use --list to see available problems.")
    parser.add_argument("--list", action="store_true",
                        help="List all available problems and exit")
    parser.add_argument("--steps", type=int, default=5000,
                        help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Number of trajectories per training step")
    parser.add_argument("--epsilon", type=float, default=0.5,
                        help="Initial epsilon for exploration (decays to 0.01)")
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Hidden dimension for policy networks")
    args = parser.parse_args()
    
    # List available problems
    problems = list_knapsack_instances(DATA_DIR)
    
    if args.list or args.problem is None:
        print("Available Knapsack Problems:")
        print("-" * 60)
        print(f"{'Problem':<10} {'Items':<8} {'Capacity':<12} {'Optimal':<10}")
        print("-" * 60)
        for p in problems:
            info = get_instance_info(DATA_DIR, p)
            opt = f"{info['optimal_profit']:.0f}" if info['optimal_profit'] else "?"
            print(f"{p:<10} {info['items']:<8} {info['capacity']:<12} {opt:<10}")
        print("-" * 60)
        if args.problem is None:
            print("\nUse --problem <name> to select a problem (e.g., --problem p01)")
            return
        print()
    
    if args.problem not in problems:
        print(f"Error: Problem '{args.problem}' not found.")
        print(f"Available: {problems}")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Load instance
    instance = load_knapsack_instance(DATA_DIR, args.problem)
    N = len(instance['profits'])
    
    print(f"Knapsack instance: {args.problem}")
    print(f"  Items: {N}")
    print(f"  Capacity: {instance['capacity']}")
    print(f"  Total profit (all items): {np.sum(instance['profits']):.0f}")
    print(f"  Total weight (all items): {np.sum(instance['weights']):.0f}")
    
    if 'optimal_profit' in instance:
        print(f"  Optimal profit: {instance['optimal_profit']:.0f}")
        print(f"  Optimal solution: {instance['optimal_solution']}")
    print()
    
    # Create environment
    env = KnapsackEnv(instance)
    
    # Create policies
    forward = KnapsackPolicy(num_items=N, hidden_dim=args.hidden_dim).to(device)
    backward = KnapsackPolicy(num_items=N, hidden_dim=args.hidden_dim).to(device)
    
    # Set instance data for feature computation
    forward.set_instance(instance['profits'], instance['weights'], instance['capacity'])
    backward.set_instance(instance['profits'], instance['weights'], instance['capacity'])
    
    # Loss and optimizer
    loss_fn = TrajectoryBalance(forward, backward).to(device)
    
    optimizer = torch.optim.Adam(
        list(forward.parameters()) +
        list(backward.parameters()) +
        list(loss_fn.parameters()),
        lr=1e-3
    )
    
    print(f"Training Knapsack with GFlowNet + TB loss...\n")
    best_state, best_profit = train(
        env=env,
        forward_policy=forward,
        backward_policy=backward,
        loss_fn=loss_fn,
        optimizer=optimizer,
        steps=args.steps,
        device=device,
        problem_name=args.problem,
        batch_size=args.batch_size,
        epsilon_start=args.epsilon
    )
    
    print(f"\n{'='*50}")
    print(f"Training complete!")
    print(f"Best solution found:")
    print(f"  Profit: {best_profit:.0f}")
    print(f"  Weight: {env.get_weight(best_state):.0f}/{env.capacity}")
    print(f"  Items selected: {np.where(best_state == 1)[0].tolist()}")
    print(f"  Selection: {best_state}")
    
    if 'optimal_profit' in instance:
        gap = (instance['optimal_profit'] - best_profit) / instance['optimal_profit'] * 100
        print(f"\n  Optimal profit: {instance['optimal_profit']:.0f}")
        print(f"  Gap to optimal: {gap:.2f}%")


if __name__ == "__main__":
    main()
