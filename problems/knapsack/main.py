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
from problems.knapsack.trainer import train
from losses.trajectorybalance import TrajectoryBalance
from losses.detailedbalance import DetailedBalance
from losses.subtrajectorybalance import SubTrajectoryBalance


# Data directory relative to this file
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def main():
    parser = argparse.ArgumentParser(description="Train GFlowNet for Knapsack Problem")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for optimizer")
    parser.add_argument("--problem", type=str, default=None,
                        help="Problem name (e.g., p01, p02). Use --list to see available problems.")
    parser.add_argument("--list", action="store_true",
                        help="List all available problems and exit")
    parser.add_argument("--steps", type=int, default=5000,
                        help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Number of trajectories per training step")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Initial epsilon for exploration (decays to 0.01)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (higher = more exploration)")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Top-P (Nucleus) sampling threshold (0.0-1.0, 1.0 = full softmax)")
    parser.add_argument("--patience", type=int, default=5000,
                        help="Early stopping patience (steps without improvement)")
    parser.add_argument("--loss", type=str, default="TB", choices=["TB", "DB", "SubTB"],
                        help="Loss function: TB, DB, SubTB")
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Hidden dimension for policy networks")
    parser.add_argument("--save-every", type=int, default=1000,
                        help="Save checkpoint every N steps (default: 500)")
    args = parser.parse_args()
    
    # Loss configuration
    LOSS_CONFIG = {
        "SubTB": {"lambda_": 1.0},
        "DB": {},
        "TB": {}
    }
    
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
    
    # Select loss function
    if args.loss == "TB":
        loss_fn = TrajectoryBalance(forward, backward).to(device)
    elif args.loss == "DB":
        loss_fn = DetailedBalance(forward, backward).to(device)
    elif args.loss == "SubTB":
        loss_params = LOSS_CONFIG.get("SubTB", {})
        loss_fn = SubTrajectoryBalance(forward, backward, **loss_params).to(device)
    
    print(f"Using Loss: {args.loss}")
    
    optimizer = torch.optim.Adam(
        list(forward.parameters()) +
        list(backward.parameters()) +
        list(loss_fn.parameters()),
        lr=args.lr
    )
    
    # Get optimal profit for early stopping
    optimal_profit = instance.get('optimal_profit', None)
    
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
        epsilon_start=args.epsilon,
        save_every=args.save_every,
        optimal_profit=optimal_profit,
        top_p=args.top_p,
        temperature=args.temperature,
        early_stop_patience=args.patience
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
