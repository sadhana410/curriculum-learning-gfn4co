# problems/tsp/main.py

import argparse
import os
import sys
import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from problems.tsp.env import TSPEnv, ConditionalTSPEnv, TSPInstanceDataset
from problems.tsp.policy import TSPPolicy, ConditionalTSPPolicy, ConditionalTSPPolicyWrapper
from problems.tsp.utils import load_tsp_file, list_tsp_instances, get_instance_info, generate_random_tsp
from problems.tsp.optimal_solver import generate_and_save_tsp, solve_tsp_optimal
from problems.tsp.trainer import train, train_conditional
from losses.trajectorybalance import TrajectoryBalance
from losses.detailedbalance import DetailedBalance
from losses.subtrajectorybalance import SubTrajectoryBalance


# Data directory relative to this file
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")


def main():
    parser = argparse.ArgumentParser(description="Train GFlowNet for Traveling Salesman Problem")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for optimizer")
    parser.add_argument("--problem", type=str, default=None,
                        help="Problem name (e.g., p_all). Use --list to see available problems.")
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
                        help="Save checkpoint every N steps")
    
    # Random instance generation
    parser.add_argument("--random", action="store_true",
                        help="Generate a random TSP instance instead of loading from file")
    parser.add_argument("--num-cities", type=int, default=20,
                        help="Number of cities for random instance")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for instance generation")
    parser.add_argument("--save-instance", action="store_true",
                        help="Save generated random instance to data directory")
    parser.add_argument("--solve-optimal", action="store_true",
                        help="Compute optimal solution for random instance (N<=20)")
    
    # Conditional training arguments
    parser.add_argument("--conditional", action="store_true",
                        help="Use conditional GFlowNet (train on multiple instances)")
    parser.add_argument("--num-instances", type=int, default=10,
                        help="[Conditional] Number of random instances to generate")
    parser.add_argument("--min-cities", type=int, default=10,
                        help="[Conditional] Minimum cities per instance")
    parser.add_argument("--max-cities", type=int, default=30,
                        help="[Conditional] Maximum cities per instance")
    parser.add_argument("--num-layers", type=int, default=3,
                        help="[Conditional] Number of attention layers in policy")
    parser.add_argument("--resume", type=str, default=None,
                        help="[Conditional] Resume from checkpoint (path or 'latest')")
    
    args = parser.parse_args()
    
    # Loss configuration
    LOSS_CONFIG = {
        "SubTB": {"lambda_": 1.0},
        "DB": {},
        "TB": {}
    }
    
    # List available problems
    problems = list_tsp_instances(DATA_DIR)
    
    if args.list:
        print("Available TSP Problems:")
        print("-" * 60)
        print(f"{'Problem':<20} {'Cities':<10} {'Optimal':<15}")
        print("-" * 60)
        for p in problems:
            info = get_instance_info(DATA_DIR, p)
            if info:
                opt = f"{info['optimal_length']:.1f}" if info.get('optimal_length') else "?"
                print(f"{p:<20} {info['nodes']:<10} {opt:<15}")
        print("-" * 60)
        print("\nUse --problem <name> to select a problem")
        print("Use --random --num-cities <N> to generate a random instance")
        return
    
    # Handle conditional training
    if args.conditional:
        main_conditional(args)
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Load or generate instance
    if args.random:
        print(f"Generating random TSP instance with {args.num_cities} cities...")
        
        # Determine if we should solve for optimal
        solve_opt = args.solve_optimal or (args.num_cities <= 15)
        
        if args.save_instance or solve_opt:
            # Use optimal_solver to generate, solve, and optionally save
            instance = generate_and_save_tsp(
                num_cities=args.num_cities,
                seed=args.seed,
                solve=solve_opt,
                name=None  # Auto-generate name
            ) if args.save_instance else None
            
            if not args.save_instance:
                # Just generate and solve without saving
                instance = generate_random_tsp(args.num_cities, seed=args.seed)
                if solve_opt:
                    print(f"Computing optimal solution...")
                    tour, length = solve_tsp_optimal(instance['distance_matrix'], verbose=True)
                    instance['optimal_tour'] = tour
                    instance['optimal_length'] = length
        else:
            instance = generate_random_tsp(args.num_cities, seed=args.seed)
        
        problem_name = f"random_{args.num_cities}"
        if args.seed:
            problem_name += f"_seed{args.seed}"
    elif args.problem:
        if args.problem not in problems:
            # Try loading directly
            filepath = os.path.join(DATA_DIR, f"{args.problem}.tsp")
            if not os.path.exists(filepath):
                filepath = os.path.join(DATA_DIR, args.problem)
            if not os.path.exists(filepath):
                print(f"Error: Problem '{args.problem}' not found.")
                print(f"Available: {problems}")
                return
            instance = load_tsp_file(filepath)
        else:
            filepath = os.path.join(DATA_DIR, f"{args.problem}.tsp")
            instance = load_tsp_file(filepath)
        problem_name = args.problem
    else:
        print("Error: Please specify --problem <name> or --random")
        print("Use --list to see available problems")
        return
    
    N = instance['N']
    
    print(f"TSP instance: {instance['name']}")
    print(f"  Cities: {N}")
    if 'optimal_length' in instance:
        print(f"  Optimal tour length: {instance['optimal_length']:.4f}")
    if 'optimal_tour' in instance:
        # Show tour with return to start
        tour_with_return = [int(x) for x in instance['optimal_tour'][:10]] + ([int(instance['optimal_tour'][0])] if N <= 10 else [])
        tour_str = str(tour_with_return)
        if N > 10:
            tour_str = tour_str[:-1] + ", ... -> 0]"
        print(f"  Optimal tour: {tour_str}")
    print()
    
    # Create environment
    env = TSPEnv(instance)
    
    # Create policies
    forward = TSPPolicy(num_cities=N, hidden_dim=args.hidden_dim).to(device)
    backward = TSPPolicy(num_cities=N, hidden_dim=args.hidden_dim).to(device)
    
    # Set instance data for feature computation
    forward.set_instance(instance['coords'], instance['distance_matrix'])
    backward.set_instance(instance['coords'], instance['distance_matrix'])
    
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
    
    # Get optimal length for early stopping
    optimal_length = instance.get('optimal_length', None)
    
    print(f"Training TSP with GFlowNet + {args.loss} loss...\n")
    best_state, best_length = train(
        env=env,
        forward_policy=forward,
        backward_policy=backward,
        loss_fn=loss_fn,
        optimizer=optimizer,
        steps=args.steps,
        device=device,
        problem_name=problem_name,
        batch_size=args.batch_size,
        epsilon_start=args.epsilon,
        save_every=args.save_every,
        optimal_length=optimal_length,
        top_p=args.top_p,
        temperature=args.temperature,
        early_stop_patience=args.patience
    )
    
    print(f"\n{'='*50}")
    print(f"Training complete!")
    print(f"Best solution found:")
    print(f"  Tour length: {best_length:.2f}")
    
    if best_state is not None:
        tour = env.get_tour_from_state(best_state)
        # Show tour with return to start
        tour_with_return = [int(x) for x in tour] + [int(tour[0])]
        print(f"  Tour: {tour_with_return}")
    
    if optimal_length:
        gap = (best_length - optimal_length) / optimal_length * 100
        print(f"\n  Optimal length: {optimal_length:.2f}")
        print(f"  Gap to optimal: {gap:.2f}%")


def main_conditional(args):
    """Conditional GFlowNet training on multiple TSP instances."""
    
    # Loss configuration
    LOSS_CONFIG = {
        "SubTB": {"lambda_": 1.0},
        "DB": {},
        "TB": {}
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Generate random instances
    print(f"Generating {args.num_instances} random TSP instances...")
    print(f"  Cities range: {args.min_cities} - {args.max_cities}")
    
    dataset = TSPInstanceDataset.generate_random(
        num_instances=args.num_instances,
        min_nodes=args.min_cities,
        max_nodes=args.max_cities,
        seed=args.seed
    )
    
    print(f"Generated {len(dataset)} instances:")
    for i, inst in enumerate(dataset.instances[:5]):
        print(f"  {inst['name']}: {inst['N']} cities")
    if len(dataset) > 5:
        print(f"  ... and {len(dataset) - 5} more")
    print()
    
    # Create conditional environment
    env = ConditionalTSPEnv(dataset)
    
    # Create conditional policy with shared backbone
    shared_policy = ConditionalTSPPolicy(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    ).to(device)
    
    # Create wrappers for forward and backward
    forward = ConditionalTSPPolicyWrapper(shared_policy, mode='forward')
    backward = ConditionalTSPPolicyWrapper(shared_policy, mode='backward')
    
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
        list(shared_policy.parameters()) +
        list(loss_fn.parameters()),
        lr=args.lr
    )
    
    problem_name = f"conditional_{args.num_instances}inst_{args.min_cities}-{args.max_cities}cities"
    
    print(f"Training Conditional TSP GFlowNet...\n")
    best_per_instance = train_conditional(
        env=env,
        forward_policy=forward,
        backward_policy=backward,
        loss_fn=loss_fn,
        optimizer=optimizer,
        steps=args.steps,
        device=device,
        save_dir=CHECKPOINT_DIR,
        log_dir=LOG_DIR,
        problem_name=problem_name,
        batch_size=args.batch_size,
        epsilon_start=args.epsilon,
        save_every=args.save_every,
        early_stop_patience=args.patience,
        top_p=args.top_p,
        temperature=args.temperature,
        resume_from=args.resume
    )
    
    print(f"\n{'='*50}")
    print(f"Conditional Training complete!")


def main_wrapper():
    """Entry point that dispatches to single or conditional training."""
    parser = argparse.ArgumentParser(description="Train GFlowNet for Traveling Salesman Problem")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for optimizer")
    parser.add_argument("--problem", type=str, default=None,
                        help="Problem name. Use --list to see available problems.")
    parser.add_argument("--list", action="store_true",
                        help="List all available problems and exit")
    parser.add_argument("--steps", type=int, default=5000,
                        help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Number of trajectories per training step")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Initial epsilon for exploration")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Top-P sampling threshold")
    parser.add_argument("--patience", type=int, default=5000,
                        help="Early stopping patience")
    parser.add_argument("--loss", type=str, default="TB", choices=["TB", "DB", "SubTB"],
                        help="Loss function")
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Hidden dimension for policy networks")
    parser.add_argument("--save-every", type=int, default=1000,
                        help="Save checkpoint every N steps")
    
    # Random instance
    parser.add_argument("--random", action="store_true",
                        help="Generate a random TSP instance")
    parser.add_argument("--num-cities", type=int, default=20,
                        help="Number of cities for random instance")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--save-instance", action="store_true",
                        help="Save generated random instance to data directory")
    parser.add_argument("--solve-optimal", action="store_true",
                        help="Compute optimal solution for random instance")
    
    # Conditional training
    parser.add_argument("--conditional", action="store_true",
                        help="Use conditional GFlowNet")
    parser.add_argument("--num-instances", type=int, default=10,
                        help="[Conditional] Number of instances")
    parser.add_argument("--min-cities", type=int, default=10,
                        help="[Conditional] Min cities per instance")
    parser.add_argument("--max-cities", type=int, default=30,
                        help="[Conditional] Max cities per instance")
    parser.add_argument("--num-layers", type=int, default=3,
                        help="[Conditional] Number of attention layers")
    parser.add_argument("--resume", type=str, default=None,
                        help="[Conditional] Resume from checkpoint")
    
    args = parser.parse_args()
    
    if args.conditional:
        main_conditional(args)
    else:
        main()


if __name__ == "__main__":
    main_wrapper()
