# problems/knapsack/evaluate.py

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

# Data and checkpoint directories relative to this file
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def load_checkpoint(checkpoint_path, num_items, hidden_dim, device):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    forward = KnapsackPolicy(num_items=num_items, hidden_dim=hidden_dim).to(device)
    forward.load_state_dict(checkpoint['forward_policy'])
    forward.eval()
    
    return forward, checkpoint


def sample_greedy(env, policy, device):
    """Sample a solution using greedy (argmax) policy."""
    state = env.reset()
    
    with torch.no_grad():
        while True:
            logits = policy(state, device=device)
            mask = torch.tensor(env.allowed_actions(state), dtype=torch.float32, device=device)
            
            if mask.sum() == 0:
                break
            
            # Greedy: pick best action
            masked_logits = logits.clone()
            masked_logits[mask == 0] = float('-inf')
            action = masked_logits.argmax().item()
            
            state, reward, done = env.step(state, action)
            if done:
                break
    
    return state, reward


def sample_stochastic(env, policy, device, num_samples=100):
    """Sample multiple solutions stochastically and return the best."""
    best_state = None
    best_profit = -float('inf')
    
    with torch.no_grad():
        for _ in range(num_samples):
            state = env.reset()
            
            while True:
                logits = policy(state, device=device)
                mask = torch.tensor(env.allowed_actions(state), dtype=torch.float32, device=device)
                
                if mask.sum() == 0:
                    break
                
                masked_logits = logits.clone()
                masked_logits[mask == 0] = float('-inf')
                probs = torch.softmax(masked_logits, dim=0)
                action = torch.multinomial(probs, 1).item()
                
                state, reward, done = env.step(state, action)
                if done:
                    break
            
            profit = env.get_profit(state)
            weight = env.get_weight(state)
            
            if weight <= env.capacity and profit > best_profit:
                best_profit = profit
                best_state = state.copy()
    
    return best_state, best_profit


def evaluate_problem(problem_name, checkpoint_path, hidden_dim, device, num_samples=100):
    """Evaluate a trained model on a specific problem."""
    # Load instance
    instance = load_knapsack_instance(DATA_DIR, problem_name)
    N = len(instance['profits'])
    
    # Create environment
    env = KnapsackEnv(instance)
    
    # Load model
    forward, checkpoint = load_checkpoint(checkpoint_path, N, hidden_dim, device)
    forward.set_instance(instance['profits'], instance['weights'], instance['capacity'])
    
    # Greedy evaluation
    greedy_state, greedy_reward = sample_greedy(env, forward, device)
    greedy_profit = env.get_profit(greedy_state)
    greedy_weight = env.get_weight(greedy_state)
    
    # Stochastic evaluation (best of N samples)
    stoch_state, stoch_profit = sample_stochastic(env, forward, device, num_samples)
    stoch_weight = env.get_weight(stoch_state) if stoch_state is not None else 0
    
    # Results
    results = {
        'problem': problem_name,
        'items': N,
        'capacity': instance['capacity'],
        'greedy_profit': greedy_profit,
        'greedy_weight': greedy_weight,
        'stochastic_profit': stoch_profit,
        'stochastic_weight': stoch_weight,
        'greedy_state': greedy_state,
        'stochastic_state': stoch_state,
        'training_step': checkpoint.get('step', '?'),
        'best_training_profit': checkpoint.get('best_profit', '?'),
    }
    
    if 'optimal_profit' in instance:
        results['optimal_profit'] = instance['optimal_profit']
        results['optimal_solution'] = instance['optimal_solution']
        results['greedy_gap'] = (instance['optimal_profit'] - greedy_profit) / instance['optimal_profit'] * 100
        results['stochastic_gap'] = (instance['optimal_profit'] - stoch_profit) / instance['optimal_profit'] * 100
    
    return results


def find_checkpoint(problem_name, checkpoint_dir):
    """Find the latest checkpoint for a specific problem."""
    checkpoints = []
    
    if os.path.exists(checkpoint_dir):
        for f in os.listdir(checkpoint_dir):
            if f.endswith('.pt'):
                # Prefer checkpoints matching the problem name
                if f.startswith(f'{problem_name}_'):
                    checkpoints.append(os.path.join(checkpoint_dir, f))
    
    # If no problem-specific checkpoints, fall back to generic ones
    if not checkpoints and os.path.exists(checkpoint_dir):
        for f in os.listdir(checkpoint_dir):
            if f.endswith('.pt') and f.startswith('checkpoint_'):
                checkpoints.append(os.path.join(checkpoint_dir, f))
    
    if not checkpoints:
        return None
    
    # Sort by step number (highest first)
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained GFlowNet on Knapsack problems")
    parser.add_argument("--problem", type=str, default=None,
                        help="Problem name to evaluate. If not specified, lists available problems.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file. If not specified, uses latest in checkpoints/")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of stochastic samples for evaluation")
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Hidden dimension (must match training)")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all problems (requires matching checkpoints)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    problems = list_knapsack_instances(DATA_DIR)
    
    if args.problem is None and not args.all:
        print("Available Knapsack Problems:")
        print("-" * 60)
        print(f"{'Problem':<10} {'Items':<8} {'Capacity':<12} {'Optimal':<10}")
        print("-" * 60)
        for p in problems:
            info = get_instance_info(DATA_DIR, p)
            opt = f"{info['optimal_profit']:.0f}" if info['optimal_profit'] else "?"
            print(f"{p:<10} {info['items']:<8} {info['capacity']:<12} {opt:<10}")
        print("-" * 60)
        print("\nUse --problem <name> to evaluate a specific problem")
        print("Use --all to evaluate all problems")
        return
    
    # Determine which problems to evaluate
    if args.all:
        eval_problems = problems
    else:
        if args.problem not in problems:
            print(f"Error: Problem '{args.problem}' not found.")
            return
        eval_problems = [args.problem]
    
    # Evaluate each problem
    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    all_results = []
    
    for problem in eval_problems:
        # Find checkpoint
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        else:
            checkpoint_path = find_checkpoint(problem, CHECKPOINT_DIR)
        
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            print(f"\n[{problem}] No checkpoint found. Skipping.")
            continue
        
        print(f"\n[{problem}] Loading checkpoint: {os.path.basename(checkpoint_path)}")
        
        try:
            results = evaluate_problem(
                problem, checkpoint_path, args.hidden_dim, device, args.samples
            )
            all_results.append(results)
            
            print(f"  Items: {results['items']}, Capacity: {results['capacity']}")
            print(f"  Training step: {results['training_step']}")
            print(f"  Best during training: {results['best_training_profit']}")
            print()
            print(f"  Greedy solution:")
            print(f"    Profit: {results['greedy_profit']:.0f}, Weight: {results['greedy_weight']:.0f}/{results['capacity']}")
            print(f"    Items: {np.where(results['greedy_state'] == 1)[0].tolist()}")
            print()
            print(f"  Stochastic (best of {args.samples}):")
            print(f"    Profit: {results['stochastic_profit']:.0f}, Weight: {results['stochastic_weight']:.0f}/{results['capacity']}")
            if results['stochastic_state'] is not None:
                print(f"    Items: {np.where(results['stochastic_state'] == 1)[0].tolist()}")
            
            if 'optimal_profit' in results:
                print()
                print(f"  Optimal: {results['optimal_profit']:.0f}")
                print(f"  Greedy gap: {results['greedy_gap']:.2f}%")
                print(f"  Stochastic gap: {results['stochastic_gap']:.2f}%")
                
        except Exception as e:
            print(f"  Error: {e}")
    
    # Summary table
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"{'Problem':<10} {'Optimal':<10} {'Greedy':<10} {'Gap%':<8} {'Stoch':<10} {'Gap%':<8}")
        print("-" * 60)
        for r in all_results:
            opt = f"{r.get('optimal_profit', '?'):.0f}" if 'optimal_profit' in r else "?"
            g_gap = f"{r['greedy_gap']:.1f}" if 'greedy_gap' in r else "?"
            s_gap = f"{r['stochastic_gap']:.1f}" if 'stochastic_gap' in r else "?"
            print(f"{r['problem']:<10} {opt:<10} {r['greedy_profit']:<10.0f} {g_gap:<8} {r['stochastic_profit']:<10.0f} {s_gap:<8}")


if __name__ == "__main__":
    main()
