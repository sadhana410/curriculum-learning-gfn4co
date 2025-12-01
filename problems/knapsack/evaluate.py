# problems/knapsack/evaluate.py

import argparse
import os
import sys
import json
from datetime import datetime
import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from problems.knapsack.env import KnapsackEnv, ConditionalKnapsackEnv
from problems.knapsack.policy import KnapsackPolicy, ConditionalKnapsackPolicy, ConditionalKnapsackPolicyWrapper
from problems.knapsack.utils import load_knapsack_instance, list_knapsack_instances, get_instance_info

# Data and checkpoint directories relative to this file
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")


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
            
            # Output best solution
            print()
            print("  " + "=" * 50)
            print("  BEST SOLUTION")
            print("  " + "=" * 50)
            best_state = results['stochastic_state']
            best_profit = results['stochastic_profit']
            best_weight = results['stochastic_weight']
            
            is_valid = best_weight <= results['capacity']
            is_optimal = is_valid and 'optimal_profit' in results and best_profit >= results['optimal_profit']
            
            status = "OPTIMAL" if is_optimal else ("VALID" if is_valid else "INVALID")
            opt_str = f"/{results['optimal_profit']:.0f}" if 'optimal_profit' in results else ""
            gap_str = f" (gap: {results['stochastic_gap']:.2f}%)" if 'stochastic_gap' in results else ""
            
            print(f"  Status: {status}")
            print(f"  Profit: {best_profit:.0f}{opt_str}{gap_str}")
            print(f"  Weight: {best_weight:.0f}/{results['capacity']}")
            if best_state is not None:
                items_selected = np.where(best_state == 1)[0].tolist()
                print(f"  Items selected: {items_selected}")
                print(f"  Selection vector: {best_state.tolist()}")
                
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
    
    # Save evaluation log
    if all_results:
        os.makedirs(LOG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if len(all_results) == 1:
            log_name = f"eval_{all_results[0]['problem']}_{timestamp}.json"
        else:
            log_name = f"eval_all_{timestamp}.json"
        log_path = os.path.join(LOG_DIR, log_name)
        
        log_data = {
            "type": "evaluation",
            "timestamp": timestamp,
            "num_samples": args.samples,
            "results": []
        }
        
        for r in all_results:
            entry = {
                "problem": r['problem'],
                "items": r['items'],
                "capacity": r['capacity'],
                "training_step": r['training_step'],
                "best_training_profit": float(r['best_training_profit']) if isinstance(r['best_training_profit'], (int, float)) else r['best_training_profit'],
                "greedy": {
                    "profit": float(r['greedy_profit']),
                    "weight": float(r['greedy_weight']),
                    "items_selected": np.where(r['greedy_state'] == 1)[0].tolist(),
                },
                "stochastic": {
                    "profit": float(r['stochastic_profit']),
                    "weight": float(r['stochastic_weight']),
                    "items_selected": np.where(r['stochastic_state'] == 1)[0].tolist() if r['stochastic_state'] is not None else [],
                },
            }
            if 'optimal_profit' in r:
                entry['optimal_profit'] = float(r['optimal_profit'])
                entry['optimal_solution'] = r['optimal_solution'].tolist() if hasattr(r['optimal_solution'], 'tolist') else r['optimal_solution']
                entry['greedy']['gap_percent'] = float(r['greedy_gap'])
                entry['stochastic']['gap_percent'] = float(r['stochastic_gap'])
            
            log_data['results'].append(entry)
        
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)
        
        print(f"\nEvaluation log saved to: {log_path}")


# ============================================================================
# Conditional Evaluation Functions
# ============================================================================

def create_conditional_policy_from_checkpoint(checkpoint, device):
    """
    Create a ConditionalKnapsackPolicy from checkpoint, handling both old and new formats.
    
    Returns:
        shared_policy: The underlying ConditionalKnapsackPolicy
        hidden_dim: Hidden dimension
        num_layers: Number of attention layers
    """
    forward_state = checkpoint['forward_policy']
    
    # Check if keys have 'shared_policy.' prefix (new format) or not (old format)
    sample_key = list(forward_state.keys())[0]
    has_prefix = sample_key.startswith('shared_policy.')
    
    # Find hidden dim from item_encoder
    if has_prefix:
        encoder_key = 'shared_policy.item_encoder.0.weight'
    else:
        encoder_key = 'item_encoder.0.weight'
    
    hidden_dim = forward_state[encoder_key].shape[0]
    
    # Count attention layers
    if has_prefix:
        attn_prefix = 'shared_policy.attention_layers.'
    else:
        attn_prefix = 'attention_layers.'
    
    num_layers = 0
    for key in forward_state.keys():
        if attn_prefix in key:
            after_prefix = key.split(attn_prefix)[1]
            layer_idx = int(after_prefix.split('.')[0])
            num_layers = max(num_layers, layer_idx + 1)
    
    # Create policy
    shared_policy = ConditionalKnapsackPolicy(
        hidden_dim=hidden_dim,
        num_layers=num_layers
    ).to(device)
    
    if has_prefix:
        # New format: load via wrapper
        forward = ConditionalKnapsackPolicyWrapper(shared_policy, mode='forward')
        forward.load_state_dict(forward_state)
        forward.eval()
        return shared_policy, hidden_dim, num_layers
    else:
        # Old format: load directly
        shared_policy.load_state_dict(forward_state)
        shared_policy.eval()
        return shared_policy, hidden_dim, num_layers


def sample_greedy_conditional(env, policy, instance, device):
    """Sample a solution using greedy (argmax) policy for conditional model."""
    state = env.reset()
    profits = instance['profits']
    weights = instance['weights']
    capacity = instance['capacity']
    
    with torch.no_grad():
        while True:
            logits = policy(state, profits, weights, capacity, device=device)
            mask = torch.tensor(env.allowed_actions(state), dtype=torch.float32, device=device)
            
            if mask.sum() == 0:
                break
            
            masked_logits = logits.clone()
            masked_logits[mask == 0] = float('-inf')
            action = masked_logits.argmax().item()
            
            state, reward, done = env.step(state, action)
            if done:
                break
    
    return state, reward


def sample_stochastic_conditional(env, policy, instance, device, num_samples=100):
    """Sample multiple solutions stochastically for conditional model."""
    best_state = None
    best_profit = -float('inf')
    
    profits = instance['profits']
    weights = instance['weights']
    capacity = instance['capacity']
    
    with torch.no_grad():
        for _ in range(num_samples):
            state = env.reset()
            
            while True:
                logits = policy(state, profits, weights, capacity, device=device)
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
            
            if weight <= capacity and profit > best_profit:
                best_profit = profit
                best_state = state.copy()
    
    return best_state, best_profit


def evaluate_conditional(args):
    """Evaluate a conditional GFlowNet model on knapsack problems."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    problems = list_knapsack_instances(DATA_DIR)
    
    # Load checkpoint
    if not args.checkpoint or not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Create policy from checkpoint
    shared_policy, hidden_dim, num_layers = create_conditional_policy_from_checkpoint(checkpoint, device)
    
    print(f"Loaded conditional model:")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Attention layers: {num_layers}")
    print(f"  Training step: {checkpoint.get('step', '?')}")
    print()
    
    # Determine which problems to evaluate
    if args.all:
        eval_problems = problems
    elif args.problems:
        eval_problems = args.problems
        for name in eval_problems:
            if name not in problems:
                print(f"Error: Problem '{name}' not found.")
                return
    else:
        print("Available Knapsack Problems:")
        print("-" * 60)
        print(f"{'Problem':<10} {'Items':<8} {'Capacity':<12} {'Optimal':<10}")
        print("-" * 60)
        for p in problems:
            info = get_instance_info(DATA_DIR, p)
            opt = f"{info['optimal_profit']:.0f}" if info['optimal_profit'] else "?"
            print(f"{p:<10} {info['items']:<8} {info['capacity']:<12} {opt:<10}")
        print("-" * 60)
        print("\nUse --problems <name1> <name2> ... to evaluate specific problems")
        print("Use --all to evaluate all problems")
        return
    
    # Evaluate each problem
    print("=" * 80)
    print("CONDITIONAL GFLOWNET EVALUATION")
    print("=" * 80)
    
    all_results = []
    
    for problem_name in eval_problems:
        instance = load_knapsack_instance(DATA_DIR, problem_name)
        N = len(instance['profits'])
        
        # Create environment for this instance
        env = KnapsackEnv(instance)
        
        print(f"\n[{problem_name}] {N} items, capacity {instance['capacity']}")
        
        # Greedy evaluation
        greedy_state, greedy_reward = sample_greedy_conditional(env, shared_policy, instance, device)
        greedy_profit = env.get_profit(greedy_state)
        greedy_weight = env.get_weight(greedy_state)
        
        # Stochastic evaluation
        stoch_state, stoch_profit = sample_stochastic_conditional(
            env, shared_policy, instance, device, args.samples
        )
        stoch_weight = env.get_weight(stoch_state) if stoch_state is not None else 0
        
        # Results
        results = {
            'problem': problem_name,
            'items': N,
            'capacity': instance['capacity'],
            'greedy_profit': float(greedy_profit),
            'greedy_weight': float(greedy_weight),
            'stochastic_profit': float(stoch_profit) if stoch_profit else 0,
            'stochastic_weight': float(stoch_weight),
            'greedy_state': greedy_state,
            'stochastic_state': stoch_state,
        }
        
        if 'optimal_profit' in instance:
            results['optimal_profit'] = float(instance['optimal_profit'])
            results['greedy_gap'] = float((instance['optimal_profit'] - greedy_profit) / instance['optimal_profit'] * 100)
            results['stochastic_gap'] = float((instance['optimal_profit'] - stoch_profit) / instance['optimal_profit'] * 100) if stoch_profit else 100.0
        
        all_results.append(results)
        
        # Print results
        greedy_valid = greedy_weight <= instance['capacity']
        stoch_valid = stoch_weight <= instance['capacity'] if stoch_state is not None else False
        
        greedy_mark = "✓" if greedy_valid else "✗"
        stoch_mark = "✓" if stoch_valid else "✗"
        
        opt_str = f"/{instance['optimal_profit']:.0f}" if 'optimal_profit' in instance else ""
        greedy_gap_str = f" (gap: {results['greedy_gap']:.1f}%)" if 'greedy_gap' in results else ""
        stoch_gap_str = f" (gap: {results['stochastic_gap']:.1f}%)" if 'stochastic_gap' in results else ""
        
        print(f"  Greedy:     {greedy_profit:.0f}{opt_str} profit, {greedy_weight:.0f}/{instance['capacity']} weight {greedy_mark}{greedy_gap_str}")
        print(f"  Stochastic: {stoch_profit:.0f}{opt_str} profit, {stoch_weight:.0f}/{instance['capacity']} weight {stoch_mark}{stoch_gap_str}")
    
    # Summary
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
        
        # Average gaps
        if any('greedy_gap' in r for r in all_results):
            avg_greedy_gap = np.mean([r['greedy_gap'] for r in all_results if 'greedy_gap' in r])
            avg_stoch_gap = np.mean([r['stochastic_gap'] for r in all_results if 'stochastic_gap' in r])
            print("-" * 60)
            print(f"{'Average':<10} {'':<10} {'':<10} {avg_greedy_gap:<8.1f} {'':<10} {avg_stoch_gap:<8.1f}")
    
    # Print best solutions
    print("\n" + "=" * 80)
    print("BEST SOLUTIONS (Stochastic)")
    print("=" * 80)
    for r in all_results:
        print(f"\n{r['problem']}:")
        stoch_state = r['stochastic_state']
        stoch_profit = r['stochastic_profit']
        stoch_weight = r['stochastic_weight']
        capacity = r['capacity']
        
        is_valid = stoch_weight <= capacity
        is_optimal = is_valid and 'optimal_profit' in r and stoch_profit >= r['optimal_profit']
        
        status = "OPTIMAL" if is_optimal else ("VALID" if is_valid else "INVALID")
        opt_str = f"/{r['optimal_profit']:.0f}" if 'optimal_profit' in r else ""
        gap_str = f" (gap: {r['stochastic_gap']:.1f}%)" if 'stochastic_gap' in r else ""
        
        print(f"  Status: {status}")
        print(f"  Profit: {stoch_profit:.0f}{opt_str}{gap_str}")
        print(f"  Weight: {stoch_weight:.0f}/{capacity}")
        if stoch_state is not None:
            items_selected = np.where(stoch_state == 1)[0].tolist()
            print(f"  Items selected: {items_selected}")
            print(f"  Selection vector: {stoch_state.tolist()}")
    
    # Save evaluation log
    if all_results:
        os.makedirs(LOG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_name = f"eval_conditional_{timestamp}.json"
        log_path = os.path.join(LOG_DIR, log_name)
        
        log_data = {
            "type": "conditional_evaluation",
            "timestamp": timestamp,
            "checkpoint": args.checkpoint,
            "num_samples": args.samples,
            "results": []
        }
        
        for r in all_results:
            entry = {
                "problem": r['problem'],
                "items": r['items'],
                "capacity": r['capacity'],
                "greedy": {
                    "profit": r['greedy_profit'],
                    "weight": r['greedy_weight'],
                    "items_selected": np.where(r['greedy_state'] == 1)[0].tolist(),
                },
                "stochastic": {
                    "profit": r['stochastic_profit'],
                    "weight": r['stochastic_weight'],
                    "items_selected": np.where(r['stochastic_state'] == 1)[0].tolist() if r['stochastic_state'] is not None else [],
                },
            }
            if 'optimal_profit' in r:
                entry['optimal_profit'] = r['optimal_profit']
                entry['greedy']['gap_percent'] = r['greedy_gap']
                entry['stochastic']['gap_percent'] = r['stochastic_gap']
            
            log_data['results'].append(entry)
        
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)
        
        print(f"\nEvaluation log saved to: {log_path}")


def main_wrapper():
    """Entry point that dispatches to single or conditional evaluation."""
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
    
    # Conditional evaluation arguments
    parser.add_argument("--conditional", action="store_true",
                        help="Evaluate conditional GFlowNet model")
    parser.add_argument("--problems", type=str, nargs="+", default=None,
                        help="[Conditional] List of problem names to evaluate")
    
    args = parser.parse_args()
    
    if args.conditional:
        evaluate_conditional(args)
    else:
        main()


if __name__ == "__main__":
    main_wrapper()
