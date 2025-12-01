# problems/tsp/evaluate.py

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

from problems.tsp.env import TSPEnv, ConditionalTSPEnv, TSPInstanceDataset
from problems.tsp.policy import TSPPolicy, ConditionalTSPPolicy, ConditionalTSPPolicyWrapper
from problems.tsp.utils import load_tsp_file, list_tsp_instances, get_instance_info, generate_random_tsp
from problems.tsp.optimal_solver import generate_and_save_tsp, solve_tsp_optimal

# Data and checkpoint directories relative to this file
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")


def load_checkpoint(checkpoint_path, num_cities, hidden_dim, device):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    forward = TSPPolicy(num_cities=num_cities, hidden_dim=hidden_dim).to(device)
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
            
            masked_logits = logits.clone()
            masked_logits[mask == 0] = float('-inf')
            action = masked_logits.argmax().item()
            
            state, reward, done = env.step(state, action)
            if done:
                break
    
    return state, reward


def sample_stochastic(env, policy, device, num_samples=100, temperature=1.0):
    """Sample multiple solutions stochastically and return the best."""
    best_state = None
    best_length = float('inf')
    
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
                probs = torch.softmax(masked_logits / temperature, dim=0)
                action = torch.multinomial(probs, 1).item()
                
                state, reward, done = env.step(state, action)
                if done:
                    break
            
            length = env.get_tour_length(state)
            
            if length < best_length:
                best_length = length
                best_state = state.copy()
    
    return best_state, best_length


def evaluate_problem(problem_name, checkpoint_path, hidden_dim, device, num_samples=100):
    """Evaluate a trained model on a specific problem."""
    # Load instance
    filepath = os.path.join(DATA_DIR, f"{problem_name}.tsp")
    if not os.path.exists(filepath):
        filepath = os.path.join(DATA_DIR, problem_name)
    instance = load_tsp_file(filepath)
    N = instance['N']
    
    # Create environment
    env = TSPEnv(instance)
    
    # Load model
    forward, checkpoint = load_checkpoint(checkpoint_path, N, hidden_dim, device)
    forward.set_instance(instance['coords'], instance['distance_matrix'])
    
    # Greedy evaluation
    greedy_state, greedy_reward = sample_greedy(env, forward, device)
    greedy_length = env.get_tour_length(greedy_state)
    greedy_tour = env.get_tour_from_state(greedy_state)
    
    # Stochastic evaluation (best of N samples)
    stoch_state, stoch_length = sample_stochastic(env, forward, device, num_samples)
    stoch_tour = env.get_tour_from_state(stoch_state) if stoch_state is not None else None
    
    # Results - include return to start in tours (convert to int for clean display)
    greedy_tour_with_return = [int(x) for x in greedy_tour] + [int(greedy_tour[0])]
    stoch_tour_with_return = [int(x) for x in stoch_tour] + [int(stoch_tour[0])] if stoch_tour is not None else None
    
    results = {
        'problem': problem_name,
        'cities': N,
        'greedy_length': greedy_length,
        'greedy_tour': greedy_tour_with_return,
        'stochastic_length': stoch_length,
        'stochastic_tour': stoch_tour_with_return,
        'greedy_state': greedy_state,
        'stochastic_state': stoch_state,
        'training_step': checkpoint.get('step', '?'),
        'best_training_length': checkpoint.get('best_length', '?'),
    }
    
    if 'optimal_length' in instance:
        results['optimal_length'] = instance['optimal_length']
        results['greedy_gap'] = (greedy_length - instance['optimal_length']) / instance['optimal_length'] * 100
        results['stochastic_gap'] = (stoch_length - instance['optimal_length']) / instance['optimal_length'] * 100
    
    return results


def find_checkpoint(problem_name, checkpoint_dir):
    """Find the latest checkpoint for a specific problem."""
    checkpoints = []
    
    if os.path.exists(checkpoint_dir):
        for f in os.listdir(checkpoint_dir):
            if f.endswith('.pt'):
                if f.startswith(f'{problem_name}_'):
                    checkpoints.append(os.path.join(checkpoint_dir, f))
    
    if not checkpoints and os.path.exists(checkpoint_dir):
        for f in os.listdir(checkpoint_dir):
            if f.endswith('.pt') and f.startswith('checkpoint_'):
                checkpoints.append(os.path.join(checkpoint_dir, f))
    
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained GFlowNet on TSP problems")
    parser.add_argument("--problem", type=str, default=None,
                        help="Problem name to evaluate")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of stochastic samples for evaluation")
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Hidden dimension (must match training)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature for stochastic evaluation")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all problems")
    
    # Random instance evaluation
    parser.add_argument("--random", action="store_true",
                        help="Evaluate on a random instance")
    parser.add_argument("--num-cities", type=int, default=20,
                        help="Number of cities for random instance")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--solve-optimal", action="store_true",
                        help="Compute optimal solution for comparison (N<=20)")
    parser.add_argument("--list", action="store_true",
                        help="List available problems and exit")
    
    # Conditional evaluation
    parser.add_argument("--conditional", action="store_true",
                        help="Evaluate conditional GFlowNet model")
    parser.add_argument("--num-instances", type=int, default=10,
                        help="[Conditional] Number of instances to evaluate")
    parser.add_argument("--min-cities", type=int, default=10,
                        help="[Conditional] Min cities per instance")
    parser.add_argument("--max-cities", type=int, default=30,
                        help="[Conditional] Max cities per instance")
    
    args = parser.parse_args()
    
    if args.conditional:
        evaluate_conditional(args)
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    problems = list_tsp_instances(DATA_DIR)
    
    if args.list or (args.problem is None and not args.all and not args.random):
        print("Available TSP Problems:")
        print("-" * 60)
        print(f"{'Problem':<20} {'Cities':<10} {'Optimal':<15}")
        print("-" * 60)
        for p in problems:
            info = get_instance_info(DATA_DIR, p)
            if info:
                opt = f"{info['optimal_length']:.4f}" if info.get('optimal_length') else "?"
                print(f"{p:<20} {info['nodes']:<10} {opt:<15}")
        print("-" * 60)
        print("\nUse --problem <name> to evaluate a specific problem")
        print("Use --random --num-cities <N> to evaluate on a random instance")
        return
    
    # Determine which problems to evaluate
    if args.random:
        print(f"Generating random TSP instance with {args.num_cities} cities...")
        instance = generate_random_tsp(args.num_cities, seed=args.seed)
        
        # Optionally compute optimal solution
        solve_opt = args.solve_optimal or (args.num_cities <= 15)
        if solve_opt:
            print(f"Computing optimal solution...")
            tour, length = solve_tsp_optimal(instance['distance_matrix'], verbose=True)
            instance['optimal_tour'] = tour
            instance['optimal_length'] = length
        
        eval_problems = [instance['name']]
    elif args.all:
        eval_problems = problems
    else:
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
            
            print(f"  Cities: {results['cities']}")
            print(f"  Training step: {results['training_step']}")
            print(f"  Best during training: {results['best_training_length']}")
            print()
            print(f"  Greedy solution:")
            print(f"    Length: {results['greedy_length']:.2f}")
            tour_display = results['greedy_tour'][:11] if len(results['greedy_tour']) > 11 else results['greedy_tour']
            print(f"    Tour: {tour_display}" + ("..." if len(results['greedy_tour']) > 11 else ""))
            print()
            print(f"  Stochastic (best of {args.samples}):")
            print(f"    Length: {results['stochastic_length']:.2f}")
            
            if 'optimal_length' in results:
                print()
                print(f"  Optimal: {results['optimal_length']:.2f}")
                print(f"  Greedy gap: {results['greedy_gap']:.2f}%")
                print(f"  Stochastic gap: {results['stochastic_gap']:.2f}%")
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary table
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"{'Problem':<15} {'Optimal':<12} {'Greedy':<12} {'Gap%':<8} {'Stoch':<12} {'Gap%':<8}")
        print("-" * 70)
        for r in all_results:
            opt = f"{r.get('optimal_length', '?'):.1f}" if 'optimal_length' in r else "?"
            g_gap = f"{r['greedy_gap']:.1f}" if 'greedy_gap' in r else "?"
            s_gap = f"{r['stochastic_gap']:.1f}" if 'stochastic_gap' in r else "?"
            print(f"{r['problem']:<15} {opt:<12} {r['greedy_length']:<12.1f} {g_gap:<8} {r['stochastic_length']:<12.1f} {s_gap:<8}")
    
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
                "cities": r['cities'],
                "training_step": r['training_step'],
                "greedy": {
                    "length": float(r['greedy_length']),
                    "tour": r['greedy_tour'],
                },
                "stochastic": {
                    "length": float(r['stochastic_length']),
                    "tour": r['stochastic_tour'],
                },
            }
            if 'optimal_length' in r:
                entry['optimal_length'] = float(r['optimal_length'])
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
    """Create a ConditionalTSPPolicy from checkpoint."""
    forward_state = checkpoint['forward_policy']
    
    sample_key = list(forward_state.keys())[0]
    has_prefix = sample_key.startswith('shared_policy.')
    
    if has_prefix:
        encoder_key = 'shared_policy.city_encoder.0.weight'
    else:
        encoder_key = 'city_encoder.0.weight'
    
    hidden_dim = forward_state[encoder_key].shape[0]
    
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
    
    shared_policy = ConditionalTSPPolicy(
        hidden_dim=hidden_dim,
        num_layers=num_layers
    ).to(device)
    
    if has_prefix:
        forward = ConditionalTSPPolicyWrapper(shared_policy, mode='forward')
        forward.load_state_dict(forward_state)
        forward.eval()
        return shared_policy, hidden_dim, num_layers
    else:
        shared_policy.load_state_dict(forward_state)
        shared_policy.eval()
        return shared_policy, hidden_dim, num_layers


def sample_greedy_conditional(env, policy, instance, device):
    """Sample a solution using greedy policy for conditional model."""
    state = env.reset()
    coords = instance['coords']
    distance_matrix = instance['distance_matrix']
    
    with torch.no_grad():
        while True:
            logits = policy(state, coords, distance_matrix, device=device)
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


def sample_stochastic_conditional(env, policy, instance, device, num_samples=100, temperature=1.0):
    """Sample multiple solutions stochastically for conditional model."""
    best_state = None
    best_length = float('inf')
    
    coords = instance['coords']
    distance_matrix = instance['distance_matrix']
    
    with torch.no_grad():
        for _ in range(num_samples):
            state = env.reset()
            
            while True:
                logits = policy(state, coords, distance_matrix, device=device)
                mask = torch.tensor(env.allowed_actions(state), dtype=torch.float32, device=device)
                
                if mask.sum() == 0:
                    break
                
                masked_logits = logits.clone()
                masked_logits[mask == 0] = float('-inf')
                probs = torch.softmax(masked_logits / temperature, dim=0)
                action = torch.multinomial(probs, 1).item()
                
                state, reward, done = env.step(state, action)
                if done:
                    break
            
            length = env.get_tour_length(state)
            
            if length < best_length:
                best_length = length
                best_state = state.copy()
    
    return best_state, best_length


def evaluate_conditional(args):
    """Evaluate a conditional GFlowNet model on TSP problems."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
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
    
    # Generate test instances
    print(f"Generating {args.num_instances} test instances...")
    dataset = TSPInstanceDataset.generate_random(
        num_instances=args.num_instances,
        min_nodes=args.min_cities,
        max_nodes=args.max_cities,
        seed=args.seed
    )
    
    # Evaluate each instance
    print("=" * 80)
    print("CONDITIONAL GFLOWNET EVALUATION")
    print("=" * 80)
    
    all_results = []
    
    for i, instance in enumerate(dataset.instances):
        N = instance['N']
        env = TSPEnv(instance)
        
        print(f"\n[{instance['name']}] {N} cities")
        
        # Greedy evaluation
        greedy_state, greedy_reward = sample_greedy_conditional(env, shared_policy, instance, device)
        greedy_length = env.get_tour_length(greedy_state)
        
        # Stochastic evaluation
        stoch_state, stoch_length = sample_stochastic_conditional(
            env, shared_policy, instance, device, args.samples, args.temperature
        )
        
        results = {
            'problem': instance['name'],
            'cities': N,
            'greedy_length': float(greedy_length),
            'stochastic_length': float(stoch_length),
            'greedy_state': greedy_state,
            'stochastic_state': stoch_state,
        }
        
        all_results.append(results)
        
        print(f"  Greedy:     {greedy_length:.2f}")
        print(f"  Stochastic: {stoch_length:.2f}")
    
    # Summary
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"{'Problem':<20} {'Cities':<10} {'Greedy':<12} {'Stochastic':<12}")
        print("-" * 60)
        for r in all_results:
            print(f"{r['problem']:<20} {r['cities']:<10} {r['greedy_length']:<12.2f} {r['stochastic_length']:<12.2f}")
        
        avg_greedy = np.mean([r['greedy_length'] for r in all_results])
        avg_stoch = np.mean([r['stochastic_length'] for r in all_results])
        print("-" * 60)
        print(f"{'Average':<20} {'':<10} {avg_greedy:<12.2f} {avg_stoch:<12.2f}")
    
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
                "cities": r['cities'],
                "greedy": {
                    "length": r['greedy_length'],
                },
                "stochastic": {
                    "length": r['stochastic_length'],
                },
            }
            log_data['results'].append(entry)
        
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)
        
        print(f"\nEvaluation log saved to: {log_path}")


if __name__ == "__main__":
    main()
