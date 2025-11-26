# problems/knapsack/utils.py

import os
import numpy as np


def load_knapsack_instance(data_dir, problem_name="p01"):
    """
    Load a knapsack instance from a problem folder.
    
    Structure expected:
        data_dir/{problem_name}/{problem_name}_c.txt - capacity (single number)
        data_dir/{problem_name}/{problem_name}_p.txt - profits (one per line)
        data_dir/{problem_name}/{problem_name}_w.txt - weights (one per line)
        data_dir/{problem_name}/{problem_name}_s.txt - optimal solution (optional, binary)
    
    Returns:
        dict with keys: 'profits', 'weights', 'capacity', 'optimal_solution' (if available)
    """
    problem_dir = os.path.join(data_dir, problem_name)
    
    capacity_file = os.path.join(problem_dir, f"{problem_name}_c.txt")
    profits_file = os.path.join(problem_dir, f"{problem_name}_p.txt")
    weights_file = os.path.join(problem_dir, f"{problem_name}_w.txt")
    solution_file = os.path.join(problem_dir, f"{problem_name}_s.txt")
    
    # Load capacity
    with open(capacity_file, 'r') as f:
        capacity = int(f.read().strip())
    
    # Load profits
    with open(profits_file, 'r') as f:
        profits = [int(line.strip()) for line in f if line.strip()]
    
    # Load weights
    with open(weights_file, 'r') as f:
        weights = [int(line.strip()) for line in f if line.strip()]
    
    instance = {
        'name': problem_name,
        'profits': np.array(profits, dtype=np.float32),
        'weights': np.array(weights, dtype=np.float32),
        'capacity': capacity,
    }
    
    # Load optimal solution if available
    if os.path.exists(solution_file):
        with open(solution_file, 'r') as f:
            solution = [int(line.strip()) for line in f if line.strip()]
        instance['optimal_solution'] = np.array(solution, dtype=np.int32)
        instance['optimal_profit'] = np.sum(instance['profits'] * instance['optimal_solution'])
    
    return instance


def list_knapsack_instances(data_dir):
    """List all knapsack problem folders in the data directory."""
    problems = []
    for name in os.listdir(data_dir):
        problem_dir = os.path.join(data_dir, name)
        if os.path.isdir(problem_dir):
            # Check if it has the required files
            if os.path.exists(os.path.join(problem_dir, f"{name}_c.txt")):
                problems.append(name)
    return sorted(problems)


def get_instance_info(data_dir, problem_name):
    """Get basic info about an instance without loading all data."""
    instance = load_knapsack_instance(data_dir, problem_name)
    return {
        'name': problem_name,
        'items': len(instance['profits']),
        'capacity': instance['capacity'],
        'optimal_profit': instance.get('optimal_profit', None),
    }
