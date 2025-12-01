# problems/tsp/optimal_solver.py
"""
Optimal TSP solvers and random instance generation.

For small instances (N <= 20), uses Held-Karp dynamic programming (exact).
For larger instances, uses branch-and-bound or nearest neighbor heuristic.
"""

import os
import numpy as np
from itertools import permutations
from typing import Tuple, Optional
import time

# Data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def generate_random_tsp(num_cities: int, seed: Optional[int] = None) -> dict:
    """
    Generate a random TSP instance with uniform random coordinates in [0, 1]^2.
    
    Args:
        num_cities: Number of cities
        seed: Random seed for reproducibility
        
    Returns:
        dict with keys:
            - 'name': instance name
            - 'coords': (N, 2) array of city coordinates
            - 'N': number of cities
            - 'distance_matrix': (N, N) Euclidean distance matrix
    """
    if seed is not None:
        np.random.seed(seed)
    
    coords = np.random.rand(num_cities, 2).astype(np.float32)
    
    # Compute Euclidean distance matrix
    diff = coords[:, None, :] - coords[None, :, :]
    distance_matrix = np.sqrt(np.sum(diff ** 2, axis=2)).astype(np.float32)
    
    return {
        'name': f'random_{num_cities}' + (f'_seed{seed}' if seed is not None else ''),
        'coords': coords,
        'N': num_cities,
        'distance_matrix': distance_matrix,
    }


def generate_clustered_tsp(num_cities: int, num_clusters: int = 3, 
                           cluster_std: float = 0.1, seed: Optional[int] = None) -> dict:
    """
    Generate a clustered TSP instance.
    
    Args:
        num_cities: Total number of cities
        num_clusters: Number of clusters
        cluster_std: Standard deviation of points around cluster centers
        seed: Random seed
        
    Returns:
        dict with TSP instance data
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate cluster centers
    centers = np.random.rand(num_clusters, 2)
    
    # Assign cities to clusters
    coords = []
    cities_per_cluster = num_cities // num_clusters
    
    for i in range(num_clusters):
        n = cities_per_cluster if i < num_clusters - 1 else num_cities - len(coords)
        cluster_coords = centers[i] + cluster_std * np.random.randn(n, 2)
        coords.extend(cluster_coords)
    
    coords = np.array(coords, dtype=np.float32)
    coords = np.clip(coords, 0, 1)  # Keep in unit square
    
    # Compute distance matrix
    diff = coords[:, None, :] - coords[None, :, :]
    distance_matrix = np.sqrt(np.sum(diff ** 2, axis=2)).astype(np.float32)
    
    return {
        'name': f'clustered_{num_cities}_{num_clusters}' + (f'_seed{seed}' if seed is not None else ''),
        'coords': coords,
        'N': num_cities,
        'distance_matrix': distance_matrix,
    }


def compute_tour_length(tour: np.ndarray, distance_matrix: np.ndarray) -> float:
    """Compute total length of a tour (including return to start)."""
    length = 0.0
    N = len(tour)
    for i in range(N):
        length += distance_matrix[tour[i], tour[(i + 1) % N]]
    return length


def save_tsp_instance(instance: dict, filepath: Optional[str] = None) -> str:
    """
    Save a TSP instance to TSPLIB format (single file with optimal solution included).
    
    Args:
        instance: TSP instance dict with 'coords', 'N', 'name', and optionally 'optimal_tour', 'optimal_length'
        filepath: Path to save (if None, saves to DATA_DIR/{name}.tsp)
        
    Returns:
        filepath: Path where the instance was saved
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    
    name = instance['name']
    coords = instance['coords']
    N = instance['N']
    
    if filepath is None:
        filepath = os.path.join(DATA_DIR, f"{name}.tsp")
    
    # Scale factor for coordinates (TSPLIB compatibility)
    SCALE = 10000
    
    # Write .tsp file in TSPLIB format with optimal solution included
    with open(filepath, 'w') as f:
        f.write(f"NAME : {name}\n")
        f.write(f"COMMENT : Generated random TSP instance with {N} cities\n")
        
        # Include optimal length and tour in comments if available
        # Tour includes return to starting city (first city repeated at end)
        # Note: optimal_length is scaled to match scaled coordinates
        if 'optimal_length' in instance:
            scaled_length = instance['optimal_length'] * SCALE
            f.write(f"COMMENT : OPTIMAL_LENGTH = {scaled_length:.6f}\n")
        if 'optimal_tour' in instance:
            # Add starting city at the end to show complete round trip (0-indexed)
            tour_with_return = list(instance['optimal_tour']) + [instance['optimal_tour'][0]]
            tour_str = " ".join(str(int(c)) for c in tour_with_return)  # 0-indexed
            f.write(f"COMMENT : OPTIMAL_TOUR = {tour_str}\n")
        
        f.write(f"TYPE : TSP\n")
        f.write(f"DIMENSION : {N}\n")
        f.write(f"EDGE_WEIGHT_TYPE : EUC_2D\n")
        f.write(f"NODE_COORD_SECTION\n")
        
        for i in range(N):
            # Scale coordinates to larger range for TSPLIB compatibility
            x = coords[i, 0] * SCALE
            y = coords[i, 1] * SCALE
            f.write(f"{i + 1} {x:.4f} {y:.4f}\n")
        
        f.write("EOF\n")
    
    print(f"Saved TSP instance to: {filepath}")
    
    return filepath


def generate_and_save_tsp(num_cities: int, seed: Optional[int] = None,
                          clustered: bool = False, num_clusters: int = 3,
                          solve: bool = True, name: Optional[str] = None) -> dict:
    """
    Generate a random TSP instance, optionally solve it, and save to data directory.
    
    Args:
        num_cities: Number of cities
        seed: Random seed
        clustered: If True, generate clustered instance
        num_clusters: Number of clusters (if clustered)
        solve: If True, find optimal/best solution
        name: Custom name (if None, auto-generated)
        
    Returns:
        instance: TSP instance dict with coordinates, distance matrix, and optimal solution
    """
    # Generate instance
    if clustered:
        instance = generate_clustered_tsp(num_cities, num_clusters, seed=seed)
    else:
        instance = generate_random_tsp(num_cities, seed=seed)
    
    # Override name if provided
    if name:
        instance['name'] = name
    
    # Solve
    if solve:
        print(f"\nSolving TSP with {num_cities} cities...")
        tour, length = solve_tsp_optimal(instance['distance_matrix'], verbose=True)
        instance['optimal_tour'] = tour
        instance['optimal_length'] = length
    
    # Save
    filepath = save_tsp_instance(instance)
    instance['filepath'] = filepath
    
    return instance


# =============================================================================
# Exact Solvers
# =============================================================================

def solve_tsp_bruteforce(distance_matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Solve TSP by brute force enumeration.
    Only feasible for N <= 10.
    
    Time complexity: O(N!)
    
    Args:
        distance_matrix: (N, N) distance matrix
        
    Returns:
        best_tour: Optimal tour as array of city indices
        best_length: Optimal tour length
    """
    N = distance_matrix.shape[0]
    
    if N > 10:
        raise ValueError(f"Brute force only feasible for N <= 10, got N={N}")
    
    best_tour = None
    best_length = float('inf')
    
    # Fix city 0 as start, permute the rest
    cities = list(range(1, N))
    
    for perm in permutations(cities):
        tour = np.array([0] + list(perm))
        length = compute_tour_length(tour, distance_matrix)
        
        if length < best_length:
            best_length = length
            best_tour = tour.copy()
    
    return best_tour, best_length


def solve_tsp_held_karp(distance_matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Solve TSP using Held-Karp dynamic programming algorithm.
    Feasible for N <= 20.
    
    Time complexity: O(N^2 * 2^N)
    Space complexity: O(N * 2^N)
    
    Args:
        distance_matrix: (N, N) distance matrix
        
    Returns:
        best_tour: Optimal tour as array of city indices
        best_length: Optimal tour length
    """
    N = distance_matrix.shape[0]
    
    if N > 20:
        raise ValueError(f"Held-Karp only feasible for N <= 20, got N={N}")
    
    if N == 1:
        return np.array([0]), 0.0
    
    if N == 2:
        return np.array([0, 1]), 2 * distance_matrix[0, 1]
    
    # dp[S][i] = minimum cost to visit all cities in set S, ending at city i
    # S is represented as a bitmask
    INF = float('inf')
    
    # Initialize
    dp = {}
    parent = {}
    
    # Base case: start from city 0, visit each other city
    for i in range(1, N):
        dp[(1 << i, i)] = distance_matrix[0, i]
        parent[(1 << i, i)] = 0
    
    # Iterate over subsets of increasing size
    for size in range(2, N):
        # Generate all subsets of size 'size' from cities 1 to N-1
        for subset in _subsets_of_size(N - 1, size):
            # Convert to bitmask (shift by 1 since we exclude city 0)
            S = sum(1 << (i + 1) for i in subset)
            
            for j in subset:
                j_bit = 1 << (j + 1)
                S_without_j = S ^ j_bit
                
                if S_without_j == 0:
                    continue
                
                best_cost = INF
                best_prev = -1
                
                for k in subset:
                    if k == j:
                        continue
                    k_bit = 1 << (k + 1)
                    
                    if (S_without_j, k + 1) in dp:
                        cost = dp[(S_without_j, k + 1)] + distance_matrix[k + 1, j + 1]
                        if cost < best_cost:
                            best_cost = cost
                            best_prev = k + 1
                
                if best_cost < INF:
                    dp[(S, j + 1)] = best_cost
                    parent[(S, j + 1)] = best_prev
    
    # Find the best final state (visiting all cities, ending at some city i)
    full_set = (1 << N) - 2  # All cities except 0
    best_length = INF
    best_last = -1
    
    for i in range(1, N):
        if (full_set, i) in dp:
            cost = dp[(full_set, i)] + distance_matrix[i, 0]
            if cost < best_length:
                best_length = cost
                best_last = i
    
    # Reconstruct tour
    tour = [0]
    S = full_set
    current = best_last
    
    while current != 0:
        tour.append(current)
        prev = parent.get((S, current), 0)
        S ^= (1 << current)
        current = prev
    
    return np.array(tour), best_length


def _subsets_of_size(n: int, k: int):
    """Generate all k-subsets of {0, 1, ..., n-1}."""
    if k == 0:
        yield []
        return
    if k > n:
        return
    
    for i in range(k - 1, n):
        for subset in _subsets_of_size(i, k - 1):
            yield subset + [i]


# =============================================================================
# Heuristic Solvers (for larger instances)
# =============================================================================

def solve_tsp_nearest_neighbor(distance_matrix: np.ndarray, start: int = 0) -> Tuple[np.ndarray, float]:
    """
    Solve TSP using nearest neighbor heuristic.
    
    Time complexity: O(N^2)
    
    Args:
        distance_matrix: (N, N) distance matrix
        start: Starting city
        
    Returns:
        tour: Tour as array of city indices
        length: Tour length
    """
    N = distance_matrix.shape[0]
    visited = [False] * N
    tour = [start]
    visited[start] = True
    
    current = start
    for _ in range(N - 1):
        best_next = -1
        best_dist = float('inf')
        
        for j in range(N):
            if not visited[j] and distance_matrix[current, j] < best_dist:
                best_dist = distance_matrix[current, j]
                best_next = j
        
        tour.append(best_next)
        visited[best_next] = True
        current = best_next
    
    return np.array(tour), compute_tour_length(np.array(tour), distance_matrix)


def solve_tsp_2opt(distance_matrix: np.ndarray, initial_tour: Optional[np.ndarray] = None,
                   max_iterations: int = 1000) -> Tuple[np.ndarray, float]:
    """
    Improve a TSP tour using 2-opt local search.
    
    Args:
        distance_matrix: (N, N) distance matrix
        initial_tour: Initial tour (if None, uses nearest neighbor)
        max_iterations: Maximum number of improvement iterations
        
    Returns:
        tour: Improved tour
        length: Tour length
    """
    N = distance_matrix.shape[0]
    
    if initial_tour is None:
        tour, _ = solve_tsp_nearest_neighbor(distance_matrix)
    else:
        tour = initial_tour.copy()
    
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        for i in range(N - 1):
            for j in range(i + 2, N):
                if j == N - 1 and i == 0:
                    continue  # Skip if it would reverse the entire tour
                
                # Calculate change in tour length if we reverse segment [i+1, j]
                # Current edges: (tour[i], tour[i+1]) and (tour[j], tour[j+1 mod N])
                # New edges: (tour[i], tour[j]) and (tour[i+1], tour[j+1 mod N])
                
                i1, i2 = tour[i], tour[i + 1]
                j1, j2 = tour[j], tour[(j + 1) % N]
                
                current_dist = distance_matrix[i1, i2] + distance_matrix[j1, j2]
                new_dist = distance_matrix[i1, j1] + distance_matrix[i2, j2]
                
                if new_dist < current_dist - 1e-10:
                    # Reverse the segment
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
    
    return tour, compute_tour_length(tour, distance_matrix)


# =============================================================================
# Main Solver Function
# =============================================================================

def solve_tsp_optimal(distance_matrix: np.ndarray, verbose: bool = True) -> Tuple[np.ndarray, float]:
    """
    Find the optimal TSP solution.
    
    - For N <= 10: Uses brute force
    - For N <= 20: Uses Held-Karp DP
    - For N > 20: Uses nearest neighbor + 2-opt (heuristic, not guaranteed optimal)
    
    Args:
        distance_matrix: (N, N) distance matrix
        verbose: Print progress information
        
    Returns:
        tour: Best tour found
        length: Tour length
    """
    N = distance_matrix.shape[0]
    
    start_time = time.time()
    
    if N <= 10:
        if verbose:
            print(f"Using brute force for N={N}...")
        tour, length = solve_tsp_bruteforce(distance_matrix)
        method = "brute_force"
    elif N <= 20:
        if verbose:
            print(f"Using Held-Karp DP for N={N}...")
        tour, length = solve_tsp_held_karp(distance_matrix)
        method = "held_karp"
    else:
        if verbose:
            print(f"Using nearest neighbor + 2-opt heuristic for N={N}...")
        tour, length = solve_tsp_nearest_neighbor(distance_matrix)
        tour, length = solve_tsp_2opt(distance_matrix, tour)
        method = "heuristic"
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"  Method: {method}")
        print(f"  Tour length: {length:.6f}")
        print(f"  Time: {elapsed:.3f}s")
    
    return tour, length


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate random TSP and find optimal solution")
    parser.add_argument("--num-cities", "-n", type=int, default=10,
                        help="Number of cities (default: 10)")
    parser.add_argument("--seed", "-s", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--clustered", action="store_true",
                        help="Generate clustered instance instead of uniform random")
    parser.add_argument("--num-clusters", type=int, default=3,
                        help="Number of clusters for clustered instance")
    parser.add_argument("--name", type=str, default=None,
                        help="Custom name for the instance")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save to file (just print results)")
    parser.add_argument("--no-solve", action="store_true",
                        help="Don't solve for optimal (just generate and save)")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Generating TSP instance with {args.num_cities} cities")
    print(f"{'='*60}\n")
    
    if args.no_save:
        # Just generate and solve without saving
        if args.clustered:
            instance = generate_clustered_tsp(args.num_cities, args.num_clusters, seed=args.seed)
        else:
            instance = generate_random_tsp(args.num_cities, seed=args.seed)
        
        if args.name:
            instance['name'] = args.name
        
        print(f"Instance: {instance['name']}")
        print(f"Cities: {instance['N']}")
        print(f"\nCoordinates (first 5):")
        for i in range(min(5, instance['N'])):
            print(f"  City {i}: ({instance['coords'][i, 0]:.4f}, {instance['coords'][i, 1]:.4f})")
        if instance['N'] > 5:
            print(f"  ... and {instance['N'] - 5} more")
        
        if not args.no_solve:
            print(f"\n{'='*60}")
            print("Finding optimal solution...")
            print(f"{'='*60}\n")
            
            tour, length = solve_tsp_optimal(instance['distance_matrix'], verbose=True)
            
            print(f"\nOptimal tour: {tour.tolist()}")
            print(f"Optimal length: {length:.6f}")
            
            verified_length = compute_tour_length(tour, instance['distance_matrix'])
            print(f"Verified length: {verified_length:.6f}")
            
            instance['optimal_tour'] = tour
            instance['optimal_length'] = length
    else:
        # Generate, solve, and save
        instance = generate_and_save_tsp(
            num_cities=args.num_cities,
            seed=args.seed,
            clustered=args.clustered,
            num_clusters=args.num_clusters,
            solve=not args.no_solve,
            name=args.name
        )
        
        print(f"\nInstance: {instance['name']}")
        print(f"Cities: {instance['N']}")
        
        if 'optimal_length' in instance:
            print(f"Optimal tour: {instance['optimal_tour'].tolist()}")
            print(f"Optimal length: {instance['optimal_length']:.6f}")
        
        print(f"\nSaved to: {instance['filepath']}")
    
    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")
