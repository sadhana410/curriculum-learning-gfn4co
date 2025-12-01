# problems/tsp/utils.py

import os
import numpy as np


def load_tsp_file(filepath):
    """
    Load a TSP instance from TSPLIB format (.tsp file).
    
    Supports:
        - EUC_2D: Euclidean distance in 2D
        - GEO: Geographic distance
        - EXPLICIT: Distance matrix provided
    
    Returns:
        dict with keys:
            - 'name': instance name
            - 'coords': (N, 2) array of node coordinates
            - 'N': number of nodes
            - 'distance_matrix': (N, N) distance matrix
            - 'optimal_tour': optimal tour if available (from comments or .opt.tour file)
            - 'optimal_length': optimal tour length if available
    """
    name = None
    dimension = None
    edge_weight_type = None
    coords = []
    reading_coords = False
    optimal_length = None
    optimal_tour = None
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('NAME'):
                name = line.split(':')[1].strip()
            elif line.startswith('DIMENSION'):
                dimension = int(line.split(':')[1].strip())
            elif line.startswith('EDGE_WEIGHT_TYPE'):
                edge_weight_type = line.split(':')[1].strip()
            elif line.startswith('COMMENT') and 'OPTIMAL_LENGTH' in line:
                # Parse optimal length from comment
                try:
                    optimal_length = float(line.split('=')[1].strip())
                except (ValueError, IndexError):
                    pass
            elif line.startswith('COMMENT') and 'OPTIMAL_TOUR' in line:
                # Parse optimal tour from comment (0-indexed)
                # Tour may include return to start (first city repeated at end)
                try:
                    tour_str = line.split('=')[1].strip()
                    tour_list = [int(x) for x in tour_str.split()]  # Already 0-indexed
                    # Remove last element if it's the same as first (return to start)
                    if len(tour_list) > 1 and tour_list[-1] == tour_list[0]:
                        tour_list = tour_list[:-1]
                    optimal_tour = np.array(tour_list, dtype=np.int32)
                except (ValueError, IndexError):
                    pass
            elif line == 'NODE_COORD_SECTION':
                reading_coords = True
            elif line == 'EOF' or line == '':
                reading_coords = False
            elif reading_coords:
                parts = line.split()
                if len(parts) >= 3:
                    # node_id, x, y
                    x = float(parts[1])
                    y = float(parts[2])
                    coords.append([x, y])
    
    coords = np.array(coords, dtype=np.float32)
    N = len(coords)
    
    # Compute distance matrix
    distance_matrix = compute_distance_matrix(coords, edge_weight_type)
    
    instance = {
        'name': name or os.path.basename(filepath).replace('.tsp', ''),
        'coords': coords,
        'N': N,
        'distance_matrix': distance_matrix,
        'edge_weight_type': edge_weight_type,
    }
    
    # Add optimal solution if found in comments
    if optimal_tour is not None:
        instance['optimal_tour'] = optimal_tour
        instance['optimal_length'] = compute_tour_length(optimal_tour, distance_matrix)
    elif optimal_length is not None:
        instance['optimal_length'] = optimal_length
    
    # Fallback: try to load from separate .opt.tour file
    if 'optimal_tour' not in instance:
        opt_path = filepath.replace('.tsp', '.opt.tour')
        if os.path.exists(opt_path):
            tour = load_tour_file(opt_path)
            if tour is not None:
                instance['optimal_tour'] = tour
                instance['optimal_length'] = compute_tour_length(tour, distance_matrix)
    
    return instance


def compute_distance_matrix(coords, edge_weight_type='EUC_2D'):
    """Compute distance matrix from coordinates."""
    N = len(coords)
    
    if edge_weight_type == 'EUC_2D':
        # Euclidean distance
        diff = coords[:, None, :] - coords[None, :, :]  # (N, N, 2)
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
    elif edge_weight_type == 'GEO':
        # Geographic distance (great circle)
        dist = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in range(N):
                if i != j:
                    dist[i, j] = geo_distance(coords[i], coords[j])
    else:
        # Default to Euclidean
        diff = coords[:, None, :] - coords[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
    
    return dist.astype(np.float32)


def geo_distance(coord1, coord2):
    """Compute geographic distance between two coordinates."""
    PI = 3.141592
    RRR = 6378.388
    
    lat1 = PI * coord1[0] / 180.0
    lon1 = PI * coord1[1] / 180.0
    lat2 = PI * coord2[0] / 180.0
    lon2 = PI * coord2[1] / 180.0
    
    q1 = np.cos(lon1 - lon2)
    q2 = np.cos(lat1 - lat2)
    q3 = np.cos(lat1 + lat2)
    
    return int(RRR * np.arccos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)


def load_tour_file(filepath):
    """Load a tour from TSPLIB .tour file."""
    tour = []
    reading_tour = False
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line == 'TOUR_SECTION':
                reading_tour = True
            elif line == '-1' or line == 'EOF':
                reading_tour = False
            elif reading_tour:
                try:
                    node = int(line)
                    if node > 0:  # TSPLIB uses 1-indexed
                        tour.append(node - 1)  # Convert to 0-indexed
                except ValueError:
                    pass
    
    return np.array(tour, dtype=np.int32) if tour else None


def compute_tour_length(tour, distance_matrix):
    """Compute total length of a tour."""
    length = 0.0
    N = len(tour)
    for i in range(N):
        length += distance_matrix[tour[i], tour[(i + 1) % N]]
    return length


def list_tsp_instances(data_dir):
    """List all TSP instances in a directory."""
    instances = []
    if os.path.exists(data_dir):
        for fname in sorted(os.listdir(data_dir)):
            if fname.endswith('.tsp'):
                instances.append(fname.replace('.tsp', ''))
    return instances


def get_instance_info(data_dir, instance_name):
    """Get basic info about a TSP instance."""
    filepath = os.path.join(data_dir, f"{instance_name}.tsp")
    if not os.path.exists(filepath):
        filepath = os.path.join(data_dir, instance_name)
        if not os.path.exists(filepath):
            return None
    
    instance = load_tsp_file(filepath)
    return {
        'name': instance['name'],
        'nodes': instance['N'],
        'optimal_length': instance.get('optimal_length', None),
    }


def generate_random_tsp(num_nodes, seed=None):
    """
    Generate a random TSP instance with uniform random coordinates.
    
    Args:
        num_nodes: Number of cities
        seed: Random seed
        
    Returns:
        dict with TSP instance data
    """
    if seed is not None:
        np.random.seed(seed)
    
    coords = np.random.rand(num_nodes, 2).astype(np.float32)
    distance_matrix = compute_distance_matrix(coords, 'EUC_2D')
    
    return {
        'name': f'random_{num_nodes}',
        'coords': coords,
        'N': num_nodes,
        'distance_matrix': distance_matrix,
        'edge_weight_type': 'EUC_2D',
    }


def generate_clustered_tsp(num_nodes, num_clusters=3, seed=None):
    """
    Generate a clustered TSP instance.
    
    Args:
        num_nodes: Total number of cities
        num_clusters: Number of clusters
        seed: Random seed
        
    Returns:
        dict with TSP instance data
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate cluster centers
    centers = np.random.rand(num_clusters, 2)
    
    # Assign nodes to clusters
    nodes_per_cluster = num_nodes // num_clusters
    coords = []
    
    for i in range(num_clusters):
        n = nodes_per_cluster if i < num_clusters - 1 else num_nodes - len(coords)
        cluster_coords = centers[i] + 0.1 * np.random.randn(n, 2)
        coords.extend(cluster_coords)
    
    coords = np.array(coords, dtype=np.float32)
    coords = np.clip(coords, 0, 1)  # Keep in unit square
    
    distance_matrix = compute_distance_matrix(coords, 'EUC_2D')
    
    return {
        'name': f'clustered_{num_nodes}_{num_clusters}',
        'coords': coords,
        'N': num_nodes,
        'distance_matrix': distance_matrix,
        'edge_weight_type': 'EUC_2D',
    }
