import numpy as np
import torch

import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from envs.base_env import BaseEnv


class GraphColoringEnv(BaseEnv):
    """
    Graph coloring environment for a single fixed graph instance.
    """
    def __init__(self, instance, num_colors=3, chromatic_number=None):
        self.adj = instance["adj"]
        self.N = self.adj.shape[0]
        self.K = num_colors
        # Chromatic number is the minimum colors needed (if known)
        self.chromatic_number = chromatic_number if chromatic_number else num_colors
        # Pre-compute numpy adjacency for faster operations
        if isinstance(self.adj, torch.Tensor):
            self._adj_np = self.adj.cpu().numpy()
        else:
            self._adj_np = np.array(self.adj)
        super().__init__(instance)

    def reset(self):
        self.state = -1 * np.ones(self.N, dtype=int)
        return self.state.copy()

    def allowed_actions(self, state):
        """Vectorized allowed actions computation."""
        mask = np.zeros((self.N, self.K), dtype=np.float32)
        
        # Find uncolored nodes
        uncolored = state == -1
        
        for node in np.where(uncolored)[0]:
            # Get neighbors of this node
            neighbors = np.where(self._adj_np[node] == 1)[0]
            # Get colors used by neighbors
            neighbor_colors = state[neighbors]
            neighbor_colors = neighbor_colors[neighbor_colors != -1]
            # All colors except those used by neighbors are valid
            valid_colors = np.ones(self.K, dtype=np.float32)
            valid_colors[neighbor_colors] = 0.0
            mask[node] = valid_colors
        
        return mask.flatten()

    def step(self, state, action):
        node = action // self.K
        color = action % self.K
        next_state = state.copy()
        next_state[node] = color
        done = self.is_terminal(next_state)
        reward = self.reward(next_state) if done else 0.0
        return next_state, reward, done

    def step_batch(self, states, actions):
        """
        Vectorized step for a batch of states.
        states: (B, N) numpy array
        actions: (B,) numpy array
        """
        B, N = states.shape
        K = self.K
        
        nodes = actions // K
        colors = actions % K
        
        next_states = states.copy()
        # Advanced indexing for batch update
        # next_states[range(B), nodes] = colors
        # But nodes is (B,), colors is (B,)
        next_states[np.arange(B), nodes] = colors
        
        # Check done (only fully colored check here, stuck check handled by sampler)
        dones = (next_states != -1).all(axis=1)
        
        # Compute rewards for done states
        rewards = np.zeros(B, dtype=np.float32)
        if dones.any():
            done_states = next_states[dones]
            rewards[dones] = self.reward_batch(done_states)
            
        return next_states, rewards, dones

    def is_terminal(self, state):
        # Terminal if all nodes are colored
        if np.all(state != -1):
            return True
        # Also terminal if no valid actions remain (stuck)
        mask = self.allowed_actions(state)
        return mask.sum() == 0

    def reward(self, state):
        # Single state reward
        colored_mask = state != -1
        same_color = (state[:, None] == state[None, :]) & colored_mask[:, None] & colored_mask[None, :]
        conflicts = int(np.sum(self._adj_np * same_color) // 2)
        
        colored = np.sum(colored_mask)
        colors_used = len(set(c for c in state if c != -1)) if colored > 0 else 0
        missing = self.N - colored

        # logR as a smooth function
        alpha = 1.0   # penalty per conflict
        beta  = 0.5   # penalty per color used
        gamma = 0.2   # penalty per uncolored node

        logR = -alpha * conflicts - beta * colors_used - gamma * missing
        
        # Scale by N to match log_pf magnitude
        logR = logR * self.N

        return float(logR)

    def reward_batch(self, states):
        """Vectorized reward for batch of states (B, N)."""
        B, N = states.shape
        
        colored_mask = (states != -1) # (B, N)
        
        # same_color: (B, N, N)
        same_color = (states[:, :, None] == states[:, None, :]) & \
                     colored_mask[:, :, None] & \
                     colored_mask[:, None, :]
                     
        # Conflicts: sum(adj * same_color)
        adj_broadcast = self._adj_np[None, :, :]
        conflicts = np.sum(adj_broadcast * same_color, axis=(1, 2)) // 2 # (B,)
        
        # Colors used: (B,)
        colors_used = np.zeros(B, dtype=np.float32)
        for k in range(self.K):
            has_color = (states == k).any(axis=1)
            colors_used += has_color
            
        colored_counts = colored_mask.sum(axis=1)
        missing = N - colored_counts
        
        # Parameters
        alpha = 1.0
        beta = 0.5
        gamma = 0.2
        
        logR = -alpha * conflicts - beta * colors_used - gamma * missing
        
        # Scale by N
        logR = logR * N
        
        return logR

    def encode_state(self, state):
        one_hot = np.zeros((self.N, self.K), dtype=np.float32)
        for i in range(self.N):
            if state[i] != -1:
                one_hot[i, state[i]] = 1.0
        return one_hot.flatten()

    def _conflict(self, state, node, color):
        for nbr in range(self.N):
            if self.adj[node, nbr] == 1 and state[nbr] == color:
                return True
        return False


class ConditionalGraphColoringEnv:
    """
    Conditional Graph Coloring Environment for training on multiple graph instances.
    
    This environment supports sampling from a distribution of graphs and is used
    for conditional GFlowNet training where the policy must generalize across
    different graph structures.
    """
    
    def __init__(self, instances, num_colors=3):
        """
        Args:
            instances: List of graph instances, each containing:
                - 'adj': adjacency matrix (numpy array or torch tensor)
                - 'chromatic_number': (optional) known chromatic number
                - 'name': (optional) instance name
            num_colors: Maximum number of colors available
        """
        self.instances = instances
        self.K = num_colors
        self.num_instances = len(instances)
        
        # Pre-process instances
        self._processed_instances = []
        for inst in instances:
            adj = inst['adj']
            if isinstance(adj, torch.Tensor):
                adj_np = adj.cpu().numpy()
            else:
                adj_np = np.array(adj)
            
            self._processed_instances.append({
                'adj': adj_np,
                'N': adj_np.shape[0],
                'chromatic_number': inst.get('chromatic_number', num_colors),
                'name': inst.get('name', 'unknown')
            })
        
        # Current instance index (for single-instance operations)
        self._current_idx = 0
        self._current_instance = self._processed_instances[0]
        
        # For compatibility with existing code
        self.N = self._current_instance['N']
        self._adj_np = self._current_instance['adj']
        self.adj = self._adj_np
        self.chromatic_number = self._current_instance['chromatic_number']
    
    def set_instance(self, idx):
        """Set the current instance by index."""
        self._current_idx = idx
        self._current_instance = self._processed_instances[idx]
        self.N = self._current_instance['N']
        self._adj_np = self._current_instance['adj']
        self.adj = self._adj_np
        self.chromatic_number = self._current_instance['chromatic_number']
    
    def sample_instance(self):
        """Randomly sample an instance and set it as current."""
        idx = np.random.randint(self.num_instances)
        self.set_instance(idx)
        return idx
    
    def get_instance(self, idx):
        """Get instance data by index."""
        return self._processed_instances[idx]
    
    def reset(self, instance_idx=None):
        """Reset environment, optionally with a specific instance."""
        if instance_idx is not None:
            self.set_instance(instance_idx)
        self.state = -1 * np.ones(self.N, dtype=int)
        return self.state.copy()
    
    def allowed_actions(self, state):
        """Compute allowed actions mask for current instance."""
        mask = np.zeros((self.N, self.K), dtype=np.float32)
        uncolored = state == -1
        
        for node in np.where(uncolored)[0]:
            neighbors = np.where(self._adj_np[node] == 1)[0]
            neighbor_colors = state[neighbors]
            neighbor_colors = neighbor_colors[neighbor_colors != -1]
            valid_colors = np.ones(self.K, dtype=np.float32)
            valid_colors[neighbor_colors] = 0.0
            mask[node] = valid_colors
        
        return mask.flatten()
    
    def step(self, state, action):
        """Take a step in the environment."""
        node = action // self.K
        color = action % self.K
        next_state = state.copy()
        next_state[node] = color
        done = self.is_terminal(next_state)
        reward = self.reward(next_state) if done else 0.0
        return next_state, reward, done
    
    def is_terminal(self, state):
        """Check if state is terminal."""
        if np.all(state != -1):
            return True
        mask = self.allowed_actions(state)
        return mask.sum() == 0
    
    def reward(self, state):
        """Compute log-reward for terminal state."""
        colored_mask = state != -1
        same_color = (state[:, None] == state[None, :]) & colored_mask[:, None] & colored_mask[None, :]
        conflicts = int(np.sum(self._adj_np * same_color) // 2)
        
        colored = np.sum(colored_mask)
        colors_used = len(set(c for c in state if c != -1)) if colored > 0 else 0
        missing = self.N - colored
        
        alpha = 1.0
        beta = 0.5
        gamma = 0.2
        
        logR = -alpha * conflicts - beta * colors_used - gamma * missing
        logR = logR * self.N
        
        return float(logR)
    
    def step_batch(self, states, actions):
        """Vectorized step for batch of states (same instance)."""
        B, N = states.shape
        K = self.K
        
        nodes = actions // K
        colors = actions % K
        
        next_states = states.copy()
        next_states[np.arange(B), nodes] = colors
        
        dones = (next_states != -1).all(axis=1)
        
        rewards = np.zeros(B, dtype=np.float32)
        if dones.any():
            done_states = next_states[dones]
            rewards[dones] = self.reward_batch(done_states)
        
        return next_states, rewards, dones
    
    def reward_batch(self, states):
        """Vectorized reward for batch of states."""
        B, N = states.shape
        
        colored_mask = (states != -1)
        same_color = (states[:, :, None] == states[:, None, :]) & \
                     colored_mask[:, :, None] & \
                     colored_mask[:, None, :]
        
        adj_broadcast = self._adj_np[None, :, :]
        conflicts = np.sum(adj_broadcast * same_color, axis=(1, 2)) // 2
        
        colors_used = np.zeros(B, dtype=np.float32)
        for k in range(self.K):
            has_color = (states == k).any(axis=1)
            colors_used += has_color
        
        colored_counts = colored_mask.sum(axis=1)
        missing = N - colored_counts
        
        alpha = 1.0
        beta = 0.5
        gamma = 0.2
        
        logR = -alpha * conflicts - beta * colors_used - gamma * missing
        logR = logR * N
        
        return logR
    
    def encode_state(self, state):
        """Encode state as feature vector."""
        one_hot = np.zeros((self.N, self.K), dtype=np.float32)
        for i in range(self.N):
            if state[i] != -1:
                one_hot[i, state[i]] = 1.0
        return one_hot.flatten()


class GraphInstanceDataset:
    """
    Dataset of graph instances for conditional GFlowNet training.
    Supports loading from files and generating random graphs.
    """
    
    def __init__(self, instances=None, data_dir=None):
        """
        Args:
            instances: List of pre-loaded instances
            data_dir: Directory containing .col files
        """
        self.instances = instances if instances is not None else []
        
        if data_dir is not None:
            self._load_from_dir(data_dir)
    
    def _load_from_dir(self, data_dir):
        """Load all .col files from directory."""
        from problems.graph_coloring.utils import load_col_file
        
        if not os.path.exists(data_dir):
            return
        
        for fname in sorted(os.listdir(data_dir)):
            if fname.endswith('.col'):
                path = os.path.join(data_dir, fname)
                adj = load_col_file(path)
                self.instances.append({
                    'adj': adj,
                    'name': fname.replace('.col', ''),
                    'chromatic_number': None  # Unknown by default
                })
    
    def add_instance(self, adj, name=None, chromatic_number=None):
        """Add a single instance."""
        self.instances.append({
            'adj': adj,
            'name': name or f'graph_{len(self.instances)}',
            'chromatic_number': chromatic_number
        })
    
    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx):
        return self.instances[idx]
    
    def sample(self, n=1):
        """Sample n random instances."""
        indices = np.random.choice(len(self.instances), size=n, replace=True)
        return [self.instances[i] for i in indices]
    
    def split(self, train_ratio=0.8, seed=42):
        """Split dataset into train and test sets."""
        np.random.seed(seed)
        indices = np.random.permutation(len(self.instances))
        split_idx = int(len(indices) * train_ratio)
        
        train_instances = [self.instances[i] for i in indices[:split_idx]]
        test_instances = [self.instances[i] for i in indices[split_idx:]]
        
        return GraphInstanceDataset(train_instances), GraphInstanceDataset(test_instances)
    
    @staticmethod
    def generate_random_graphs(num_graphs, min_nodes=10, max_nodes=50, 
                               edge_prob=0.3, seed=None):
        """Generate random Erdos-Renyi graphs."""
        if seed is not None:
            np.random.seed(seed)
        
        instances = []
        for i in range(num_graphs):
            n = np.random.randint(min_nodes, max_nodes + 1)
            adj = np.zeros((n, n), dtype=np.float32)
            
            for u in range(n):
                for v in range(u + 1, n):
                    if np.random.random() < edge_prob:
                        adj[u, v] = 1
                        adj[v, u] = 1
            
            instances.append({
                'adj': adj,
                'name': f'random_{i}',
                'chromatic_number': None
            })
        
        return GraphInstanceDataset(instances)
