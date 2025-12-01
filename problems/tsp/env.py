# problems/tsp/env.py

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from envs.base_env import BaseEnv


class TSPEnv(BaseEnv):
    """
    Traveling Salesman Problem environment for GFlowNet.
    
    State: Array of length N where:
        - state[i] = position in tour (0 to N-1) if city i has been visited
        - state[i] = -1 if city i has not been visited yet
        
    Action: Index of city to visit next (0 to N-1)
    
    The tour starts from city 0 and must visit all cities exactly once,
    returning to city 0 at the end.
    """
    
    def __init__(self, instance, alpha=1.0, beta=0.1):
        """
        Args:
            instance: dict containing:
                - 'coords': (N, 2) array of city coordinates
                - 'distance_matrix': (N, N) distance matrix
                - 'N': number of cities
            alpha: Reward coefficient for tour length (negative)
            beta: Scaling factor for reward normalization
        """
        self.coords = np.array(instance['coords'], dtype=np.float32)
        self.distance_matrix = np.array(instance['distance_matrix'], dtype=np.float32)
        self.N = instance['N']
        
        # Reward hyperparameters
        self.alpha = alpha
        self.beta = beta
        
        # Precompute for normalization
        self.max_distance = np.max(self.distance_matrix)
        self.avg_distance = np.mean(self.distance_matrix[self.distance_matrix > 0])
        
        # Normalize distance matrix for reward computation
        self.distance_matrix_norm = self.distance_matrix / (self.max_distance + 1e-8)
        
        super().__init__(instance)
    
    def reset(self):
        """
        Reset environment. Start tour from city 0.
        
        Returns:
            state: Array where state[0] = 0 (first city in tour), rest = -1
        """
        self.state = -1 * np.ones(self.N, dtype=np.int32)
        self.state[0] = 0  # City 0 is the first city in the tour
        self.current_step = 1  # Next position to fill
        self.current_city = 0  # Current city we're at
        return self.state.copy()
    
    def allowed_actions(self, state):
        """
        Get mask of allowed actions.
        
        Actions: 0 to N-1 (visit city i)
        Only unvisited cities are valid actions.
        
        Returns:
            mask: (N,) array where 1 = valid action, 0 = invalid
        """
        mask = np.zeros(self.N, dtype=np.float32)
        
        # Unvisited cities are valid actions
        unvisited = state == -1
        mask[unvisited] = 1.0
        
        return mask
    
    def step(self, state, action):
        """
        Take a step: visit the specified city.
        
        Args:
            state: Current state
            action: City index to visit
            
        Returns:
            next_state: Updated state
            reward: Log-reward (only at terminal state)
            done: Whether episode is finished
        """
        next_state = state.copy()
        
        # Find current position in tour (how many cities visited)
        current_step = np.sum(state != -1)
        
        # Mark this city as visited at current position
        next_state[action] = current_step
        
        done = self.is_terminal(next_state)
        reward = self.reward(next_state) if done else 0.0
        
        return next_state, reward, done
    
    def step_batch(self, states, actions):
        """
        Vectorized step for batch of states.
        
        Args:
            states: (B, N) array
            actions: (B,) array
            
        Returns:
            next_states: (B, N) array
            rewards: (B,) array
            dones: (B,) boolean array
        """
        B, N = states.shape
        next_states = states.copy()
        
        # Find current step for each trajectory
        current_steps = np.sum(states != -1, axis=1)  # (B,)
        
        # Mark visited cities
        next_states[np.arange(B), actions] = current_steps
        
        # Check done
        dones = (next_states != -1).all(axis=1)
        
        # Compute rewards for done states
        rewards = np.zeros(B, dtype=np.float32)
        if dones.any():
            done_states = next_states[dones]
            rewards[dones] = self.reward_batch(done_states)
        
        return next_states, rewards, dones
    
    def is_terminal(self, state):
        """Check if all cities have been visited."""
        return np.all(state != -1)
    
    def get_tour_from_state(self, state):
        """
        Convert state to tour order.
        
        Args:
            state: Array where state[i] = position of city i in tour
            
        Returns:
            tour: Array of city indices in visit order
        """
        # Sort cities by their position in the tour
        tour = np.argsort(state)
        return tour
    
    def get_tour_length(self, state):
        """
        Compute total tour length from state.
        
        Args:
            state: Terminal state with all cities visited
            
        Returns:
            length: Total tour length (including return to start)
        """
        tour = self.get_tour_from_state(state)
        length = 0.0
        for i in range(self.N):
            length += self.distance_matrix[tour[i], tour[(i + 1) % self.N]]
        return length
    
    def reward(self, state):
        """
        Compute log-reward for terminal state.
        
        Uses negative tour length scaled by N to match log_pf magnitude.
        
        Args:
            state: Terminal state
            
        Returns:
            logR: Log-reward (higher is better, so negative tour length)
        """
        tour_length = self.get_tour_length(state)
        
        # Normalize tour length
        tour_length_norm = tour_length / (self.N * self.avg_distance + 1e-8)
        
        # Log-reward: negative normalized tour length
        logR = -self.alpha * tour_length_norm
        
        # Scale by N to match log_pf magnitude
        logR = logR * self.N
        
        return float(logR)
    
    def reward_batch(self, states):
        """
        Vectorized reward for batch of terminal states.
        
        Args:
            states: (B, N) array of terminal states
            
        Returns:
            logR: (B,) array of log-rewards
        """
        B, N = states.shape
        
        # Get tours for each state
        tours = np.argsort(states, axis=1)  # (B, N)
        
        # Compute tour lengths
        lengths = np.zeros(B, dtype=np.float32)
        for i in range(N):
            from_cities = tours[:, i]
            to_cities = tours[:, (i + 1) % N]
            lengths += self.distance_matrix[from_cities, to_cities]
        
        # Normalize
        lengths_norm = lengths / (N * self.avg_distance + 1e-8)
        
        # Log-reward
        logR = -self.alpha * lengths_norm
        logR = logR * N
        
        return logR
    
    def encode_state(self, state):
        """
        Encode state as feature vector.
        
        Features per city:
            - visited (binary)
            - position in tour (normalized)
            - x coordinate (normalized)
            - y coordinate (normalized)
        """
        features = np.zeros((self.N, 4), dtype=np.float32)
        
        # Visited flag
        features[:, 0] = (state != -1).astype(np.float32)
        
        # Position in tour (normalized)
        visited_mask = state != -1
        features[visited_mask, 1] = state[visited_mask] / (self.N - 1)
        
        # Coordinates (normalized to [0, 1])
        coords_min = self.coords.min(axis=0)
        coords_max = self.coords.max(axis=0)
        coords_norm = (self.coords - coords_min) / (coords_max - coords_min + 1e-8)
        features[:, 2:4] = coords_norm
        
        return features.flatten()


class TSPInstanceDataset:
    """
    Dataset of TSP instances for conditional training.
    """
    
    def __init__(self, instances=None, data_dir=None):
        """
        Args:
            instances: List of pre-loaded instances
            data_dir: Directory containing .tsp files
        """
        self.instances = instances if instances is not None else []
        
        if data_dir is not None:
            self._load_from_dir(data_dir)
    
    def _load_from_dir(self, data_dir):
        """Load all .tsp files from directory."""
        from problems.tsp.utils import load_tsp_file
        
        if not os.path.exists(data_dir):
            return
        
        for fname in sorted(os.listdir(data_dir)):
            if fname.endswith('.tsp'):
                path = os.path.join(data_dir, fname)
                try:
                    instance = load_tsp_file(path)
                    self.instances.append(instance)
                except Exception as e:
                    print(f"Warning: Failed to load {fname}: {e}")
    
    def add_instance(self, instance):
        """Add a single instance."""
        self.instances.append(instance)
    
    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx):
        return self.instances[idx]
    
    def sample(self):
        """Sample a random instance."""
        idx = np.random.randint(len(self.instances))
        return idx, self.instances[idx]
    
    @staticmethod
    def generate_random(num_instances, min_nodes=10, max_nodes=50, seed=None):
        """Generate random TSP instances."""
        from problems.tsp.utils import generate_random_tsp
        
        if seed is not None:
            np.random.seed(seed)
        
        instances = []
        for i in range(num_instances):
            n = np.random.randint(min_nodes, max_nodes + 1)
            inst = generate_random_tsp(n, seed=None)
            inst['name'] = f'random_{i}_{n}'
            instances.append(inst)
        
        return TSPInstanceDataset(instances)


class ConditionalTSPEnv:
    """
    Conditional TSP environment that can handle multiple instances.
    
    This environment wraps multiple TSP instances and provides methods
    to sample and interact with them for conditional GFlowNet training.
    """
    
    def __init__(self, instances, alpha=1.0, beta=0.1):
        """
        Args:
            instances: List of instance dicts or TSPInstanceDataset
            alpha: Reward coefficient for tour length
            beta: Scaling factor
        """
        if isinstance(instances, TSPInstanceDataset):
            self.instances = instances.instances
        else:
            self.instances = instances
        
        self.alpha = alpha
        self.beta = beta
        
        self.num_instances = len(self.instances)
        
        # Create individual environments for each instance
        self.envs = []
        for inst in self.instances:
            env = TSPEnv(inst, alpha=alpha, beta=beta)
            self.envs.append(env)
        
        # Current instance index
        self._current_idx = 0
    
    def get_instance(self, idx):
        """Get instance by index."""
        return self.instances[idx]
    
    def get_env(self, idx):
        """Get environment for instance by index."""
        return self.envs[idx]
    
    def sample_instance(self):
        """Sample a random instance, returns (idx, instance)."""
        idx = np.random.randint(self.num_instances)
        return idx, self.instances[idx]
    
    def reset(self, instance_idx=None):
        """
        Reset environment for a specific instance.
        
        Args:
            instance_idx: Index of instance to use. If None, samples randomly.
            
        Returns:
            state: Initial state
            instance_idx: Index of the instance being used
            instance: The instance dict
        """
        if instance_idx is None:
            instance_idx = np.random.randint(self.num_instances)
        
        self._current_idx = instance_idx
        state = self.envs[instance_idx].reset()
        
        return state, instance_idx, self.instances[instance_idx]
    
    def step(self, state, action, instance_idx=None):
        """Take a step in the environment."""
        if instance_idx is None:
            instance_idx = self._current_idx
        
        return self.envs[instance_idx].step(state, action)
    
    def allowed_actions(self, state, instance_idx=None):
        """Get allowed actions mask for state."""
        if instance_idx is None:
            instance_idx = self._current_idx
        
        return self.envs[instance_idx].allowed_actions(state)
    
    def is_terminal(self, state, instance_idx=None):
        """Check if state is terminal."""
        if instance_idx is None:
            instance_idx = self._current_idx
        
        return self.envs[instance_idx].is_terminal(state)
    
    def reward(self, state, instance_idx=None):
        """Compute reward for terminal state."""
        if instance_idx is None:
            instance_idx = self._current_idx
        
        return self.envs[instance_idx].reward(state)
    
    def get_tour_length(self, state, instance_idx=None):
        """Get total tour length."""
        if instance_idx is None:
            instance_idx = self._current_idx
        
        return self.envs[instance_idx].get_tour_length(state)
    
    def get_tour_from_state(self, state, instance_idx=None):
        """Convert state to tour order."""
        if instance_idx is None:
            instance_idx = self._current_idx
        
        return self.envs[instance_idx].get_tour_from_state(state)
