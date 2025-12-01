# problems/knapsack/env.py

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from envs.base_env import BaseEnv


class KnapsackEnv(BaseEnv):
    """
    Knapsack environment for GFlowNet.
    
    State: binary array of length N indicating which items are selected.
            -1 means item not yet decided, 0 means not selected, 1 means selected.
    Action: index of item to decide (0 to 2*N-1)
            action < N means select item (action), action >= N means skip item (action - N)
    """
    
    def __init__(self, instance, alpha=0.1, beta=0.01, gamma=1.0):
        """
        instance should contain:
            - 'profits': array of item profits/values
            - 'weights': array of item weights
            - 'capacity': knapsack capacity
        """
        self.profits = np.array(instance['profits'], dtype=np.float32)
        self.values = self.profits # Alias for reward function
        self.weights = np.array(instance['weights'], dtype=np.float32)
        self.capacity = instance['capacity']
        self.N = len(self.profits)
        
        # Hyperparameters for reward
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Precompute for reward normalization
        self.max_profit = np.sum(self.profits)
        
        super().__init__(instance)
    
    def reset(self):
        # -1 means undecided, 0 means not selected, 1 means selected
        self.state = -1 * np.ones(self.N, dtype=np.int32)
        return self.state.copy()
    
    def allowed_actions(self, state):
        """
        Actions: 0 to N-1 = select item i, N to 2N-1 = skip item i
        Only undecided items can be acted upon.
        Selecting an item is only valid if it fits in remaining capacity.
        """
        mask = np.zeros(2 * self.N, dtype=np.float32)
        
        # Current weight used
        selected = state == 1
        current_weight = np.sum(self.weights[selected])
        remaining_capacity = self.capacity - current_weight
        
        # For each undecided item
        undecided = state == -1
        
        for i in np.where(undecided)[0]:
            # Can always skip (action = N + i)
            mask[self.N + i] = 1.0
            # Can select only if it fits (action = i)
            if self.weights[i] <= remaining_capacity:
                mask[i] = 1.0
        
        return mask
    
    def step(self, state, action):
        next_state = state.copy()
        
        if action < self.N:
            # Select item
            next_state[action] = 1
        else:
            # Skip item
            next_state[action - self.N] = 0
        
        done = self.is_terminal(next_state)
        reward = self.reward(next_state) if done else 0.0
        return next_state, reward, done
    
    def step_batch(self, states, actions):
        """
        Vectorized step for batch of states.
        """
        B, N = states.shape
        next_states = states.copy()
        
        select_mask = actions < N
        skip_mask = ~select_mask
        
        # Select updates
        if select_mask.any():
            idx = np.where(select_mask)[0]
            next_states[idx, actions[idx]] = 1
            
        # Skip updates
        if skip_mask.any():
            idx = np.where(skip_mask)[0]
            next_states[idx, actions[idx] - N] = 0
            
        # Done check (fully decided)
        dones = (next_states != -1).all(axis=1)
        
        # Rewards
        rewards = np.zeros(B, dtype=np.float32)
        if dones.any():
            done_states = next_states[dones]
            rewards[dones] = self.reward_batch(done_states)
            
        return next_states, rewards, dones
    
    def is_terminal(self, state):
        # Terminal if all items are decided
        if np.all(state != -1):
            return True
        # Also terminal if no valid actions remain
        mask = self.allowed_actions(state)
        return mask.sum() == 0
    
    def reward(self, state):
        """
        state: 1D numpy array or torch tensor of {0,1}, shape (n,)
        returns: scalar float log-reward
        """
        # Convert to numpy if needed
        if "torch" in str(type(state)):
            state = state.detach().cpu().numpy()
        state = state.astype(float)

        total_weight = float(np.dot(state, self.weights))
        total_value = float(np.dot(state, self.values))

        # How far from capacity
        slack = max(0.0, self.capacity - total_weight)          # underfilled
        overflow = max(0.0, total_weight - self.capacity)       # overweight

        # Log-reward: balance value, slack, and overflow
        logR = (
            self.alpha * total_value
            - self.beta * slack
            - self.gamma * overflow
        )

        # Scale by N to align magnitude with TB loss (sum over N steps)
        logR = logR * self.N

        return float(logR)
    
    def reward_batch(self, states):
        """Vectorized reward for batch of states (B, N)."""
        B, N = states.shape
        
        # Cast to float for dot product
        states_float = states.astype(np.float32)
        
        total_weight = np.dot(states_float, self.weights)
        total_value = np.dot(states_float, self.values)
        
        # How far from capacity
        slack = np.maximum(0.0, self.capacity - total_weight)
        overflow = np.maximum(0.0, total_weight - self.capacity)
        
        # Log-reward
        logR = (
            self.alpha * total_value
            - self.beta * slack
            - self.gamma * overflow
        )
        
        # Scale by N to align magnitude with TB loss (sum over N steps)
        logR = logR * N
        
        return logR
    
    def encode_state(self, state):
        """Encode state as feature vector."""
        # One-hot encode: undecided, not selected, selected
        encoding = np.zeros((self.N, 3), dtype=np.float32)
        encoding[state == -1, 0] = 1.0  # undecided
        encoding[state == 0, 1] = 1.0   # not selected
        encoding[state == 1, 2] = 1.0   # selected
        return encoding.flatten()
    
    def get_profit(self, state):
        """Get total profit of current selection."""
        selected = state == 1
        return np.sum(self.profits[selected])
    
    def get_weight(self, state):
        """Get total weight of current selection."""
        selected = state == 1
        return np.sum(self.weights[selected])


class KnapsackInstanceDataset:
    """
    Dataset of knapsack instances for conditional training.
    Loads instances from a data directory.
    """
    
    def __init__(self, data_dir, instance_names=None):
        """
        Args:
            data_dir: Directory containing instance folders
            instance_names: List of instance names to load. If None, loads all.
        """
        from problems.knapsack.utils import load_knapsack_instance, list_knapsack_instances
        
        self.data_dir = data_dir
        
        if instance_names is None:
            instance_names = list_knapsack_instances(data_dir)
        
        self.instances = []
        for name in instance_names:
            try:
                inst = load_knapsack_instance(data_dir, name)
                self.instances.append(inst)
            except Exception as e:
                print(f"Warning: Failed to load instance {name}: {e}")
        
        if not self.instances:
            raise ValueError(f"No valid instances found in {data_dir}")
        
        print(f"Loaded {len(self.instances)} knapsack instances")
    
    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx):
        return self.instances[idx]
    
    def sample(self):
        """Sample a random instance."""
        idx = np.random.randint(len(self.instances))
        return idx, self.instances[idx]


class ConditionalKnapsackEnv:
    """
    Conditional Knapsack environment that can handle multiple instances.
    
    This environment wraps multiple knapsack instances and provides methods
    to sample and interact with them for conditional GFlowNet training.
    """
    
    def __init__(self, instances, alpha=0.1, beta=0.01, gamma=1.0):
        """
        Args:
            instances: List of instance dicts or KnapsackInstanceDataset
            alpha: Reward coefficient for profit
            beta: Reward coefficient for slack penalty
            gamma: Reward coefficient for overflow penalty
        """
        if isinstance(instances, KnapsackInstanceDataset):
            self.instances = instances.instances
        else:
            self.instances = instances
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.num_instances = len(self.instances)
        
        # Create individual environments for each instance
        self.envs = []
        for inst in self.instances:
            env = KnapsackEnv(inst, alpha=alpha, beta=beta, gamma=gamma)
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
        """
        Take a step in the environment.
        
        Args:
            state: Current state
            action: Action to take
            instance_idx: Instance index (uses current if None)
            
        Returns:
            next_state, reward, done
        """
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
    
    def get_profit(self, state, instance_idx=None):
        """Get total profit of current selection."""
        if instance_idx is None:
            instance_idx = self._current_idx
        
        return self.envs[instance_idx].get_profit(state)
    
    def get_weight(self, state, instance_idx=None):
        """Get total weight of current selection."""
        if instance_idx is None:
            instance_idx = self._current_idx
        
        return self.envs[instance_idx].get_weight(state)
