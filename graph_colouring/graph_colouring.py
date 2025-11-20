import numpy as np
from env import BaseEnv 

class GraphColoringEnv(BaseEnv):
    """
        State: vector of size N
        -1 = uncolored
        {0..K-1} = color assignment
    Actions: encoded as a single integer = node * K + color
    """

    def __init__(self, instance, num_colors=3):
        """
        instance must contain:
            instance["adj"] = adjacency matrix (NxN numpy array)
        """
        self.adj = instance["adj"]
        self.N = self.adj.shape[0]
        self.K = num_colors
        super().__init__(instance)


    def reset(self):
        self.state = -1 * np.ones(self.N, dtype=int)
        return self.state.copy()

    def allowed_actions(self, state):
        """
        1 = legal (node not colored & color does not conflict)
        0 = illegal
        """
        mask = np.zeros(self.N * self.K, dtype=np.float32)

        for node in range(self.N):
            if state[node] != -1:
                continue  # already colored

            for color in range(self.K):
                if not self._conflict(state, node, color):
                    action_id = node * self.K + color
                    mask[action_id] = 1.0

        return mask

    def step(self, state, action):
        """
        Returns (next_state, reward, done)
        """
        node = action // self.K
        color = action % self.K

        next_state = state.copy()
        next_state[node] = color

        done = self.is_terminal(next_state)
        reward = 0.0

        if done:
            reward = self.reward(next_state)

        return next_state, reward, done

    def is_terminal(self, state):
        return np.all(state != -1)

    def reward(self, terminal_state):
        """Reward = exp(-num_colors_used) if valid, else 0."""
        # check invalid coloring
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if self.adj[i, j] == 1 and terminal_state[i] == terminal_state[j]:
                    return 0.0

        # unique colors used
        num_colors_used = len(set(terminal_state.tolist()))
        return float(np.exp(-num_colors_used))

    def encode_state(self, state):
        """
        Flattened one-hot encoding of length N*K.
        -1 => all zeros
        """
        one_hot = np.zeros((self.N, self.K), dtype=np.float32)
        for i in range(self.N):
            if state[i] != -1:
                one_hot[i, state[i]] = 1.0
        return one_hot.flatten()

    def _conflict(self, state, node, color):
        #Check for conflicts
        for nbr in range(self.N):
            if self.adj[node, nbr] == 1 and state[nbr] == color:
                return True
        return False
