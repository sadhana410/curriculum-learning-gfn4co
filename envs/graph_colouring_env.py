import numpy as np
from envs.base_env import BaseEnv

class GraphColoringEnv(BaseEnv):
    def __init__(self, instance, num_colors=3):
        self.adj = instance["adj"]
        self.N = self.adj.shape[0]
        self.K = num_colors
        super().__init__(instance)

    def reset(self):
        self.state = -1 * np.ones(self.N, dtype=int)
        return self.state.copy()

    def allowed_actions(self, state):
        mask = np.zeros(self.N * self.K, dtype=np.float32)
        for node in range(self.N):
            if state[node] != -1:
                continue
            for color in range(self.K):
                if not self._conflict(state, node, color):
                    mask[node * self.K + color] = 1.0
        return mask

    def step(self, state, action):
        node = action // self.K
        color = action % self.K
        next_state = state.copy()
        next_state[node] = color
        done = self.is_terminal(next_state)
        reward = self.reward(next_state) if done else 0.0
        return next_state, reward, done

    def is_terminal(self, state):
        return np.all(state != -1)

    def reward(self, state):
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if self.adj[i, j] == 1 and state[i] == state[j]:
                    return 0.0
        num_colors = len(set(state.tolist()))
        return float(np.exp(-num_colors))

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
