class BaseEnv:
    def __init__(self, instance):
        #task specific instances (colouring/knapsack/tsp)
        self.instance = instance
        self.reset()

    def reset(self):
        #returns initial state s0
        raise NotImplementedError

    def allowed_actions(self, state):
        #returns list or mask of allowed actions from this state
        raise NotImplementedError

    def step(self, state, action):
        #return next_state, reward, done (boolean)
        raise NotImplementedError

    def is_terminal(self, state):
        #return True if state is terminal
        raise NotImplementedError

    def reward(self, terminal_state):
        #return reward R(x) for terminal state x
        raise NotImplementedError

    def encode_state(self, state):
        #convert to tensor / features for model input
        raise NotImplementedError
