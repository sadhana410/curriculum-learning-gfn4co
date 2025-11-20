class BaseEnv:
    def __init__(self, instance):
        self.instance = instance
        self.reset()

    def reset(self):
        raise NotImplementedError

    def allowed_actions(self, state):
        raise NotImplementedError

    def step(self, state, action):
        raise NotImplementedError

    def is_terminal(self, state):
        raise NotImplementedError

    def reward(self, terminal_state):
        raise NotImplementedError

    def encode_state(self, state):
        raise NotImplementedError
