class BaseAgent:
    def __init__(self):
        pass
    def act(self, state, budget):
        raise NotImplementedError
    def update(self, batch):
        raise NotImplementedError