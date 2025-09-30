import random
from collections import deque, namedtuple
import torch

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'cost', 'beta', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def test():
    # ==== Config ====
    capacity = 5
    batch_size = 3
    state_dim = 4
    n_actions = 2

    # ==== Create buffer ====
    buffer = ReplayBuffer(capacity=capacity)

    # ==== Push some fake transitions ====
    for i in range(7):  # push more than capacity to test overwrite
        s = torch.randn(state_dim)
        a = torch.randint(0, n_actions, (1,)).item()
        r = float(i)  # reward just i
        c = float(-i)  # cost just -i
        beta = torch.rand(1).item()
        s_next = torch.randn(state_dim)
        done = (i % 2 == 0)  # alternate True/False

        buffer.push(s, a, r, c, beta, s_next, done)
        print(f"Pushed transition {i}")

    print(f"\nBuffer length (should be {capacity}):", len(buffer))

    # ==== Sample batch ====
    batch = buffer.sample(batch_size)
    print("\nSampled batch of size", batch_size)

    # Namedtuple unpack
    for i, transition in enumerate(batch):
        print(f"\nSample {i}:")
        print(" state:", transition.state)
        print(" action:", transition.action)
        print(" reward:", transition.reward)
        print(" cost:", transition.cost)
        print(" beta:", transition.beta)
        print(" next_state:", transition.next_state)
        print(" done:", transition.done)