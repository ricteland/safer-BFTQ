# agent.py
import torch
import numpy as np
from agents.bftq.bftq import BFTQ
from utils.replay_buffer import ReplayBuffer
from models.q_net import BudgetedQNet
from agents.bftq.policies import (
    PytorchBudgetedFittedPolicy,
    RandomBudgetedPolicy,
    EpsilonGreedyBudgetedPolicy
)

class BFTQAgent:
    def __init__(self, state_dim, n_actions, config, network=BudgetedQNet, device="cpu", logger=None, tb_logger=None):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.device = device
        self.logger = logger
        self.tb_logger = tb_logger

        # === Networks ===
        self.q_net = network(size_state=state_dim, n_actions=n_actions, layers=config.get("layers", [64, 64])).to(device)
        self.target_net = network(size_state=state_dim, n_actions=n_actions, layers=config.get("layers", [64, 64])).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        # === Buffer & optimizer ===
        self.replay_buffer = ReplayBuffer(capacity=config.get("buffer_size", 10000))
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=config.get("learning_rate", 1e-3))

        # === Training logic lives in BFTQ ===
        self.bftq = BFTQ(self.q_net, self.target_net, self.optimizer, self.replay_buffer, config, device=device, logger=self.logger, tb_logger=self.tb_logger)

        # === Policies ===
        greedy_policy = PytorchBudgetedFittedPolicy(self.q_net, np.linspace(0, 1, 100), device, config.get("hull_options", {}))
        random_policy = RandomBudgetedPolicy(n_actions=n_actions)
        self.policy = EpsilonGreedyBudgetedPolicy(greedy_policy, random_policy, config=config["exploration"])

    def act(self, state, beta):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).flatten().unsqueeze(0)
        # print("DEBUG agent.act: state type =", type(state),
        #       "shape =", state.shape if hasattr(state, "shape") else "no shape")
        action, new_beta, q_r, q_c = self.policy.execute(state, beta)
        old_beta = beta
        return action, new_beta, q_r, q_c, old_beta

    def push_transition(self, *args):
        self.replay_buffer.push(*args)

    def update(self):
        return self.bftq.update()

    def save_model(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load_model(self, path):
        self.q_net.load_state_dict(torch.load(path))
