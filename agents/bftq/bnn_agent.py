import torch
import numpy as np
from agents.bftq.bnn_bftq import BNNBFTQ
from utils.replay_buffer import ReplayBuffer
from models.bnn import BayesianQNet
from agents.bftq.risk_averse_policies import PessimisticPytorchBudgetedFittedPolicy
from agents.bftq.policies import RandomBudgetedPolicy, EpsilonGreedyBudgetedPolicy

class BNNBFTQAgent:
    def __init__(self, state_dim, n_actions, config, network=BayesianQNet, device="cpu", logger=None, tb_logger=None):
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

        # === Buffer ===
        self.replay_buffer = ReplayBuffer(capacity=config.get("buffer_size", 10000))

        # === Training logic lives in BNNBFTQ ===
        self.bftq = BNNBFTQ(self.q_net, self.target_net, self.replay_buffer, config, device=device, logger=self.logger, tb_logger=self.tb_logger)

        # === Policies ===
        greedy_policy = PessimisticPytorchBudgetedFittedPolicy(
            network=self.q_net, 
            betas_for_discretisation=np.linspace(0, 1, 100), 
            device=device, 
            hull_options=config.get("hull_options", {}),
            k=config.get("k", 1.96) # Add k for pessimism
        )
        random_policy = RandomBudgetedPolicy(n_actions=n_actions)
        self.policy = EpsilonGreedyBudgetedPolicy(greedy_policy, random_policy, config=config["exploration"])

    def act(self, state, beta):
        # The BNN-based policy expects a different forward pass
        # However, the policy itself handles the sampling, so we can just pass the state
        state = torch.tensor(state, dtype=torch.float32, device=self.device).flatten().unsqueeze(0)
        action, new_beta, q_r, q_c = self.policy.execute(state, beta)
        old_beta = beta
        return action, new_beta, q_r, q_c, old_beta

    def push_transition(self, *args):
        self.replay_buffer.push(*args)

    def update(self):
        # The BFTQ update logic needs to be adapted for BNNs
        # For now, we assume the existing BFTQ class can handle it if the network output is mean+std
        # This might need to be changed to a custom BNNBFTQ class if the loss is different
        return self.bftq.update()

    def set_training_mode(self, mode):
        self.policy.pi_greedy.training_mode = mode

    def save_model(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load_model(self, path):
        self.q_net.load_state_dict(torch.load(path))
