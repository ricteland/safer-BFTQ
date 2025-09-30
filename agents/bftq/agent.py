import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sympy.physics.units import action

from utils.replay_buffer import ReplayBuffer
from agents.bftq.policies import (
    PytorchBudgetedFittedPolicy,
    RandomBudgetedPolicy,
    EpsilonGreedyBudgetedPolicy
)
from models.q_net import BudgetedQNet


class BFTQAgent:
    def __init__(self, state_dim, n_actions, config, network, device="cpu"):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.device = device

        # === Hyperparams ===
        self.gamma = config.get("gamma", 0.99)
        self.batch_size = config.get("batch_size", 32)
        self.buffer_size = config.get("buffer_size", 10000)
        self.learning_rate = config.get("learning_rate", 1e-3)
        self.target_update = config.get("target_update", 100)

        # === Network ===
        self.q_net = network(size_state=state_dim, n_actions=n_actions, layers=config.get("layers", [64, 64])).to(device)
        self.target_net = network(size_state=state_dim, n_actions=n_actions, layers=config.get("layers", [64, 64])).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)

        # === ReplayBuffer ===
        self.replay_buffer = ReplayBuffer(capacity=self.buffer_size)

        # === Polcicies ===

        greedy_policy = PytorchBudgetedFittedPolicy(
            network=self.q_net,
            betas_for_discretisation=np.linspace(0, 1, 100),
            device=device,
            hull_options=config.get("hull_options", dict(library="scipy", decimals=2, remove_duplicates=True)),
            clamp_qc=config.get("clamp_qc", None)
        )
        random_policy = RandomBudgetedPolicy(n_actions=n_actions)
        self.policy = EpsilonGreedyBudgetedPolicy(greedy_policy, random_policy, config=['exploration'])

        self.steps = 0

    def act(self, state, beta):
        action, new_beta = self.policy.execute(state, beta)
        return action, new_beta

    def push_transition(self, state, action, reward, cost, beta, next_state, done):
        self.replay_buffer.push(state, action, reward, cost, beta, next_state, done)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        transitions = self.replay_buffer.sample(self.batch_size)
        batch = list(zip(*transitions))
        state, action, reward, cost, beta, next_state, done = batch

        # === Convert to tensors ===
        state = torch.stack(state).to(self.device)
        next_state = torch.stack(next_state).to(self.device)
        action = torch.tensor(action, device=self.device).long()
        reward = torch.tensor(reward, device=self.device).float()
        cost = torch.tensor(cost, device=self.device).float()
        beta = torch.tensor(beta, device=self.device).float().unsqueeze(1)
        done = torch.tensor(done, device=self.device).float()

        # === Current Q_r and Q_c ===
        q_r, q_c = self.q_net(state, beta)

        if getattr(self.q_net, "output_type", "q_values") == "q_values":
            q_r = q_r.gather(1, action.unsqueeze(1)).squeeze()
            q_c = q_c.gather(1, action.unsqueeze(1)).squeeze()
        elif getattr(self.q_net, "output_type", "q_values") == "mean_std":
            q_r_mean = q_r[:, :self.n_actions]
            q_r_std = q_r[:, self.n_actions:]
            q_c_mean = q_c[:, :self.n_actions]
            q_c_std = q_c[:, self.n_actions:]

            # For now, just use means for Bellman update
            q_r = q_r_mean.gather(1, action.unsqueeze(1)).squeeze()
            q_c = q_c_mean.gather(1, action.unsqueeze(1)).squeeze()

        else:
            raise ValueError(f"Unknown output type: {getattr(self.q_net, 'output_type', 'q_values')}")


        # === Target Q_r and Q_c ===
        with torch.no_grad():
            next_q_values = self.target_net(next_state, beta)

            if getattr(self.q_net, "output_type", "q_values") == "q_values":
                next_q_r = next_q_values[:, :self.n_actions].max(1)[0]
                next_q_c = next_q_values[:, self.n_actions:].max(1)[0]

            elif getattr(self.q_net, "output_type", "q_values") == "mean_std":
                #TODO implement this
                pass
            else:
                raise ValueError(f"Unknown output type: {getattr(self.q_net, 'output_type', 'q_values')}")

            target_q_r = reward + self.gamma * (1 - done) * next_q_r
            target_q_c = cost + self.gamma * (1 - done) * next_q_c

        # === Loss ==
        loss_r = nn.MSELoss()(q_r, target_q_r)
        loss_c = nn.MSELoss()(q_c, target_q_c)
        loss = loss_r + loss_c

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # === Update target net ===
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()
