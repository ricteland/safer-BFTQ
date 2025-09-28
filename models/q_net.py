import torch
import torch.nn as nn
import torch.nn.functional as F

class BudgetedQNet(nn.Module):
    def __init__(self, action_dim, state_dim, hidden_sizes):
        super().__init__()

        input_dim = state_dim + 1
        layers = []
        last_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        self.shared = nn.Sequential(*layers)
        self.q_reward = nn.Linear(last_dim, action_dim)
        self.q_cost = nn.Linear(last_dim, action_dim)

    def forward(self, x, state, budget):
        x = torch.cat([state, budget], dim = 1)

        x= self.shared(x)

        q_r = self.q_reward(x)
        q_c = self.q_cost(x)

        return q_r, q_c