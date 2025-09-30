import torch
import torch.nn as nn
import torch.nn.functional as F

class BudgetedQNet(nn.Module):
    output_type = "q_values"

    def __init__(self, size_state, n_actions, layers = [64, 64]):
        super(BudgetedQNet, self).__init__()
        sizes = [size_state + 1] + layers  # +1 for beta
        self.size_state = size_state
        self.size_action = n_actions

        # hidden layers
        net_layers = []
        for i in range(len(sizes) - 1):
            net_layers.append(nn.Linear(sizes[i], sizes[i+1]))
            net_layers.append(nn.ReLU())
        self.hidden = nn.Sequential(*net_layers)

        # output layer: reward+cost
        self.predict = nn.Linear(sizes[-1], 2 * n_actions)

    def forward(self, state, beta):
        # if state.dim() != beta.dim():
        #     print("DEBUG SHAPES in forward():")
        #     print(f"  state.shape = {state.shape}")
        #     print(f"  beta.shape  = {beta.shape}")
        # state: [B, state_dim], beta: [B, 1]
        x = torch.cat([state, beta], dim=-1)
        h = self.hidden(x)
        out = self.predict(h)          # [B, 2*n_actions]
        q_r = out[:, :self.size_action]
        q_c = out[:, self.size_action:]
        return q_r, q_c

def test():
    # ==== Configuration ====
    batch_size = 4
    state_dim = 5       # e.g. 5 features
    n_actions = 3       # number of actions
    hidden_sizes = [32, 32]   # two hidden layers with 32 units

    # ==== Create model ====
    net = BudgetedQNet(size_state=state_dim, layers=hidden_sizes, n_actions=n_actions)
    print(net)

    # ==== Create dummy input ====
    states = torch.randn(batch_size, state_dim)  # [B, state_dim]
    betas = torch.rand(batch_size, 1)            # [B, 1]
    print("State shape:", states.shape)
    print("Beta shape:", betas.shape)

    # ==== Forward pass ====
    q_r, q_c = net(states, betas)

    print("Q_r shape:", q_r.shape)   # [B, n_actions]
    print("Q_r:", q_r)
    print("Q_c shape:", q_c.shape)   # [B, n_actions]
    print("Q_c:", q_c)