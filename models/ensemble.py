import torch
import torch.nn as nn
import torch.nn.functional as F
from models.q_net import BudgetedQNet

class EnsembleQNet(nn.Module):
    output_type = "distribution"

    def __init__(self, size_state, n_actions, layers=[64, 64], n_models=5):
        super(EnsembleQNet, self).__init__()
        self.n_models = n_models
        self.models = nn.ModuleList([BudgetedQNet(size_state, n_actions, layers) for _ in range(n_models)])

    def forward(self, state, beta):
        q_r_samples = []
        q_c_samples = []
        for model in self.models:
            q_r, q_c = model(state, beta)
            q_r_samples.append(q_r)
            q_c_samples.append(q_c)

        q_r_samples = torch.stack(q_r_samples)
        q_c_samples = torch.stack(q_c_samples)

        mean_q_r = q_r_samples.mean(dim=0)
        std_q_r = q_r_samples.std(dim=0)
        mean_q_c = q_c_samples.mean(dim=0)
        std_q_c = q_c_samples.std(dim=0)

        return (mean_q_r, std_q_r), (mean_q_c, std_q_c)

    def predict_dist(self, state, beta):
        return self.forward(state, beta)

def test():
    # ==== Configuration ====#
    batch_size = 4
    state_dim = 5       # e.g. 5 features
    n_actions = 3       # number of actions
    hidden_sizes = [32, 32]   # two hidden layers with 32 units
    n_models = 5

    # ==== Create model ====#
    net = EnsembleQNet(size_state=state_dim, layers=hidden_sizes, n_actions=n_actions, n_models=n_models)
    print(net)

    # ==== Create dummy input ====#
    states = torch.randn(batch_size, state_dim)  # [B, state_dim]
    betas = torch.rand(batch_size, 1)            # [B, 1]
    print("State shape:", states.shape)
    print("Beta shape:", betas.shape)

    # ==== Forward pass ====#
    (mean_q_r, std_q_r), (mean_q_c, std_q_c) = net.predict_dist(states, betas)

    print("Mean Q_r shape:", mean_q_r.shape)   # [B, n_actions]
    print("Mean Q_r:", mean_q_r)
    print("Std Q_r shape:", std_q_r.shape)   # [B, n_actions]
    print("Std Q_r:", std_q_r)
    print("Mean Q_c shape:", mean_q_c.shape)   # [B, n_actions]
    print("Mean Q_c:", mean_q_c)
    print("Std Q_c shape:", std_q_c.shape)   # [B, n_actions]
    print("Std Q_c:", std_q_c)

if __name__ == '__main__':
    test()
