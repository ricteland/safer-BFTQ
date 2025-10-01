import torch
import torch.nn as nn
import torch.nn.functional as F

class MCDropoutQNet(nn.Module):
    output_type = "distribution"

    def __init__(self, size_state, n_actions, layers=[64, 64], dropout_p=0.5, n_samples=10):
        super(MCDropoutQNet, self).__init__()
        sizes = [size_state + 1] + layers  # +1 for beta
        self.size_state = size_state
        self.size_action = n_actions
        self.dropout_p = dropout_p
        self.n_samples = n_samples

        # hidden layers
        net_layers = []
        for i in range(len(sizes) - 1):
            net_layers.append(nn.Linear(sizes[i], sizes[i+1]))
            net_layers.append(nn.ReLU())
            net_layers.append(nn.Dropout(self.dropout_p))
        self.hidden = nn.Sequential(*net_layers)

        # output layer: reward+cost
        self.predict = nn.Linear(sizes[-1], 2 * n_actions)

    def forward(self, state, beta):
        # state: [B, state_dim], beta: [B, 1]
        x = torch.cat([state, beta], dim=-1)
        h = self.hidden(x)
        out = self.predict(h)          # [B, 2*n_actions]
        q_r = out[:, :self.size_action]
        q_c = out[:, self.size_action:]
        return q_r, q_c

    def predict_dist(self, state, beta):
        # Enable dropout during inference
        self.train()
        q_r_samples = []
        q_c_samples = []
        for _ in range(self.n_samples):
            q_r, q_c = self.forward(state, beta)
            q_r_samples.append(q_r)
            q_c_samples.append(q_c)

        q_r_samples = torch.stack(q_r_samples)
        q_c_samples = torch.stack(q_c_samples)

        mean_q_r = q_r_samples.mean(dim=0)
        std_q_r = q_r_samples.std(dim=0)
        mean_q_c = q_c_samples.mean(dim=0)
        std_q_c = q_c_samples.std(dim=0)

        return (mean_q_r, std_q_r), (mean_q_c, std_q_c)


def test():
    # ==== Configuration ====#
    batch_size = 4
    state_dim = 5       # e.g. 5 features
    n_actions = 3       # number of actions
    hidden_sizes = [32, 32]   # two hidden layers with 32 units

    # ==== Create model ====#
    net = MCDropoutQNet(size_state=state_dim, layers=hidden_sizes, n_actions=n_actions)
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
