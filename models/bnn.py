import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

class BayesianQNet(PyroModule):
    output_type = "mean_std"

    def __init__(self, size_state, n_actions, layers=[64, 64]):
        super().__init__()
        self.size_state = size_state
        self.n_actions = n_actions
        sizes = [size_state + 1] + layers  # +1 for beta

        # Define layers with priors
        layer_modules = []
        for i in range(len(sizes) - 1):
            layer_modules.append(PyroModule[nn.Linear](sizes[i], sizes[i+1]))
            layer_modules[-1].weight = PyroSample(dist.Normal(0., 1.).expand([sizes[i+1], sizes[i]]).to_event(2))
            layer_modules[-1].bias = PyroSample(dist.Normal(0., 1.).expand([sizes[i+1]]).to_event(1))
            layer_modules.append(nn.ReLU())
        
        self.hidden = nn.Sequential(*layer_modules)

        # Output layer for reward and cost
        self.predict = PyroModule[nn.Linear](sizes[-1], 2 * n_actions)
        self.predict.weight = PyroSample(dist.Normal(0., 1.).expand([2 * n_actions, sizes[-1]]).to_event(2))
        self.predict.bias = PyroSample(dist.Normal(0., 1.).expand([2 * n_actions]).to_event(1))

    def forward(self, state, beta, n_samples=10):
        # state: [B, state_dim], beta: [B, 1]
        x = torch.cat([state, beta], dim=-1)
        
        # Sample from the posterior distribution of weights
        sampled_q_values = []
        for _ in range(n_samples):
            h = self.hidden(x)
            out = self.predict(h)
            sampled_q_values.append(out)

        # Stack and compute mean and std
        sampled_q_values = torch.stack(sampled_q_values) # [n_samples, B, 2*n_actions]
        
        mean_q = torch.mean(sampled_q_values, dim=0)
        std_q = torch.std(sampled_q_values, dim=0)

        q_r_mean = mean_q[:, :self.n_actions]
        q_c_mean = mean_q[:, self.n_actions:]
        
        q_r_std = std_q[:, :self.n_actions]
        q_c_std = std_q[:, self.n_actions:]

        return (q_r_mean, q_r_std), (q_c_mean, q_c_std)


def test():
    # ==== Configuration ====
    batch_size = 4
    state_dim = 5
    n_actions = 3
    hidden_sizes = [32, 32]

    # ==== Create model ====
    net = BayesianQNet(size_state=state_dim, layers=hidden_sizes, n_actions=n_actions)
    print(net)

    # ==== Create dummy input ====
    states = torch.randn(batch_size, state_dim)
    betas = torch.rand(batch_size, 1)
    print("State shape:", states.shape)
    print("Beta shape:", betas.shape)

    # ==== Forward pass ====
    (q_r_mean, q_r_std), (q_c_mean, q_c_std) = net(states, betas)

    print("Q_r_mean shape:", q_r_mean.shape)
    print("Q_r_std shape:", q_r_std.shape)
    print("Q_c_mean shape:", q_c_mean.shape)
    print("Q_c_std shape:", q_c_std.shape)

if __name__ == "__main__":
    test()
