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
        
        # --- Layer definitions ---
        self.fc1 = PyroModule[nn.Linear](size_state + 1, layers[0])
        self.fc1.weight = PyroSample(dist.Normal(0., 1.).expand([layers[0], size_state + 1]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0., 1.).expand([layers[0]]).to_event(1))

        self.fc2 = PyroModule[nn.Linear](layers[0], layers[1])
        self.fc2.weight = PyroSample(dist.Normal(0., 1.).expand([layers[1], layers[0]]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(0., 1.).expand([layers[1]]).to_event(1))

        self.predict = PyroModule[nn.Linear](layers[1], 2 * n_actions)
        self.predict.weight = PyroSample(dist.Normal(0., 1.).expand([2 * n_actions, layers[1]]).to_event(2))
        self.predict.bias = PyroSample(dist.Normal(0., 1.).expand([2 * n_actions]).to_event(1))

        self.relu = nn.ReLU()

    def forward(self, state, beta, target_q_r=None, target_q_c=None, action=None, n_samples=10):
        x = torch.cat([state, beta], dim=-1)

        # The guide is sampled once per forward pass in training mode
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        out = self.predict(h)

        if target_q_r is not None:
            # --- Training mode ---
            q_r_pred = out[:, :self.n_actions]
            q_c_pred = out[:, self.n_actions:]

            q_r_pred_action = q_r_pred.gather(1, action.unsqueeze(1)).squeeze(-1)
            q_c_pred_action = q_c_pred.gather(1, action.unsqueeze(1)).squeeze(-1)

            with pyro.plate("data", size=state.shape[0]):
                pyro.sample("obs_r", dist.Normal(q_r_pred_action, 0.1), obs=target_q_r)
                pyro.sample("obs_c", dist.Normal(q_c_pred_action, 0.1), obs=target_q_c)

        else:
            # --- Inference mode ---
            # To get uncertainty, we need to sample from the guide, which is done by calling the model multiple times.
            # This is handled by the pyro.infer.Predictive class.
            # However, to keep the agent code simple, we do it manually here.
            
            sampled_q_values = []
            for _ in range(n_samples):
                # The guide is sampled implicitly when we call the model.
                h = self.relu(self.fc1(x))
                h = self.relu(self.fc2(h))
                out = self.predict(h)
                sampled_q_values.append(out)

            sampled_q_values = torch.stack(sampled_q_values)
            
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
