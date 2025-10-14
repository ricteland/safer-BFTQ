import torch
import numpy as np
# from agents.bftq.bnn_bftq import BNNBFTQ
from agents.bftq.bnn_bftq2 import BNNBFTQ
from utils.replay_buffer import ReplayBuffer
from models.bnn import BayesianQNet
from agents.bftq.risk_averse_policies import PessimisticPytorchBudgetedFittedPolicy
from agents.bftq.policies import RandomBudgetedPolicy, EpsilonGreedyBudgetedPolicy

from utils.normalize import RunningMeanStd

class BNNBFTQAgent:
    def __init__(self, state_dim, n_actions, config, network=BayesianQNet, device="cpu", logger=None, tb_logger=None, debug=False):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.device = device
        self.logger = logger
        self.tb_logger = tb_logger
        self.debug = debug
        self.batch_size = config.get("batch_size", 32)

        # ata modification logs 10
        # use a running mean std normalization utility
        self.obs_normalizer = RunningMeanStd(shape=state_dim)

        # === Networks ===
        self.q_net = network(size_state=state_dim, n_actions=n_actions, layers=config.get("layers", [64, 64])).to(device)
        self.target_net = network(size_state=state_dim, n_actions=n_actions, layers=config.get("layers", [64, 64])).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        # === Buffer ===
        self.replay_buffer = ReplayBuffer(capacity=config.get("buffer_size", 10000))

        # === Training logic lives in BNNBFTQ ===
        self.bftq = BNNBFTQ(
            self.q_net,
            self.target_net,
            config,
            device=device,
            logger=self.logger,
            tb_logger=self.tb_logger,
            debug=self.debug
        )

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
        # normalize the state before acting
        state_norm = self.normalize_obs(state)
        state_tensor = torch.tensor(state_norm, dtype=torch.float32, device=self.device).flatten().unsqueeze(0)
        action, new_beta, q_r, q_c = self.policy.execute(state_tensor, beta)
        old_beta = beta
        return action, new_beta, q_r, q_c, old_beta

    def normalize_obs(self, obs):
        # Helper to normalize a single or batch of observations
        return np.clip((obs - self.obs_normalizer.mean) / self.obs_normalizer.std, -10, 10)

    # def act(self, state, beta):
    #     # The BNN-based policy expects a different forward pass
    #     # However, the policy itself handles the sampling, so we can just pass the state
    #     state = torch.tensor(state, dtype=torch.float32, device=self.device).flatten().unsqueeze(0)
    #     action, new_beta, q_r, q_c = self.policy.execute(state, beta)
    #     old_beta = beta
    #     return action, new_beta, q_r, q_c, old_beta

    def push_transition(self, state, *args):
        # self.replay_buffer.push(*args)
        self.obs_normalizer.update(np.array([state]))
        self.replay_buffer.push(state, *args)

    def update(self):
        # The BFTQ update logic needs to be adapted for BNNs
        # For now, we assume the existing BFTQ class can handle it if the network output is mean+std
        # This might need to be changed to a custom BNNBFTQ class if the loss is different

        # return self.bftq.update()
        if len(self.replay_buffer) < self.batch_size:
            return None

        # 1. sample unnormalized transitions from the buffer
        transitions = self.replay_buffer.sample(self.batch_size)
        state, action, reward, cost, beta, next_state, done = zip(*transitions)

        # 2. normalize the states and next_states from the batch
        state_norm = self.normalize_obs(np.array(state))
        next_state_norm = self.normalize_obs(np.array(next_state))

        # 3. convert to tensors
        state_tensor = torch.tensor(state_norm, dtype=torch.float32, device=self.device).view(self.batch_size, -1)
        next_state_tensor = torch.tensor(next_state_norm, dtype=torch.float32, device=self.device).view(self.batch_size,
                                                                                                        -1)
        action_tensor = torch.tensor(action, device=self.device, dtype=torch.long)
        reward_tensor = torch.tensor(reward, device=self.device, dtype=torch.float)
        cost_tensor = torch.tensor(cost, device=self.device, dtype=torch.float)
        beta_tensor = torch.tensor(beta, device=self.device, dtype=torch.float).unsqueeze(1)
        done_tensor = torch.tensor(done, device=self.device, dtype=torch.float)

        # 4. call the core logic with the pre-norm batch
        return self.bftq.update_from_batch(
            state_tensor, action_tensor, reward_tensor, cost_tensor,
            beta_tensor, next_state_tensor, done_tensor
        )


    def set_training_mode(self, mode):
        self.policy.pi_greedy.training_mode = mode

    def save_model(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load_model(self, path):
        self.q_net.load_state_dict(torch.load(path))
