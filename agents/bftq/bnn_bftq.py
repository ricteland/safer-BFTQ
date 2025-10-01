import torch
import torch.nn as nn
import numpy as np
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.infer.autoguide import AutoDiagonalNormal

from agents.bftq.bftq import BFTQ

class BNNBFTQ(BFTQ):
    def __init__(self, q_net, target_net, replay_buffer, config, device="cpu", logger=None, tb_logger=None):
        # The optimizer is now handled by Pyro's SVI
        super().__init__(q_net, target_net, None, replay_buffer, config, device, logger, tb_logger)
        
        # Set up Pyro's SVI
        self.guide = AutoDiagonalNormal(self.q_net)
        self.svi = SVI(self.q_net, 
                       self.guide, 
                       Adam({"lr": config.get("learning_rate", 1e-3)}), 
                       loss=Trace_ELBO())

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        transitions = self.replay_buffer.sample(self.batch_size)
        state, action, reward, cost, beta, next_state, done = zip(*transitions)

        # --- convert to tensors ---
        state = torch.tensor(np.array(state), dtype=torch.float32, device=self.device).view(self.batch_size, -1)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32, device=self.device).view(self.batch_size, -1)
        action = torch.tensor(action, device=self.device, dtype=torch.long)
        reward = torch.tensor(reward, device=self.device, dtype=torch.float)
        cost = torch.tensor(cost, device=self.device, dtype=torch.float)
        beta = torch.tensor(beta, device=self.device, dtype=torch.float).unsqueeze(1)
        done = torch.tensor(done, device=self.device, dtype=torch.float)

        # --- target Q ---
        with torch.no_grad():
            # We need to sample from the target network to get a stable target
            (next_q_r_mean, _), (next_q_c_mean, _) = self.target_net(next_state, beta)
            next_q_r = next_q_r_mean.max(1)[0]
            next_q_c = next_q_c_mean.max(1)[0]
            target_q_r = reward + self.gamma * (1 - done) * next_q_r
            target_q_c = cost + self.gamma * (1 - done) * next_q_c

        # --- SVI loss + update ---
        # The model's forward pass will be called inside SVI.step
        # We need to condition the model on the observed data (the targets)
        loss = self.svi.step(state, beta, target_q_r, target_q_c, action)

        # --- sync target net ---
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if self.logger:
            self.logger.info(f"Step {self.steps}: loss={loss:.4f}")
        if self.tb_logger:
            self.tb_logger.log_scalar('loss/total', loss, self.steps)

        return loss
