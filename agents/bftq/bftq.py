# bftq.py
import torch
import torch.nn as nn
import numpy as np

class BFTQ:
    def __init__(self, q_net, target_net, optimizer, replay_buffer, config, device="cpu", logger=None, tb_logger=None):
        self.q_net = q_net
        self.target_net = target_net
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        self.config = config
        self.device = device
        self.steps = 0
        self.gamma = config.get("gamma", 0.99)
        self.batch_size = config.get("batch_size", 32)
        self.target_update = config.get("target_update", 100)
        self.logger = logger
        self.tb_logger = tb_logger

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        transitions = self.replay_buffer.sample(self.batch_size)
        state, action, reward, cost, beta, next_state, done = zip(*transitions)


        # --- convert to tensors ---
        state = torch.tensor(np.array(state), dtype=torch.float32, device=self.device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32, device=self.device)
        action = torch.tensor(action, device=self.device).long()
        reward = torch.tensor(reward, device=self.device).float()
        cost = torch.tensor(cost, device=self.device).float()
        beta = torch.tensor(beta, device=self.device).float().unsqueeze(1)
        done = torch.tensor(done, device=self.device).float()
        state = state.view(state.size(0), -1)  # [batch, state_dim]
        next_state = next_state.view(next_state.size(0), -1)
        # --- forward pass ---
        # print("DEBUG shapes before network:")
        # print(f"  state.shape     = {state}")
        # print(f"  beta.shape      = {beta}")
        # print(f"  next_state.shape= {next_state}")

        if getattr(self.q_net, "output_type", "q_values") == "q_values":
            q_r, q_c = self.q_net(state, beta)
            # pick the Q-values for the chosen action
            q_r = q_r.gather(1, action.unsqueeze(1)).squeeze()
            q_c = q_c.gather(1, action.unsqueeze(1)).squeeze()

        elif self.q_net.output_type == "distribution":
            (q_r_mean, q_r_std), (q_c_mean, q_c_std) = self.q_net.predict_dist(state, beta)

            # pick the mean for the chosen action
            q_r = q_r_mean.gather(1, action.unsqueeze(1)).squeeze()
            q_c = q_c_mean.gather(1, action.unsqueeze(1)).squeeze()

            # NOTE: std is available if you want to use it in loss or exploration
            q_r_uncertainty = q_r_std.gather(1, action.unsqueeze(1)).squeeze()
            q_c_uncertainty = q_c_std.gather(1, action.unsqueeze(1)).squeeze()

        else:
            raise ValueError(f"Unknown output_type {self.q_net.output_type}")

        # --- target Q ---
        with torch.no_grad():
            if self.q_net.output_type == "q_values":
                next_q_r, next_q_c = self.target_net(next_state, beta)
                next_q_r = next_q_r.max(1)[0]
                next_q_c = next_q_c.max(1)[0]

            elif self.q_net.output_type == "distribution":
                (next_q_r_mean, _), (next_q_c_mean, _) = self.target_net.predict_dist(next_state, beta)
                next_q_r = next_q_r_mean.max(1)[0]
                next_q_c = next_q_c_mean.max(1)[0]

            else:
                raise ValueError(f"Unknown output_type {self.q_net.output_type}")

            target_q_r = reward + self.gamma * (1 - done) * next_q_r
            target_q_c = cost + self.gamma * (1 - done) * next_q_c
        # --- loss + update ---
        loss_r = nn.MSELoss()(q_r, target_q_r)
        loss_c = nn.MSELoss()(q_c, target_q_c)
        loss = loss_r + loss_c

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # --- sync target net ---
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if self.logger:
            self.logger.info(f"Step {self.steps}: loss={loss.item():.4f}, loss_r={loss_r.item():.4f}, loss_c={loss_c.item():.4f}")
        if self.tb_logger:
            self.tb_logger.log_scalar('loss/total', loss.item(), self.steps)
            self.tb_logger.log_scalar('loss/reward', loss_r.item(), self.steps)
            self.tb_logger.log_scalar('loss/cost', loss_c.item(), self.steps)


        return loss.item()
