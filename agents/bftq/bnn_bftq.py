import torch
import torch.nn as nn
import numpy as np
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from pyro.infer.autoguide import AutoDiagonalNormal

from agents.bftq.bftq import BFTQ

class BNNBFTQ(BFTQ):
    def __init__(self, q_net, target_net, replay_buffer, config, device="cpu", logger=None, tb_logger=None, debug=False):
        # The optimizer is now handled by Pyro's SVI
        super().__init__(q_net, target_net, None, replay_buffer, config, device, logger, tb_logger)

        self.debug = debug
        if self.debug:
            print(f"[{self.__class__.__name__}] Debug mode enabled.")

        self.q_net.debug = self.debug
        self.target_net.debug = self.debug

        # ata modification logs 6:
        # try with reduced LR:
        lr = config.get("learning_rate", 1e-3) / 10.0
        if self.debug:
            print(f"[{self.__class__.__name__}] Using learning rate: {lr}")


        # ata modification logs 9:
        # add gradient clipping
        clip_norm = 10.0
        optimizer_args = {
            "lr": lr,
            "clip_norm": clip_norm,
        }
        if self.debug:
            print(f"[{self.__class__.__name__}] Using ClippedAdam with norm: {clip_norm}")

        optimizer = ClippedAdam(optimizer_args)


        # Set up Pyro's SVI
        self.guide = AutoDiagonalNormal(self.q_net)
        self.svi = SVI(self.q_net, 
                       self.guide, 
                       optimizer,
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

        # ata modification logs 7:
        # normalizing the target Q values
        if self.debug:
            print("\n--- [BFTQ Update @ Step] ---")
            print(f"  [Targets Pre-Norm] Q_r: mean={target_q_r.mean():.3f}, std={target_q_r.std():.3f}")
            print(f"  [Targets Pre-Norm] Q_c: mean={target_q_c.mean():.3f}, std={target_q_c.std():.3f}")

        # normalize reward targets
        q_r_mean = target_q_r.mean()
        q_r_std = target_q_r.std() + 1e-6
        target_q_r_norm = (target_q_r - q_r_mean) / q_r_std

        # normalize cost targets
        q_c_mean = target_q_c.mean()
        q_c_std = target_q_c.std() + 1e-6
        target_q_c_norm = (target_q_c - q_c_mean) / q_c_std

        if self.debug:
            print(f"  [Targets Post-Norm] Q_r: mean={target_q_r_norm.mean():.3f}, std={target_q_r_norm.std():.3f}")
            print(f"  [Targets Post-Norm] Q_c: mean={target_q_c_norm.mean():.3f}, std={target_q_c_norm.std():.3f}")

        # --- SVI loss + update ---
        # The model's forward pass will be called inside SVI.step
        # We need to condition the model on the observed data (the targets)
        unscaled_loss = self.svi.step(state, beta, target_q_r_norm, target_q_c_norm, action)

        # ata modification logs 8: scale the elbo loss by batch size
        loss = unscaled_loss / self.batch_size

        # --- sync target net ---
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if self.logger:
            log_message = (
                f"Step {self.steps}: "
                f"unscaled_loss={unscaled_loss:.4f}, "
                f"scaled_loss={loss:.4f}"
            )
            # Log to console less frequently to avoid clutter, especially in debug mode
            if self.steps % 10 == 0 or self.debug:
                self.logger.info(log_message)

        if self.tb_logger:
            # Log both to see the effect of scaling
            self.tb_logger.log_scalar('loss/total_unscaled', unscaled_loss, self.steps)
            self.tb_logger.log_scalar('loss/total_scaled', loss, self.steps)

        return loss
