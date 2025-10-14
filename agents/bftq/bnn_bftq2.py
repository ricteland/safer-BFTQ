import torch
import torch.nn as nn
import numpy as np
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam, Adam
from pyro.infer.autoguide import AutoDiagonalNormal


class BNNBFTQ:
    def __init__(self, q_net, target_net, config, device="cpu", logger=None, tb_logger=None, debug=False):
        self.q_net = q_net
        self.target_net = target_net
        self.gamma = config.get("gamma", 0.99)
        self.batch_size = config.get("batch_size", 32)
        self.target_update = config.get("target_update", 100)
        self.device = device
        self.logger = logger
        self.tb_logger = tb_logger
        self.debug = debug
        self.steps = 0

        self.q_net.debug = self.debug
        self.target_net.debug = self.debug

        lr = config.get("learning_rate", 1e-3)
        if self.debug:
            print(f"[{self.__class__.__name__}] Using learning rate: {lr}")

        clip_norm = 10.0
        if self.debug:
            print(f"[{self.__class__.__name__}] Using ClippedAdam with norm: {clip_norm}")

        # optimizer_args = {"lr": lr, "clip_norm": clip_norm}
        # optimizer = ClippedAdam(optimizer_args)
        # optimizer = Adam({"lr": config.get("learning_rate", 1e-3)}),

        self.guide = AutoDiagonalNormal(self.q_net)
        self.svi = SVI(
            self.q_net,
            self.guide,
            # Adam({"lr": lr}),
            ClippedAdam({"lr": lr, "clip_norm": clip_norm}),
            loss=Trace_ELBO()
        )

    def update_from_batch(self, state, action, reward, cost, beta, next_state, done):

        # calculate target q-values
        with torch.no_grad():
            (next_q_r_mean, _), (next_q_c_mean, _) = self.target_net(next_state, beta)
            next_q_r = next_q_r_mean.max(1)[0]
            next_q_c = next_q_c_mean.max(1)[0]
            target_q_r = reward + self.gamma * (1 - done) * next_q_r
            target_q_c = cost + self.gamma * (1 - done) * next_q_c

        # normalize target q-values ---
        if self.debug:
            print("\n--- [BFTQ Update @ Step] ---")
            print(f"  [Targets Pre-Norm] Q_r: mean={target_q_r.mean():.3f}, std={target_q_r.std():.3f}")
            print(f"  [Targets Pre-Norm] Q_c: mean={target_q_c.mean():.3f}, std={target_q_c.std():.3f}")

        q_r_mean = target_q_r.mean()
        q_r_std = target_q_r.std() + 1e-6
        target_q_r_norm = (target_q_r - q_r_mean) / q_r_std

        q_c_mean = target_q_c.mean()
        q_c_std = target_q_c.std() + 1e-6
        target_q_c_norm = (target_q_c - q_c_mean) / q_c_std

        if self.debug:
            print(f"  [Targets Post-Norm] Q_r: mean={target_q_r_norm.mean():.3f}, std={target_q_r_norm.std():.3f}")
            print(f"  [Targets Post-Norm] Q_c: mean={target_q_c_norm.mean():.3f}, std={target_q_c_norm.std():.3f}")

        # svi loss calculation & update step
        unscaled_loss = self.svi.step(state, beta, target_q_r_norm, target_q_c_norm, action)
        loss = unscaled_loss / self.batch_size

        # sync target network
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # logging
        if self.logger:
            log_message = f"Step {self.steps}: scaled_loss={loss:.4f}"
            if self.steps % 10 == 0 or self.debug:
                self.logger.info(log_message)

        if self.tb_logger:
            self.tb_logger.log_scalar('loss/total_scaled', loss, self.steps)

        return loss