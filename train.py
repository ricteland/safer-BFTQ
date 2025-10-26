# train.py

import time
import argparse
import numpy as np
import os
from datetime import datetime

import gymnasium as gym
import highway_env
from gymnasium import Wrapper
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from models.q_net import BudgetedQNet
from models.bnn import BayesianQNet
from models.ensemble import EnsembleQNet
from models.mc_dropout import MCDropoutQNet
from agents.bftq.agent import BFTQAgent
from agents.bftq.bnn_agent import BNNBFTQAgent
from agents.bftq.ensemble_agent import EnsembleBFTQAgent
from agents.bftq.mc_agent import MCBFTQAgent
from utils.logger import configure_logger, TensorBoardLogger


# === Wrapper to expose lane_id in info (works with SubprocVecEnv) ===
class LaneInfoWrapper(Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        try:
            info["lane_id"] = int(self.env.unwrapped.vehicle.lane_index[2])
        except Exception:
            # Fallback if not available for any reason
            info["lane_id"] = None
        return obs, reward, terminated, truncated, info


def compute_cost(info, H, env_id="two-way-v0"):
    """
    Compute safety cost for both two-way-v0 and highway-v0.

    two-way-v0:
        Cost = 1/H if crashed or lane_id == 0  (opposite lane)
    highway-v0:
        Cost = (rightmost - lane_id) * (1/H)  + crash cost
        → rightmost = 3 → cost 0
        → leftmost = 0 → cost 3/H
    """
    crashed = info.get("crashed", False)
    ego_vehicle = info.get("ego_vehicle", None)

    if ego_vehicle is not None and hasattr(ego_vehicle, "lane_index"):
        _, _, lane_id = ego_vehicle.lane_index
    else:
        lane_id = None

    if env_id == "two-way-v0":
        on_wrong = (lane_id == 0)
        return (1.0 / H) if (crashed or on_wrong) else 0.0

    elif env_id == "highway-v0":
        if lane_id is None:
            return (1.0 / H) if crashed else 0.0
        lane_cost = max(0, 3 - lane_id) * (1.0 / H)
        crash_cost = (1.0 / H) if crashed else 0.0
        return lane_cost + crash_cost

    else:
        # Default fallback
        return (1.0 / H) if crashed else 0.0


def main():
    parser = argparse.ArgumentParser(description="Training script for BFTQ and variants")
    parser.add_argument("--model", type=str, required=True, choices=["baseline", "bnn", "mc", "ensemble"])
    parser.add_argument("--num-envs", type=int, default=14)
    parser.add_argument("--total-episodes", type=int, default=500)
    parser.add_argument("--training-mode", type=str, default="pessimistic", choices=["pessimistic", "mean"])
    parser.add_argument("--k", type=float, default=1.96)
    parser.add_argument("--n-models", type=int, default=5)
    parser.add_argument("--dropout-p", type=float, default=0.5)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--logdir", type=str, default="logs")

    parser.add_argument("--env-id", type=str, default="two-way-v0", help="The ID of the highway-env environment to use.")
    parser.add_argument("--run-name", type=str, required=True, help="A unique name for the run, used for the TensorBoard log directory.")
    args = parser.parse_args()


    # === Setup ===
    model_name_upper = args.model.upper()
    logger = configure_logger(f"{model_name_upper}_BFTQ_train")
    os.makedirs(args.logdir, exist_ok=True)


    tb_logger = TensorBoardLogger(log_dir=f"{args.logdir}/{args.run_name}")
    device = "cpu"

    logger.info(f"Using device: {device}")
    logger.info(f"Training model: {args.model} | Episodes: {args.total_episodes}")
    if args.debug:
        logger.info("***** DEBUG MODE ENABLED *****")

    # === Env setup with per-env config ===
    def _wrap(env):
        return FlattenObservation(LaneInfoWrapper(env))

    env_config = {}
    if args.env_id == "highway-v0":
        env_config = {"duration": 15, "simulation_frequency": 15}
    elif args.env_id == "two-way-v0":
        env_config = {"simulation_frequency": 15}

    env = make_vec_env(
        args.env_id,
        n_envs=args.num_envs,
        env_kwargs={"config": env_config},  # <<<<<<<<<<<< this line sets duration
        vec_env_cls=SubprocVecEnv,
        wrapper_class=_wrap,
    )

    # Dynamic horizon H = duration * simulation_frequency
    if args.env_id == "highway-v0":
        H = 15*15
    else:
        H = 40
    print(f"H = {H} from {args.env_id}")

    state_dim = env.observation_space.shape[0]
    n_actions = int(env.action_space.n)

    # === Shared config ===
    config = {
        "gamma": 0.99,
        "batch_size": 32,
        "buffer_size": 50000,
        "learning_rate": 1e-3,
        "target_update": 100,
        "layers": [64, 64],
        "exploration": {"temperature": 1.0, "final_temperature": 0.1, "tau": 5000},
        "hull_options": dict(library="scipy", decimals=2, remove_duplicates=True),
    }

    # === Model selection ===
    agent_map = {
        "baseline": (BFTQAgent, BudgetedQNet),
        "bnn": (BNNBFTQAgent, BayesianQNet),
        "mc": (MCBFTQAgent, MCDropoutQNet),
        "ensemble": (EnsembleBFTQAgent, EnsembleQNet),
    }
    AgentClass, NetworkClass = agent_map[args.model]

    if args.model in ["bnn", "mc", "ensemble"]:
        config["k"] = args.k
    if args.model == "mc":
        config["dropout_p"] = args.dropout_p
        config["n_samples"] = args.n_samples
    if args.model == "ensemble":
        config["n_models"] = args.n_models

    agent = AgentClass(
        state_dim, n_actions, config, network=NetworkClass,
        device=device, logger=logger, tb_logger=tb_logger
    )
    if hasattr(agent, "set_training_mode"):
        agent.set_training_mode(args.training_mode)

    # === Training loop ===
    n_episodes = 0
    global_step = 0
    total_rewards_per_env = np.zeros(args.num_envs)
    total_costs_per_env = np.zeros(args.num_envs)

    states = env.reset()
    betas = np.random.uniform(size=args.num_envs)           # current beta per env
    init_betas = betas.copy()                               # initial beta per episode

    while n_episodes < args.total_episodes:
        actions = []
        q_r_list, q_c_list, old_beta_list, new_beta_list = [], [], [], []

        # Act
        for i in range(args.num_envs):
            action, new_beta, q_r, q_c, old_beta = agent.act(states[i], betas[i])
            actions.append(action)
            q_r_list.append(q_r)
            q_c_list.append(q_c)
            old_beta_list.append(old_beta)
            new_beta_list.append(new_beta)
            betas[i] = new_beta

        # Step
        next_states, rewards, dones, infos = env.step(actions)

        # Store
        for i in range(args.num_envs):
            cost = compute_cost(infos[i], H)
            agent.push_transition(states[i], actions[i], rewards[i], cost, betas[i], next_states[i], dones[i])
            total_rewards_per_env[i] += rewards[i]
            total_costs_per_env[i] += cost

        # Log step means
        tb_logger.log_scalar("step/reward_mean", float(np.mean(rewards)), global_step)
        tb_logger.log_scalar("step/pred_qr_mean", float(np.mean(q_r_list)), global_step)
        tb_logger.log_scalar("step/pred_qc_mean", float(np.mean(q_c_list)), global_step)
        tb_logger.log_scalar("step/beta_old_mean", float(np.mean(old_beta_list)), global_step)
        tb_logger.log_scalar("step/beta_new_mean", float(np.mean(new_beta_list)), global_step)
        tb_logger.log_scalar("step/beta_var", float(np.var(new_beta_list)), global_step)
        global_step += 1

        # Episode ends
        for i in range(args.num_envs):
            if dones[i]:
                n_episodes += 1
                total_reward = total_rewards_per_env[i]
                total_cost = total_costs_per_env[i]
                initial_beta = init_betas[i]  # true initial beta for this finished episode

                # Log episode stats
                tb_logger.log_scalar("episode/total_reward", float(total_reward), n_episodes)
                tb_logger.log_scalar("episode/total_env_cost", float(total_cost), n_episodes)
                tb_logger.log_scalar("episode/initial_beta", float(initial_beta), n_episodes)

                logger.info(
                    f"Episode {n_episodes}/{args.total_episodes} | "
                    f"Reward: {total_reward:.3f} | Env cost: {total_cost:.3f} | Init beta: {initial_beta:.3f}"
                )

                # Reset counters for that env and resample next episode's beta
                total_rewards_per_env[i] = 0.0
                total_costs_per_env[i] = 0.0
                new_init = np.random.uniform()
                betas[i] = new_init
                init_betas[i] = new_init

                if n_episodes >= args.total_episodes:
                    break

        states = next_states

        # Update agent
        if len(agent.replay_buffer) > config["batch_size"]:
            for _ in range(args.num_envs):
                agent.update()

    # Cleanup and save
    env.close()
    tb_logger.close()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("model_weights", exist_ok=True)
    save_path = f"model_weights/{args.model}_bftq_model_{timestamp}_{str(int(args.k*100))}.pt"
    agent.save_model(save_path)
    logger.info(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
