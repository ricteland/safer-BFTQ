import argparse
import time
import os
import datetime
import highway_env
import gymnasium as gym
import numpy as np
from agents.bftq.agent import BFTQAgent
from agents.bftq.bnn_agent import BNNBFTQAgent
from agents.bftq.mc_agent import MCBFTQAgent
from agents.bftq.ensemble_agent import EnsembleBFTQAgent
from models.q_net import BudgetedQNet
from models.bnn import BayesianQNet
from models.mc_dropout import MCDropoutQNet
from models.ensemble import EnsembleQNet
from utils.logger import configure_logger, TensorBoardLogger
from gymnasium.wrappers import FlattenObservation
from gymnasium import Wrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv


# === Wrapper to expose lane_id in info (works with SubprocVecEnv) ===
class LaneInfoWrapper(Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        try:
            info["lane_id"] = int(self.env.unwrapped.vehicle.lane_index[2])
            info["ego_vehicle"] = self.env.unwrapped.vehicle # Add ego_vehicle to info
        except Exception:
            # Fallback if not available for any reason
            info["lane_id"] = None
            info["ego_vehicle"] = None
        return obs, reward, terminated, truncated, info

AGENT_MAP = {
    "bftq": (BFTQAgent, BudgetedQNet),
    "bnn": (BNNBFTQAgent, BayesianQNet),
    "mc": (MCBFTQAgent, MCDropoutQNet),
    "ensemble": (EnsembleBFTQAgent, EnsembleQNet),
}


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, default="bftq",
                        help="The type of model to load (bftq, bnn, mc, ensemble)")
    parser.add_argument("--model-path", type=str, required=True, help="The path to the trained model")
    parser.add_argument("--n-episodes", type=int, default=10, help="The number of episodes to run")
    parser.add_argument("--num-envs", type=int, default=14, help="The number of parallel environments to run")
    parser.add_argument("--env", type=str, default="merge-v0", help="The environment to use")
    parser.add_argument("--mode", type=str, default="pessimistic",
                        help="The action selection mode (pessimistic or mean)")
    parser.add_argument("--fixed-beta", type=float, default=None, help="If set, use this fixed beta value instead of random sampling.")
    parser.add_argument("--k", type=float, default=-3, help="The k value for pessimistic BFTQ.")
    args = parser.parse_args()

    logger = configure_logger(f'{args.model_type.upper()}_BFTQ_inference')
    tb_logger = TensorBoardLogger(log_dir=f"logs/tensorboard_{args.model_type}_inference")

    # === Env setup with per-env config ===
    def _wrap(env):
        return FlattenObservation(LaneInfoWrapper(env))

    env_config = {}
    if args.env == "highway-v0":
        env_config = {"duration": 15, "simulation_frequency": 15}
    elif args.env == "two-way-v0":
        env_config = {"simulation_frequency": 15}
    elif args.env == "merge-v0":
        env_config = {"simulation_frequency": 15}

    env = make_vec_env(
        args.env,
        n_envs=args.num_envs,
        env_kwargs={"config": env_config},
        vec_env_cls=SubprocVecEnv,
        wrapper_class=_wrap,
    )

    # Dynamic horizon H = duration * simulation_frequency
    if args.env == "highway-v0":
        H = 15*15
    else:
        H = 40
    # print(f'Horizon: {H} from {args.env}')

    state_dim = env.observation_space.shape[0]
    n_actions = int(env.action_space.n)

    config = {
        "gamma": 0.99,
        "batch_size": 16,
        "buffer_size": 50000,
        "learning_rate": 1e-3,
        "target_update": 100,
        "layers": [64, 64],
        "exploration": {"temperature": 0.0, "final_temperature": 0.0, "tau": 0},
        "hull_options": dict(library="scipy", decimals=2, remove_duplicates=True),
        "k": args.k,
        "dropout_p": 0.5,
        "n_samples": 10,
        "n_models": 5,
    }

    agent_class, network_class = AGENT_MAP[args.model_type]
    agent = agent_class(state_dim, n_actions, config, network=network_class, device="cpu", logger=logger)
    agent.load_model(args.model_path)

    if hasattr(agent, "set_training_mode"):
        agent.set_training_mode(args.mode)

    n_episodes = 0
    global_step = 0
    total_rewards_per_env = np.zeros(args.num_envs)
    total_costs_per_env = np.zeros(args.num_envs)

    all_episode_rewards = []
    all_episode_costs = []
    all_initial_betas = []

    states = env.reset()
    betas = np.random.uniform(size=args.num_envs)           # current beta per env
    init_betas = betas.copy()                               # initial beta per episode

    while n_episodes < args.n_episodes:
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
            cost = compute_cost(infos[i], H, args.env)

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

                all_episode_rewards.append(total_reward)
                all_episode_costs.append(total_cost)
                all_initial_betas.append(initial_beta)

                logger.info(
                    f"Episode {n_episodes}/{args.n_episodes} | "
                    f"Reward: {total_reward:.3f} | Env cost: {total_cost:.3f} | Init beta: {initial_beta:.3f}"
                )

                # Reset counters for that env and resample next episode's beta
                total_rewards_per_env[i] = 0.0
                total_costs_per_env[i] = 0.0
                if args.fixed_beta is None:
                    new_init = np.random.uniform()
                    betas[i] = new_init
                    init_betas[i] = new_init
                else:
                    betas[i] = args.fixed_beta
                    init_betas[i] = args.fixed_beta

                if n_episodes >= args.n_episodes:
                    break

        states = next_states

    # Cleanup and save
    env.close()
    tb_logger.close()

    logger.info(f"Average reward: {np.mean(all_episode_rewards):.3f}")
    logger.info(f"Average cost: {np.mean(all_episode_costs):.5f}")
    logger.info(f"Average budget:{np.mean(all_initial_betas):.5f} ")


if __name__ == "__main__":
    main()
