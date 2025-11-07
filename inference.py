import argparse
import time
import os
import json  
from types import SimpleNamespace
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
    "baseline": (BFTQAgent, BudgetedQNet),
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
    lane_id = info.get("lane_id", None)

    # ego_vehicle = info.get("ego_vehicle", None)

    # if ego_vehicle is not None and hasattr(ego_vehicle, "lane_index"):
    #     _, _, lane_id = ego_vehicle.lane_index
    # else:
    #     lane_id = None

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


def main():  # mod: changed all CLI arguments to now depend on individual experiments, not paths
    parser = argparse.ArgumentParser(description="Inference script for BFTQ and variants")
    parser.add_argument("--run-name", type=str, required=True, help="The unique name of the run to load.")
    parser.add_argument("--logdir", type=str, default="logs", help="The base directory where logs are stored.")
    parser.add_argument("--n-episodes", type=int, default=50, help="The number of episodes to run for evaluation.")
    parser.add_argument("--num-envs", type=int, default=14, help="The number of parallel environments to run.")
    parser.add_argument("--env", type=str, default="two-way-v0", help="The environment to evaluate on.")
    parser.add_argument("--fixed-beta", type=float, default=None, help="If set, use this fixed beta value for all episodes.")
    args = parser.parse_args()


    # load config and model from the experiment dir
    exp_dir = os.path.join(args.logdir, args.run_name)
    config_path = os.path.join(exp_dir, "config.json")
    model_path = os.path.join(exp_dir, "model.pt")

    if not os.path.exists(config_path) or not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find config.json or model.pt in {exp_dir}")

    with open(config_path, 'r') as f:
        train_args_dict = json.load(f)
    # convert dict to a namespace object (e.g., train_args.model)
    train_args = SimpleNamespace(**train_args_dict)

    logger = configure_logger(f'{train_args.model.upper()}_BFTQ_inference')
    logger.info(f"Loading experiment from: {exp_dir}")
    logger.info(f"Loaded training args: {train_args}")

    # create a sub-dir for specific inference run's logs
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    inference_log_dir = os.path.join(exp_dir, f"inference_{timestamp}")
    tb_logger = TensorBoardLogger(log_dir=inference_log_dir)

    # === Env setup with per-env config ===
    def _wrap(env):
        return FlattenObservation(LaneInfoWrapper(env))

    env_config = {}
    if args.env == "highway-v0":
        env_config = {"duration": 15, "simulation_frequency": 15}
    elif args.env == "two-way-v0" or args.env == "merge-v0":
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

    # agent config is built from the loaded train_args
    config = {
        "gamma": 0.99,
        "batch_size": 32, # Not used in inference but required by agent
        "buffer_size": 1, # Not used in inference
        "learning_rate": 0, # Not used in inference
        "target_update": 100, # Not used in inference
        "layers": [64, 64],
        "exploration": {"temperature": 0.0, "final_temperature": 0.0, "tau": 0}, # No exploration
        "hull_options": dict(library="scipy", decimals=2, remove_duplicates=True),
    }

    if train_args.model in ["bnn", "mc", "ensemble"]:
        config["k"] = train_args.k
    if train_args.model == "mc":
        config["dropout_p"] = train_args.dropout_p
        config["n_samples"] = train_args.n_samples
    if train_args.model == "ensemble":
        config["n_models"] = train_args.n_models

    agent_class, network_class = AGENT_MAP[train_args.model]
    agent = agent_class(state_dim, n_actions, config, network=network_class, device="cpu", logger=logger)
    
    logger.info(f"Loading model weights from {model_path}")
    agent.load_model(model_path)

    if hasattr(agent, "set_training_mode"):
        agent.set_training_mode(train_args.training_mode)

    n_episodes = 0
    global_step = 0
    total_rewards_per_env = np.zeros(args.num_envs)
    total_costs_per_env = np.zeros(args.num_envs)

    all_episode_rewards = []
    all_episode_costs = []
    all_initial_betas = []

    states = env.reset()
    if args.fixed_beta is not None:
        betas = np.full(args.num_envs, args.fixed_beta)
    else:
        betas = np.random.uniform(size=args.num_envs)
    init_betas = betas.copy()

    while n_episodes < args.n_episodes:
        actions = []
        # Act
        for i in range(args.num_envs):
            action, _, _, _, _ = agent.act(states[i], betas[i])
            actions.append(action)

        # Step
        next_states, rewards, dones, infos = env.step(actions)

        # Store
        for i in range(args.num_envs):
            cost = compute_cost(infos[i], H, args.env)
            total_rewards_per_env[i] += rewards[i]
            total_costs_per_env[i] += cost
        global_step += 1

        # Episode ends
        for i in range(args.num_envs):
            if dones[i]:
                n_episodes += 1
                total_reward = total_rewards_per_env[i]
                total_cost = total_costs_per_env[i]
                initial_beta = init_betas[i]

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
                # If fixed_beta is used, beta stays the same

                if n_episodes >= args.n_episodes:
                    break
        states = next_states

    # Cleanup and save
    env.close()
    tb_logger.close()

    avg_reward = np.mean(all_episode_rewards)
    avg_cost = np.mean(all_episode_costs)
    avg_budget = np.mean(all_initial_betas)

    logger.info(f"Average reward: {avg_reward:.3f}")
    logger.info(f"Average cost: {avg_cost:.5f}")
    logger.info(f"Average budget: {avg_budget:.5f}")

    # allows  orchestrator script to capture and log these results.
    print(f"FINAL_METRICS: avg_reward={avg_reward}, avg_cost={avg_cost}, avg_budget={avg_budget}")


if __name__ == "__main__":
    main()