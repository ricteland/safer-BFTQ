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
    parser.add_argument("--env", type=str, default="merge-v0", help="The environment to use")
    parser.add_argument("--mode", type=str, default="pessimistic",
                        help="The action selection mode (pessimistic or mean)")
    args = parser.parse_args()

    logger = configure_logger(f'{args.model_type.upper()}_BFTQ_inference')
    tb_logger = TensorBoardLogger(log_dir=f"logs/tensorboard_{args.model_type}_inference")

    env = gym.make(args.env, render_mode="human", config={"show_trajectories": True, "duration": 15, "simulation_frequency": 15})
    env = FlattenObservation(env)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    config = {
        "gamma": 0.99,
        "batch_size": 16,
        "buffer_size": 50000,
        "learning_rate": 1e-3,
        "target_update": 100,
        "layers": [64, 64],
        "exploration": {"temperature": 0.0, "final_temperature": 0.0, "tau": 0},
        "hull_options": dict(library="scipy", decimals=2, remove_duplicates=True),
        "k": -3,
        "dropout_p": 0.5,
        "n_samples": 10,
        "n_models": 5,
    }

    agent_class, network_class = AGENT_MAP[args.model_type]
    agent = agent_class(state_dim, n_actions, config, network=network_class, device="cpu", logger=logger)
    agent.load_model(args.model_path)

    if hasattr(agent, "set_training_mode"):
        agent.set_training_mode(args.mode)

    all_total_rewards, all_total_costs, all_total_budgets = [], [], []
    if args.env == "highway-v0":
        H = env.unwrapped.config["duration"] * env.unwrapped.config["simulation_frequency"]
    else:
        H = 40
    global_step = 0
    print(f'Horizon: {H} from {args.env}')
    for ep in range(args.n_episodes):
        state, _ = env.reset()
        done = False

        # Initialize the budget (beta) at the start of each episode
        initial_budget = np.random.uniform()
        beta = initial_budget  # Store initial budget
        all_total_budgets.append(initial_budget)
        total_reward, total_env_cost, total_pred_cost = 0, 0, 0

        # Log the initial budget
        logger.info(f"Episode {ep}: Initial Budget (Beta) = {initial_budget:.4f}")
        tb_logger.log_scalar("episode/initial_budget", initial_budget, ep)

        while not done:
            action, new_beta, q_r, q_c, old_beta = agent.act(state, beta)
            next_state, reward, terminated, truncated, info = env.step(action)

            # attach ego vehicle for cost computation
            info["ego_vehicle"] = env.unwrapped.vehicle
            cost = compute_cost(info, H, args.env)

            done = terminated or truncated

            # Accumulate
            total_reward += reward
            total_env_cost += cost
            total_pred_cost += q_c  # <-- model-predicted cumulative cost
            beta = new_beta
            state = next_state

            # Logging
            tb_logger.log_scalar("step/reward", reward, global_step)
            tb_logger.log_scalar("step/pred_qr", q_r, global_step)
            tb_logger.log_scalar("step/pred_qc", q_c, global_step)
            tb_logger.log_scalar("step/env_cost", cost, global_step)
            tb_logger.log_scalar("step/beta_old", old_beta, global_step)
            tb_logger.log_scalar("step/beta_new", new_beta, global_step)
            global_step += 1

            env.render()
            time.sleep(0.08)

        all_total_rewards.append(total_reward)
        all_total_costs.append(total_env_cost)

        tb_logger.log_scalar("episode/total_reward", total_reward, ep)
        tb_logger.log_scalar("episode/predicted_total_cost", total_pred_cost, ep)
        tb_logger.log_scalar("episode/env_total_cost", total_env_cost, ep)


        logger.info(
            f"Episode {ep}: reward={total_reward:.2f}, pred_cost={total_pred_cost:.4f}, env_cost={total_env_cost:.4f}, crashed={info.get('crashed', False)}, initial_budget={initial_budget:.4f}, violation = {total_env_cost > initial_budget}"
        )

    logger.info(f"Average reward: {np.mean(all_total_rewards):.3f}")
    logger.info(f"Average cost: {np.mean(all_total_costs):.5f}")
    logger.info(f"Average budget:{np.mean(all_total_budgets):.5f} ")
    env.close()
    tb_logger.close()


if __name__ == "__main__":
    main()