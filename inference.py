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


def compute_cost(info, horizon):
    """
    Compute safety cost per time step, following the paper's approach.

    Cost = 1/H whenever the ego-vehicle is 'unsafe':
      - crashed = True, or
      - not driving in right lane (right_lane_reward < 1).
    """
    # Safety cost = 1/H if crashed or off right lane
    crashed = info.get("crashed", False)
    right_lane = info["rewards"].get("right_lane_reward", 1.0)

    if crashed or right_lane < 1.0:
        return 1.0 / horizon
    else:
        return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, default="bftq", help="The type of model to load (bftq, bnn, mc, ensemble)")
    parser.add_argument("--model-path", type=str, required=True, help="The path to the trained model")
    parser.add_argument("--n-episodes", type=int, default=10, help="The number of episodes to run")
    parser.add_argument("--env", type=str, default="merge-v0", help="The environment to use")
    parser.add_argument("--mode", type=str, default="pessimistic", help="The action selection mode (pessimistic or mean)")
    args = parser.parse_args()

    # === Logger ===
    model_name_upper = args.model_type.upper()
    logger = configure_logger(f'{model_name_upper}_BFTQ_inference')
    tb_logger = TensorBoardLogger(log_dir=f"logs/tensorboard_{args.model_type}_inference")

    env = gym.make(args.env, render_mode="human")
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
        "k": 1.96, # Risk-aversion parameter
        "dropout_p": 0.5, # Dropout probability
        "n_samples": 10, # Number of samples for MC Dropout
        "n_models": 5 # Number of models in the ensemble
    }

    agent_class, network_class = AGENT_MAP[args.model_type]
    state_dim = int(np.prod(env.observation_space.shape))
    agent = agent_class(state_dim, n_actions, config, network=network_class, device="cpu", logger=logger)

    agent.load_model(args.model_path)

    if hasattr(agent, "set_training_mode"):
        agent.set_training_mode(args.mode)

    all_total_rewards = []
    all_total_costs = []
    global_step = 0
    H = env.get_attr("config")[0]["duration"]

    for ep in range(args.n_episodes):
        state, _ = env.reset()
        done = False
        beta = np.random.uniform()
        total_reward = 0
        total_cost = 0

        while not done:
            action, new_beta, q_r, q_c, old_beta = agent.act(state, beta)
            next_state, reward, terminated, truncated, info = env.step(action)
            cost = compute_cost(info, H)
            done = terminated or truncated

            state = next_state
            total_reward += reward
            total_cost += cost
            beta = new_beta

            # === Log to TensorBoard per step ===
            tb_logger.log_scalar("step/reward", reward, global_step)
            tb_logger.log_scalar("step/cost", cost, global_step)
            tb_logger.log_scalar("step/pred_qr", q_r, global_step)
            tb_logger.log_scalar("step/pred_qc", q_c, global_step)
            tb_logger.log_scalar("step/beta_old", old_beta, global_step)
            tb_logger.log_scalar("step/beta_new", new_beta, global_step)
            global_step += 1

            env.render()
            time.sleep(0.08)

        all_total_rewards.append(total_reward)
        all_total_costs.append(total_cost)

        # === Log to TensorBoard per episode ===
        tb_logger.log_scalar("episode/total_reward", total_reward, ep)
        tb_logger.log_scalar("episode/total_cost", total_cost, ep)
        tb_logger.log_scalar("episode/initial_beta", beta, ep)

        logger.info(f"Episode {ep}, total reward: {total_reward:.2f}, total cost: {total_cost:.2f}")

    logger.info(f"Average total reward: {np.mean(all_total_rewards):.2f}")
    logger.info(f"Average total cost: {np.mean(all_total_costs):.2f}")
    env.close()
    tb_logger.close()


if __name__ == "__main__":
    main()
