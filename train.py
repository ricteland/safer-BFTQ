# train.py

# Python imports
import time
import torch
import argparse
import numpy as np
import gymnasium as gym
import os
from datetime import datetime
# Environment
import highway_env

# Parallelization imports
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# Local imports
from models.q_net import BudgetedQNet
from models.bnn import BayesianQNet
from models.ensemble import EnsembleQNet
from models.mc_dropout import MCDropoutQNet
from agents.bftq.agent import BFTQAgent
from agents.bftq.bnn_agent import BNNBFTQAgent
from agents.bftq.ensemble_agent import EnsembleBFTQAgent
from agents.bftq.mc_agent import MCBFTQAgent
from utils.logger import configure_logger, TensorBoardLogger
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
    # args
    parser = argparse.ArgumentParser(description="Main training logic for the agents")
    parser.add_argument("--model", type=str, required=True, choices=["baseline", "bnn", "mc", "ensemble"],
                        help="The type of model to train.")
    parser.add_argument("--num-envs", type=int, default=14, help="Number of parallel environments (CPU cores).")
    parser.add_argument("--total-episodes", type=int, default=500, help="Total number of episodes to train for.")
    parser.add_argument("--training-mode", type=str, default="pessimistic", choices=["pessimistic", "mean"],
                        help="The action selection mode for training.")

    # model-specific hyperparameters
    parser.add_argument("--k", type=float, default=1.96, help="Risk-aversion parameter (for bnn, mc, ensemble).")
    parser.add_argument("--n-models", type=int, default=5, help="Number of models in the ensemble.")
    parser.add_argument("--dropout-p", type=float, default=0.5, help="Dropout probability for MC Dropout.")
    parser.add_argument("--n-samples", type=int, default=10, help="Number of samples for MC Dropout.")

    parser.add_argument("--debug", action="store_true", help="Enable detailed debug prints.")

    parser.add_argument("-logdir", type=str, default='logs')

    args = parser.parse_args()

    #  setup & config
    model_name_upper = args.model.upper()
    logger = configure_logger(f'{model_name_upper}_BFTQ_train')

    os.makedirs(args.logdir, exist_ok=True)

    tb_logger = TensorBoardLogger(log_dir=f"{args.logdir}/tensorboard_{args.model}")

    device = "cpu"  # only the baseline works with cuda (yet), switching btw. cuda and cpu does not make a huge diff.

    logger.info(f"Using device: {device}")
    logger.info(f"Training model type: {args.model} for {args.total_episodes} episodes.")
    if args.debug:
        logger.info("***** DEBUG MODE ENABLED *****")

    # create the pool of parallel environments
    env = make_vec_env(
        "two-way-v0",
        n_envs=args.num_envs,
        vec_env_cls=SubprocVecEnv,
        wrapper_class=FlattenObservation  # ensure input to agent is 1d
    )
    # Set duration to 15 seconds for all sub-environments
    env.set_attr("config", [{"duration": 15} for _ in range(args.num_envs)])

    # Environment horizon (episode length)
    #H = env.get_attr("config")[0]["duration"]
    H=15
    logger.info(f"Detected horizon (H): {H}")

    state_dim = env.observation_space.shape[0]
    n_actions = int(env.action_space.n)



    # base config - shared by all models
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

    # select model
    agent_map = {
        "baseline": (BFTQAgent, BudgetedQNet),
        "bnn": (BNNBFTQAgent, BayesianQNet),
        "mc": (MCBFTQAgent, MCDropoutQNet),
        "ensemble": (EnsembleBFTQAgent, EnsembleQNet)
    }
    AgentClass, NetworkClass = agent_map[args.model]

    # add model-specific hyperparameters to config
    if args.model in ["bnn", "mc", "ensemble"]:
        config["k"] = args.k
    if args.model == "mc":
        config["dropout_p"] = args.dropout_p
        config["n_samples"] = args.n_samples
    if args.model == "ensemble":
        config["n_models"] = args.n_models

    agent_kwargs = {
        "state_dim": state_dim,
        "n_actions": n_actions,
        "config": config,
        "network": NetworkClass,
        "device": device,
        "logger": logger,
        "tb_logger": tb_logger
    }

    if args.model == "bnn":
        agent_kwargs["debug"] = args.debug

    agent = AgentClass(**agent_kwargs)

    # set the training mode if the agent supports it
    if hasattr(agent, "set_training_mode"):
        agent.set_training_mode(args.training_mode)


    # ----- training logic starts here -----
    n_episodes = 0
    global_step = 0
    total_rewards_per_env = np.zeros(args.num_envs)
    total_costs_per_env = np.zeros(args.num_envs)
    start_time = time.time()

    states = env.reset()
    betas = np.random.uniform(size=args.num_envs)

    while n_episodes < args.total_episodes:
        actions = []
        q_r_list, q_c_list, old_beta_list, new_beta_list = [], [], [], []

        # === Step 1: Act in each environment ===
        for i in range(args.num_envs):
            action, new_beta, q_r, q_c, old_beta = agent.act(states[i], betas[i])
            actions.append(action)
            q_r_list.append(q_r)
            q_c_list.append(q_c)
            old_beta_list.append(old_beta)
            new_beta_list.append(new_beta)
            betas[i] = new_beta

        # === Step 2: Environment step ===
        next_states, rewards, dones, infos = env.step(actions)
        # === Step 3: Push transitions and accumulate ===
        for i in range(args.num_envs):
            cost = compute_cost(infos[i], H)
            agent.push_transition(
                states[i], actions[i], rewards[i],
                cost, betas[i], next_states[i], dones[i]
            )
            total_rewards_per_env[i] += rewards[i]
            total_costs_per_env[i] += cost

        # === Step 4: Log EVERYTHING to TensorBoard per step ===
        # One averaged log across all envs (keeps logs manageable)
        tb_logger.log_scalar("step/reward_mean", np.mean(rewards), global_step)
        tb_logger.log_scalar("step/pred_qr_mean", np.mean(q_r_list), global_step)
        tb_logger.log_scalar("step/pred_qc_mean", np.mean(q_c_list), global_step)
        tb_logger.log_scalar("step/beta_old_mean", np.mean(old_beta_list), global_step)
        tb_logger.log_scalar("step/beta_new_mean", np.mean(new_beta_list), global_step)
        tb_logger.log_scalar("step/beta_var", np.var(new_beta_list), global_step)

        # Optional: log per-environment values (comment out if too large)
        # for i in range(args.num_envs):
        #     tb_logger.log_scalar(f"env_{i}/reward", rewards[i], global_step)
        #     tb_logger.log_scalar(f"env_{i}/pred_qr", q_r_list[i], global_step)
        #     tb_logger.log_scalar(f"env_{i}/pred_qc", q_c_list[i], global_step)
        #     tb_logger.log_scalar(f"env_{i}/beta_old", old_beta_list[i], global_step)
        #     tb_logger.log_scalar(f"env_{i}/beta_new", new_beta_list[i], global_step)

        global_step += 1  # <-- increment per environment step

        # === Step 5: Check for completed episodes ===
        for i in range(args.num_envs):
            if dones[i]:
                n_episodes += 1

                # Log to TensorBoard (episode-level aggregates)
                tb_logger.log_scalar("episode/total_reward", total_rewards_per_env[i], n_episodes)
                tb_logger.log_scalar("episode/total_pred_cost", total_costs_per_env[i], n_episodes)
                tb_logger.log_scalar("episode/last_beta_old", old_beta_list[i], n_episodes)
                tb_logger.log_scalar("episode/last_beta_new", new_beta_list[i], n_episodes)
                tb_logger.log_scalar("episode/initial_beta", betas[i], n_episodes)

                # Minimal console output
                logger.info(
                    f"Episode {n_episodes}/{args.total_episodes} | "
                    f"Reward: {total_rewards_per_env[i]:.3f} | "
                    f"Pred cost: {total_costs_per_env[i]:.3f} | "
                    f"Init beta: {betas[i]:.3f}"
                )

                # Reset episode counters
                total_rewards_per_env[i] = 0
                total_costs_per_env[i] = 0
                betas[i] = np.random.uniform()

                if n_episodes >= args.total_episodes:
                    break

        # === Step 6: Prepare next state ===
        states = next_states

        # === Step 7: Agent update ===
        if len(agent.replay_buffer) > config["batch_size"]:
            for _ in range(args.num_envs):
                agent.update()

    # === Cleanup ===
    end_time = time.time()
    logger.info(f"Training finished in {end_time - start_time:.2f} seconds.")

    env.close()
    tb_logger.close()

    # === Dynamic save path ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("model_weights", exist_ok=True)
    save_path = f"model_weights/{args.model}_bftq_model_{timestamp}.pt"

    agent.save_model(save_path)
    logger.info(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
