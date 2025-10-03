# train.py

# Python imports
import time
import torch
import argparse
import numpy as np
import gymnasium as gym

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
    args = parser.parse_args()

    #  setup & config
    model_name_upper = args.model.upper()
    logger = configure_logger(f'{model_name_upper}_BFTQ_train')
    tb_logger = TensorBoardLogger(log_dir=f"logs/tensorboard_{args.model}")

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"  # only the baseline works with cuda (yet), switching btw. cuda and cpu does not make a huge diff.
    logger.info(f"Using device: {device}")
    logger.info(f"Training model type: {args.model} for {args.total_episodes} episodes.")

    # create the pool of parallel environments
    env = make_vec_env(
        "merge-v0",
        n_envs=args.num_envs,
        vec_env_cls=SubprocVecEnv,
        wrapper_class=FlattenObservation  # ensure input to agent is 1d
    )

    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n



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

    agent = AgentClass(
        state_dim,
        n_actions,
        config,
        network=NetworkClass,
        device=device,
        logger=logger,
        tb_logger=tb_logger
    )

    # set the training mode if the agent supports it
    if hasattr(agent, "set_training_mode"):
        agent.set_training_mode(args.training_mode)

    # ----- training logic starts here -----
    n_episodes = 0
    total_rewards_per_env = np.zeros(args.num_envs)
    start_time = time.time()

    states = env.reset()
    betas = np.random.uniform(size=args.num_envs)

    while n_episodes < args.total_episodes:
        actions = []
        for i in range(args.num_envs):
            action, new_beta = agent.act(states[i], betas[i]) # agent now uses the SubprocVecEnv, so NUM_ENV steps are taken at once
            actions.append(action) # we need to collect these actions and then iterate through them per core
            betas[i] = new_beta

        next_states, rewards, dones, infos = env.step(actions)

        for i in range(args.num_envs):
            agent.push_transition(
                states[i], actions[i], rewards[i],
                infos[i].get("cost", 0), betas[i], next_states[i], dones[i]
            )
            total_rewards_per_env[i] += rewards[i]

            if dones[i]:
                n_episodes += 1
                logger.info(
                    f"Episode {n_episodes}/{args.total_episodes}, "
                    f"total reward: {total_rewards_per_env[i]:.4f}"
                )
                tb_logger.log_scalar('reward/total_reward', total_rewards_per_env[i], n_episodes)

                # reset episode specific counters
                total_rewards_per_env[i] = 0
                betas[i] = np.random.uniform()

                # if reached our goal, stop
                if n_episodes >= args.total_episodes:
                    break

        states = next_states

        if len(agent.replay_buffer) > config["batch_size"]:
            for _ in range(args.num_envs):
                agent.update()

    # cleanup
    end_time = time.time()
    logger.info(f"Training finished in {end_time - start_time:.2f} seconds.")

    env.close()
    tb_logger.close()

    save_path = f"model_weights/{args.model}_bftq_model.pt"
    agent.save_model(save_path)
    logger.info(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()