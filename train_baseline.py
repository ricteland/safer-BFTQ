# Python imports
import time
import torch
import argparse
import numpy as np
import gymnasium as gym

# Env
import highway_env

# Parallelization
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# Local imports
from models.q_net import BudgetedQNet
from agents.bftq.agent import BFTQAgent
from utils.logger import configure_logger, TensorBoardLogger



def main():
    # === Logger ===
    logger = configure_logger('BFTQ_train')
    tb_logger = TensorBoardLogger()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"  # switching btw. cpu & gpu does not add overhead (at least on my system)

    logger.info(f"Using device: {device}")

    NUM_ENVS = 14  # basically number of cpu cores

    env = make_vec_env(  # creates pool of envs via subprocvecenv
        "merge-v0",
        n_envs=NUM_ENVS,
        vec_env_cls=SubprocVecEnv,
        wrapper_class=FlattenObservation # ensure input to agent is 1d
    )

    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

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

    agent = BFTQAgent(
        state_dim,
        n_actions,
        config,
        network=BudgetedQNet,
        device=device,
        logger=logger,
        tb_logger=tb_logger
    )

    TARGET_EPISODES = 100  # in batched format, we cannot have -exactly- this number of episodes,
    n_episodes = 0
    total_rewards_per_env = np.zeros(NUM_ENVS)

    start = time.time()

    states = env.reset()
    betas = np.random.uniform(size=NUM_ENVS)  # batch of betas

    while n_episodes < TARGET_EPISODES:
        actions = []
        for i in range(NUM_ENVS):
            action, new_beta = agent.act(states[i], betas[i])  # agent now uses the SubprocVecEnv, so NUM_ENV steps are taken at once
            actions.append(action)  # we need to collect these actions and then iterate through them per core
            betas[i] = new_beta

        next_states, rewards, dones, infos = env.step(actions)

        for i in range(NUM_ENVS):
            agent.push_transition(
                states[i], actions[i], rewards[i],
                infos[i].get("cost", 0), betas[i], next_states[i], dones[i]
            )

            total_rewards_per_env[i] += rewards[i]

            if dones[i]:
                n_episodes += 1
                logger.info(
                    f"Episode {n_episodes}/{TARGET_EPISODES}, "
                    f"total reward: {total_rewards_per_env[i]:.4f}"
                )
                tb_logger.log_scalar('reward/total_reward', total_rewards_per_env[i], n_episodes)

                # reset episode specific counters
                total_rewards_per_env[i] = 0
                betas[i] = np.random.uniform()

                # if reached our goal, stop
                if n_episodes >= TARGET_EPISODES:
                    break


        states = next_states

        if len(agent.replay_buffer) > config["batch_size"]:
            for _ in range(NUM_ENVS):
                agent.update()

    print(f'Time taken: {time.time() - start}')

    env.close()
    tb_logger.close()
    agent.save_model("model_weights/bftq_model.pt")


if __name__ == "__main__":
    main()