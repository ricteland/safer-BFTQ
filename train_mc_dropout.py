import highway_env
import gymnasium as gym
import numpy as np
from agents.bftq.mc_agent import MCBFTQAgent
from models.mc_dropout import MCDropoutQNet
from utils.logger import configure_logger, TensorBoardLogger


print([env_spec.id for env_spec in gym.envs.registry.values() if "highway" in env_spec.id])


import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-mode", type=str, default="pessimistic", help="The action selection mode for training (pessimistic or mean)")
    parser.add_argument("--inference-mode", type=str, default="pessimistic", help="The action selection mode for inference (pessimistic or mean)")
    args = parser.parse_args()
    # === Logger ===
    logger = configure_logger('MC_BFTQ_train')
    tb_logger = TensorBoardLogger(log_dir="logs/tensorboard_mc")

    env = gym.make("merge-v0", render_mode="human")
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    config = {
        "gamma": 0.99,
        "batch_size": 16,
        "buffer_size": 50000,
        "learning_rate": 1e-3,
        "target_update": 100,
        "layers": [64, 64],
        "exploration": {"temperature": 1.0, "final_temperature": 0.1, "tau": 5000},
        "hull_options": dict(library="scipy", decimals=2, remove_duplicates=True),
        "k": 1.96, # Risk-aversion parameter
        "dropout_p": 0.5, # Dropout probability
        "n_samples": 10 # Number of samples for MC Dropout
    }

    state_dim = int(np.prod(env.observation_space.shape))
    agent = MCBFTQAgent(state_dim, n_actions, config, network=MCDropoutQNet, device="cpu", logger=logger, tb_logger=tb_logger)
    if hasattr(agent, "set_training_mode"):
        agent.set_training_mode(args.training_mode)

    n_episodes = 100
    for ep in range(n_episodes):
        state, _ = env.reset()
        done = False
        beta = np.random.uniform()
        total_reward = 0

        while not done:
            action, beta = agent.act(state, beta)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            cost = info.get("cost", 0)
            agent.push_transition(state, action, reward, cost, beta, next_state, done)
            agent.update()

            state = next_state
            total_reward += reward

            env.render()

        logger.info(f"Episode {ep}, total reward: {total_reward}")
        tb_logger.log_scalar('reward/total_reward', total_reward, ep)
        tb_logger.log_scalar('beta/beta', beta, ep)


    env.close()
    tb_logger.close()

    # Save the model
    agent.save_model("model_weights/mc_bftq_model.pt")

if __name__ == "__main__":
    main()
