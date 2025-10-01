import highway_env
import gymnasium as gym
import numpy as np
from agents.bftq.agent import BFTQAgent
from models.q_net import BudgetedQNet
from utils.logger import configure_logger, TensorBoardLogger


print([env_spec.id for env_spec in gym.envs.registry.values() if "highway" in env_spec.id])


def main():
    # === Logger ===
    logger = configure_logger('BFTQ_train')
    tb_logger = TensorBoardLogger()

    env = gym.make("highway-v0", render_mode="human")
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
    }

    state_dim = int(np.prod(env.observation_space.shape))
    agent = BFTQAgent(state_dim, n_actions, config, network=BudgetedQNet, device="cpu", logger=logger, tb_logger=tb_logger)

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

            cost = info.get("cost", 0)  # HighwayEnv may not have this; you can set to 0
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

if __name__ == "__main__":
    main()
