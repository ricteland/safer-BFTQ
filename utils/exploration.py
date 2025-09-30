from gymnasium import spaces
import numpy as np
from gymnasium.utils import seeding
from abc import abstractmethod, ABC

from rl_agents.configuration import Configurable


class DiscreteDistribution(Configurable, ABC):
    def __init__(self, config=None, **kwargs):
        super(DiscreteDistribution, self).__init__(config)
        self.np_random = None

    @abstractmethod
    def get_distribution(self):
        """
        :return: a distribution over actions {action:probability}
        """
        raise NotImplementedError()

    def sample(self):
        """
        :return: an action sampled from the distribution
        """
        distribution = self.get_distribution()
        return self.np_random.choice(list(distribution.keys()), 1, p=np.array(list(distribution.values())))[0]

    def seed(self, seed=None):
        """
            Seed the policy randomness source
        :param seed: the seed to be used
        :return: the used seed
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_time(self, time):
        """ Set the local time, allowing to schedule the distribution temperature. """
        pass

    def step_time(self):
        """ Step the local time, allowing to schedule the distribution temperature. """
        pass


def exploration_factory(exploration_config, action_space):
    """
        Handles creation of exploration policies
    :param exploration_config: configuration dictionary of the policy, must contain a "method" key
    :param action_space: the environment action space
    :return: a new exploration policy
    """
    from rl_agents.agents.common.exploration.boltzmann import Boltzmann
    from rl_agents.agents.common.exploration.epsilon_greedy import EpsilonGreedy
    from rl_agents.agents.common.exploration.greedy import Greedy

    if exploration_config['method'] == 'Greedy':
        return Greedy(action_space, exploration_config)
    elif exploration_config['method'] == 'EpsilonGreedy':
        return EpsilonGreedy(action_space, exploration_config)
    elif exploration_config['method'] == 'Boltzmann':
        return Boltzmann(action_space, exploration_config)
    else:
        raise ValueError("Unknown exploration method")

class EpsilonGreedy(DiscreteDistribution):
    """
        Uniform distribution with probability epsilon, and optimal action with probability 1-epsilon
    """

    def __init__(self, action_space, config=None):
        super(EpsilonGreedy, self).__init__(config)
        self.action_space = action_space
        if isinstance(self.action_space, spaces.Tuple):
            self.action_space = self.action_space.spaces[0]
        if not isinstance(self.action_space, spaces.Discrete):
            raise TypeError("The action space should be discrete")
        self.config['final_temperature'] = min(self.config['temperature'], self.config['final_temperature'])
        self.optimal_action = None
        self.epsilon = 0
        self.time = 0
        self.writer = None
        self.seed()

    @classmethod
    def default_config(cls):
        return dict(temperature=1.0,
                    final_temperature=0.1,
                    tau=5000)

    def get_distribution(self):
        distribution = {action: self.epsilon / self.action_space.n for action in range(self.action_space.n)}
        distribution[self.optimal_action] += 1 - self.epsilon
        return distribution

    def update(self, values):
        """
            Update the action distribution parameters
        :param values: the state-action values
        :param step_time: whether to update epsilon schedule
        """
        self.optimal_action = np.argmax(values)
        self.epsilon = self.config['final_temperature'] + \
            (self.config['temperature'] - self.config['final_temperature']) * \
            np.exp(- self.time / self.config['tau'])
        if self.writer:
            self.writer.add_scalar('exploration/epsilon', self.epsilon, self.time)

    def step_time(self):
        self.time += 1

    def set_time(self, time):
        self.time = time

    def set_writer(self, writer):
        self.writer = writer