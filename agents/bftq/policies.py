import abc
import copy
import torch
import numpy as np

from agents.bftq.greedy_policy import optimal_mixture, pareto_frontier_at
# from rl_agents.agents.common.utils import sample_simplex

def sample_simplex(coeff, bias, min_x, max_x, np_random=np.random):
    """
    Sample from a simplex.

    The simplex is defined by:
        w.x + b <= 0
        x_min <= x <= x_max

    Warning: this is not uniform sampling.

    :param coeff: coefficient w
    :param bias: bias b
    :param min_x: lower bound on x
    :param max_x: upper bound on x
    :param np_random: source of randomness
    :return: a sample from the simplex
    """
    x = np.zeros(len(coeff))
    indexes = np.asarray(range(0, len(coeff)))
    np_random.shuffle(indexes)
    remain_indexes = np.copy(indexes)
    for i_index, index in enumerate(indexes):
        remain_indexes = remain_indexes[1:]
        current_coeff = np.take(coeff, remain_indexes)
        full_min = np.full(len(remain_indexes), min_x)
        full_max = np.full(len(remain_indexes), max_x)
        dot_max = np.dot(current_coeff, full_max)
        dot_min = np.dot(current_coeff, full_min)
        min_xi = (bias - dot_max) / coeff[index]
        max_xi = (bias - dot_min) / coeff[index]
        min_xi = np.max([min_xi, min_x])
        max_xi = np.min([max_xi, max_x])
        xi = min_xi + np_random.random_sample() * (max_xi - min_xi)
        bias = bias - xi * coeff[index]
        x[index] = xi
        if len(remain_indexes) == 1:
            break
    last_index = remain_indexes[0]
    x[last_index] = bias / coeff[last_index]
    return x


class BudgetedPolicy:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def execute(self, state, beta):
        pass


class EpsilonGreedyBudgetedPolicy(BudgetedPolicy):
    def __init__(self, pi_greedy, pi_random, config, np_random=np.random):
        super().__init__()
        self.pi_greedy = pi_greedy
        self.pi_random = pi_random
        self.config = config
        self.np_random = np_random
        self.time = 0

    def execute(self, state, beta):
        if self.config['tau'] == 0:
            epsilon = self.config['final_temperature']
        else:
            epsilon = self.config['final_temperature'] + (self.config['temperature'] - self.config['final_temperature']) * \
                       np.exp(- self.time / self.config['tau'])
        self.time += 1

        if self.np_random.random() > epsilon:
            return self.pi_greedy.execute(state, beta)
        else:
            return self.pi_random.execute(state, beta)

    def set_time(self, time):
        self.time = time


class RandomBudgetedPolicy(BudgetedPolicy):
    def __init__(self, n_actions, np_random=np.random):
        self.n_actions = n_actions
        self.np_random = np_random

    def execute(self, state, beta):
        action_probs = self.np_random.random(self.n_actions)
        action_probs /= np.sum(action_probs)
        budget_probs = sample_simplex(coeff=action_probs, bias=beta, min_x=0, max_x=1, np_random=self.np_random)
        action = self.np_random.choice(a=range(self.n_actions), p=action_probs)
        beta = budget_probs[action]
        return action, beta, 0, 0


class PytorchBudgetedFittedPolicy(BudgetedPolicy):
    def __init__(self, network, betas_for_discretisation, device, hull_options, clamp_qc=None, np_random=np.random):
        self.betas_for_discretisation = betas_for_discretisation
        self.device = device
        self.hull_options = hull_options
        self.clamp_qc = clamp_qc
        self.np_random = np_random
        self.network = network

    def load_network(self, network_path):
        self.network = torch.load(network_path, map_location=self.device)

    def set_network(self, network):
        self.network = copy.deepcopy(network)

    def execute(self, state, beta):
        mixture, _, q_r, q_c = self.greedy_policy(state, beta)
        choice = mixture.sup if self.np_random.uniform() < mixture.probability_sup else mixture.inf
        return choice.action, choice.budget, q_r, q_c

    def greedy_policy(self, state, beta):
        # print("DEBUG greedy_policy: state type =", type(state),
        #       "shape =", state.shape if hasattr(state, "shape") else "no shape")

        with torch.no_grad():
            # ensure state is a torch tensor of shape [1, state_dim]
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, device=self.device, dtype=torch.float32)
            state = state.view(1, -1)  # flatten and add batch dim

            hull = pareto_frontier_at(
                state=state,
                value_network=self.network,
                betas=self.betas_for_discretisation,
                device=self.device,
                hull_options=self.hull_options,
                clamp_qc=self.clamp_qc)

        mixture = optimal_mixture(hull[0], beta)
        # === Extract predicted cost/reward based on mixture ===
        inf_p = mixture.inf
        sup_p = mixture.sup
        p = mixture.probability_sup

        # Handle degenerate and edge cases gracefully
        if mixture.status == "regular":
            pred_qr = (1 - p) * inf_p.qr + p * sup_p.qr
            pred_qc = (1 - p) * inf_p.qc + p * sup_p.qc
        else:
            # when too_much_budget / too_little_budget / not_solvable
            pred_qr = inf_p.qr
            pred_qc = inf_p.qc

        # Return mixture, hull, and predicted values
        return mixture, hull, pred_qr, pred_qc

def test():
    class DummyGreedyPolicy:
        def execute(self, state, beta):
            return 1, beta   # always action 1

    pi_greedy = DummyGreedyPolicy()
    pi_random = RandomBudgetedPolicy(n_actions=3)

    config = dict(temperature=1.0, final_temperature=0.1, tau=50)
    policy = EpsilonGreedyBudgetedPolicy(pi_greedy, pi_random, config)

    # Try different timesteps
    for t in range(25):
        policy.set_time(t)
        action, beta = policy.execute(state=None, beta=0.5)
        print(f"Time {t}: action={action}, beta={beta}")
