import torch

from utils import *

class BFTQAgent:
    def __init__(self, q_model, optimizer, gamma, epsilon_schedule, budget_space):
        self.q_model = q_model
        self.optimizer = optimizer
        self.gamma = gamma
        self.epsilon_schedule = epsilon_schedule
        self.budget_space = budget_space
        self.steps = 0

    def default_config(cls):
        return {
            "gamma": 0.9,
            "gamma_c": 0.9,
            "epochs": None,
            "delta_stop": 0.,
            "memory_capacity": 10000,
            "beta": 0,
            "betas_for_duplication": "np.arange(0, 1, 0.1)",
            "betas_for_discretisation": "np.arange(0, 1, 0.1)",
            "exploration": {
                "temperature": 1.0,
                "final_temperature": 0.1,
                "tau": 5000
            },
            "optimizer": {
                "type": "ADAM",
                "learning_rate": 1e-3,
                "weight_decay": 1e-3
            },
            "loss_function": "l2",
            "loss_function_c": "l2",
            "regression_epochs": 500,
            "clamp_qc": None,
            "nn_loss_stop_condition": 0.0,
            "weights_losses": [1., 1.],
            "split_batches": 1,
            "processes": 1,
            "samples_per_batch": 500,
            "device": "cuda:best",
            "hull_options": {
                "decimals": None,
                "qhull_options": "",
                "remove_duplicates": False,
                "library": "scipy"
            },
            "reset_network_each_epoch": True,
            "network": {
                "beta_encoder_type": "LINEAR",
                "size_beta_encoder": 10,
                "activation_type": "RELU",
                "reset_type": "XAVIER",
                "layers": [
                    64,
                    64
                ]
            }
        }

    def act(self, state):
        """
        Choose action using exploration_technique
        """
        #TODO: call exploration utils
        pass

    def update(self, batch):
        """
        Perform BFTQ update
        batch = (s, a, r, c, budget, s')
        """
        #TODO: Implement Budgeted Bellman backup

    def save(self, path):
        torch.save(self.q_model.state_dict(), path)

    def load(self, path):
        self.q_model.load_state_dict(torch.load(path))