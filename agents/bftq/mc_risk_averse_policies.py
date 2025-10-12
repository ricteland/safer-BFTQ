from collections import namedtuple

import numpy as np
import torch

from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError

from agents.bftq.convex_hull_graham import convex_hull_graham
from agents.bftq.policies import PytorchBudgetedFittedPolicy, BudgetedPolicy
from agents.bftq.greedy_policy import optimal_mixture, ValuePoint


class MCPessimisticPytorchBudgetedFittedPolicy(PytorchBudgetedFittedPolicy):
    def __init__(self, network, betas_for_discretisation, device, hull_options, k, clamp_qc=None, np_random=np.random, training_mode="pessimistic"):
        super().__init__(network, betas_for_discretisation, device, hull_options, clamp_qc, np_random)
        self.k = k
        self.training_mode = training_mode

    def greedy_policy(self, state, beta):
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, device=self.device, dtype=torch.float32)
            state = state.view(1, -1)

            hull = pessimistic_pareto_frontier_at(
                state=state,
                value_network=self.network,
                betas=self.betas_for_discretisation,
                device=self.device,
                hull_options=self.hull_options,
                k=self.k if self.training_mode == "pessimistic" else 0,
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


def pessimistic_pareto_frontier_at(state, value_network, betas, device, hull_options, k, clamp_qc=None):
    with torch.no_grad():
        if state.ndim == 1:
            state = state.unsqueeze(0)
        elif state.ndim == 2 and state.size(0) == 1:
            pass
        else:
            raise ValueError(f"Unexpected state shape {state.shape}")

        ss = state.repeat(len(betas), 1)
        bb = torch.as_tensor(betas, dtype=torch.float32, device=device).unsqueeze(1)

        (q_r_mean, _), (q_c_mean, q_c_std) = value_network.predict_dist(ss, bb)
        
        q_r_mean = q_r_mean.detach().cpu().numpy()
        q_c_mean = q_c_mean.detach().cpu().numpy()
        q_c_std = q_c_std.detach().cpu().numpy()
    return pessimistic_pareto_frontier(q_r_mean, q_c_mean, q_c_std, betas, hull_options, k, clamp_qc)


def pessimistic_pareto_frontier(q_r_mean, q_c_mean, q_c_std, betas, hull_options, k, clamp_qc=None):
    n_actions = q_r_mean.shape[1]
    
    pessimistic_q_c = q_c_mean + k * q_c_std

    if clamp_qc is not None:
        pessimistic_q_c = np.clip(pessimistic_q_c, a_min=clamp_qc[0], a_max=clamp_qc[1])

    all_points = [ValuePoint(action=i_a, budget=beta, qc=pessimistic_q_c[i_b][i_a], qr=q_r_mean[i_b][i_a])
                  for i_b, beta in enumerate(betas) for i_a in range(n_actions)]
    
    max_point = max(all_points, key=lambda p: p.qr)
    points = [point for point in all_points if point.qc <= max_point.qc]

    point_values = np.array([[point.qc, point.qr] for point in points])
    if hull_options["decimals"]:
        point_values = np.round(point_values, decimals=hull_options["decimals"])
    if hull_options["remove_duplicates"]:
        point_values, indices = np.unique(point_values, axis=0, return_index=True)
        points = [points[i] for i in indices]

    colinearity = False
    vertices = []
    if len(points) >= 3:
        if hull_options["library"] == "scipy":
            try:
                hull = ConvexHull(point_values, qhull_options=hull_options.get("qhull_options", ""))
                vertices = hull.vertices
            except QhullError:
                colinearity = True
        elif hull_options["library"] == "pure_python":
            assert hull_options["remove_duplicates"]
            hull = convex_hull_graham(point_values.tolist())
            vertices = np.array([np.where(np.all(point_values == vertex, axis=1)) for vertex in hull]).squeeze()
    else:
        colinearity = True

    if not colinearity:
        points_v = [points[i] for i in vertices]
        point_max_qr = max(points_v, key=lambda p: p.qr)
        point_max_qr_min_qc = min([p for p in points_v if p.qr == point_max_qr.qr], key=lambda p: p.qr)
        start = points_v.index(point_max_qr_min_qc)
        top_points = []
        for i in range(len(vertices)):
            top_points.append(points_v[(start + i) % len(vertices)])
            if points_v[(start + i + 1) % len(vertices)].qc >= points_v[(start + i) % len(vertices)].qc:
                break
    else:
        top_points = points

    top_points = sorted(top_points, key=lambda p: p.qc) if colinearity else list(reversed(top_points))
    return top_points, all_points
