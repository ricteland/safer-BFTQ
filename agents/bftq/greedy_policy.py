from collections import namedtuple

import numpy as np
import torch

from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError

from agents.bftq.convex_hull_graham import convex_hull_graham

TransitionBFTQ = namedtuple('TransitionBFTQ', ('state', 'action', 'reward', 'next_state', 'terminal', 'cost', 'beta'))
ValuePoint = namedtuple('HullPoint', ('action', 'budget', 'qc', 'qr'))
Mixture = namedtuple('Mixture', ('inf', 'sup', 'probability_sup', 'status'))


def optimal_mixture(frontier, beta):
    """
        Find the mixture policy with maximum rewards and expected cost under beta.

        1. Pick points such that: H[k−1].qc <= beta < H[k].qc
        2. Mix with probability: p = (beta − H[k−1].qc)/(H[k].qc − H[k−1].qc)

    :param frontier: the pareto-optimal frontier F of points (qc, qr) at a given state, across all actions & budgets
    :param beta: a desired cost budget
    :return: the mixture policy
    """
    if not frontier:
        raise Exception("Frontier is empty")
    for inf, sup in zip(frontier[:-1], frontier[1:]):
        if inf.qc <= beta < sup.qc:
            return Mixture(inf=inf, sup=sup, probability_sup=(beta - inf.qc) / (sup.qc - inf.qc), status="regular")
    else:
        if beta < frontier[0].qc:
            return Mixture(inf=frontier[0], sup=frontier[0], probability_sup=0, status="not_solvable")
        else:
            return Mixture(inf=frontier[-1], sup=frontier[-1], probability_sup=1, status="too_much_budget")


def pareto_frontier(values, betas, hull_options, clamp_qc=None):
    """
        Compute Pareto frontier F of {(qc, qr)} at a state s, generated across all actions and betas

        It is the top frontier of the convex hull of undominated points.
    :param values: an array of (qr, qc) values at a state s, of shape [betas x (actions x 2)]
    :param betas: the list of budgets corresponding to these values
    :param hull_options: options for convex hull computation
    :param clamp_qc: option to clamp qc in frontier computation
    :return: the frontier, all points
    """
    # Clamp Qc
    n_actions = values.shape[1] // 2
    if clamp_qc is not None:
        values[:, n_actions:] = np.clip(values[:, n_actions:], a_min=clamp_qc[0], a_max=clamp_qc[1])

    # Q-: Filter out dominated points Q+
    all_points = [ValuePoint(action=i_a, budget=beta, qc=values[i_b][i_a + n_actions], qr=values[i_b][i_a])
                  for i_b, beta in enumerate(betas) for i_a in range(n_actions)]
    max_point = max(all_points, key=lambda p: p.qr)
    points = [point for point in all_points if point.qc <= max_point.qc]

    # Round and remove duplicates
    point_values = np.array([[point.qc, point.qr] for point in points])
    if hull_options["decimals"]:
        point_values = np.round(points, decimals=hull_options["decimals"])
    if hull_options["remove_duplicates"]:
        point_values, indices = np.unique(point_values, axis=0, return_index=True)
        points = [points[i] for i in indices]

    # Compute convex hull C(Q-)
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
            assert hull_options["remove_duplicated_points"]
            hull = convex_hull_graham(point_values.tolist())
            vertices = np.array([np.where(np.all(point_values == vertex, axis=1)) for vertex in hull]).squeeze()
    else:
        colinearity = True

    # Filter top part of the convex hull: F = top(C(Q-))
    if not colinearity:
        # Start at point with max qr but min qc
        points_v = [points[i] for i in vertices]
        point_max_qr = max(points_v, key=lambda p: p.qr)
        point_max_qr_min_qc = min([p for p in points_v if p.qr == point_max_qr.qr], key=lambda p: p.qr)
        start = points_v.index(point_max_qr_min_qc)
        # Continue until qc stops decreasing (vertices are in CCW order)
        top_points = []
        for k in range(len(vertices)):
            top_points.append(points_v[(start + k) % len(vertices)])
            if points_v[(start + k + 1) % len(vertices)].qc >= points_v[(start + k) % len(vertices)].qc:
                break
    else:
        top_points = points

    top_points = sorted(top_points, key=lambda p: p.qc) if colinearity else list(reversed(top_points))
    return top_points, all_points


def pareto_frontier_at(state, value_network, betas, device, hull_options, clamp_qc=None):
    with torch.no_grad():
        if state.ndim == 1:
            state = state.unsqueeze(0)
        elif state.ndim == 2 and state.size(0) == 1:
            pass
        else:
            raise ValueError(f"Unexpected state shape {state.shape}")

        # Repeat state for each beta
        ss = state.repeat(len(betas), 1)                  # [num_betas, state_dim]
        bb = torch.as_tensor(betas, dtype=torch.float32, device=device).unsqueeze(1)  # [num_betas, 1]

        # Forward pass
        q_r, q_c = value_network(ss, bb)
        values = torch.cat([q_r, q_c], dim=1).detach().cpu().numpy()

    return pareto_frontier(values, betas, hull_options=hull_options, clamp_qc=clamp_qc)



