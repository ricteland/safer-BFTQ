import matplotlib.pyplot as plt
from agents.bftq.convex_hull_graham import convex_hull_graham

def plot_convex_hull_from_qvalues(q_r, q_c, beta=None, chosen_idx=None):
    """
    Plot the convex hull in cost-reward space given Q-values.

    Args:
        q_r (tensor or array): [n_actions] rewards
        q_c (tensor or array): [n_actions] costs
        beta (float, optional): budget to visualize
        chosen_idx (int, optional): index of chosen action
    """
    # Convert to list of (cost, reward) pairs
    points = [(float(c), float(r)) for r, c in zip(q_r, q_c)]

    # Compute convex hull
    hull = convex_hull_graham(points)

    # Plot all actions
    xs, ys = zip(*points)
    plt.scatter(xs, ys, color="blue", label="Actions")

    # Plot convex hull polygon
    hx, hy = zip(*(hull + [hull[0]]))  # close polygon
    plt.plot(hx, hy, color="red", label="Convex hull")

    # Highlight chosen action
    if chosen_idx is not None:
        c, r = points[chosen_idx]
        plt.scatter([c], [r], color="green", s=100, marker="*", label="Chosen action")

    # Show budget line
    if beta is not None:
        plt.axvline(x=beta, color="gray", linestyle="--", label=f"Budget β={beta:.2f}")

    plt.xlabel("Cost")
    plt.ylabel("Reward")
    plt.title("Q-value Frontier")
    plt.legend()
    plt.show()


def plot_training_curves(rewards, costs):
    plt.figure()
    plt.plot(rewards, label="Reward")
    plt.plot(costs, label="Cost")
    plt.xlabel("Episode")
    plt.legend()
    plt.show()


# import torch
# from models.q_net import BudgetedQNet
#
# # Dummy network
# net = BudgetedQNet(size_state=5, layers=[32, 32], n_actions=6)
# state = torch.randn(1, 5)
# beta = torch.tensor([[0.5]])
# q_r, q_c = net(state, beta)
#
# print("Q_r:", q_r)
# print("Q_c:", q_c)
#
# # Visualize
# plot_convex_hull_from_qvalues(q_r[0], q_c[0], beta=0.5, chosen_idx=1)
