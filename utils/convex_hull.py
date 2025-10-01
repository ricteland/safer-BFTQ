from functools import reduce


def convex_hull_graham(points):
    '''
    Returns points on convex hull in CCW order according to Graham's scan algorithm.
    By Tom Switzer <thomas.switzer@gmail.com>.
    '''
    TURN_LEFT, TURN_RIGHT, TURN_NONE = (1, -1, 0)

    def cmp(a, b):
        return (a > b) - (a < b)

    def turn(p, q, r):
        return cmp((q[0] - p[0])*(r[1] - p[1]) - (r[0] - p[0])*(q[1] - p[1]), 0)

    def _keep_left(hull, r):
        while len(hull) > 1 and turn(hull[-2], hull[-1], r) != TURN_LEFT:
            hull.pop()
        if not len(hull) or hull[-1] != r:
            hull.append(r)
        return hull

    points = sorted(points)
    l = reduce(_keep_left, points, [])
    u = reduce(_keep_left, reversed(points), [])
    return l.extend(u[i] for i in range(1, len(u) - 1)) or l


def test():
    import matplotlib.pyplot as plt
    # ==== Define some points (random + square corners) ====
    points = [
        (0, 0), (1, 0), (0, 1), (1, 1),  # square corners
        (0.5, 0.5), (0.2, 0.3), (0.8, 0.4)  # interior points
    ]

    print("Input points:", points)

    # ==== Compute convex hull ====
    hull = convex_hull_graham(points)
    print("Convex hull:", hull)

    # ==== Plot ====
    xs, ys = zip(*points)
    plt.scatter(xs, ys, label="Points", color="blue")

    # Close the hull for plotting
    hx, hy = zip(*(hull + [hull[0]]))
    plt.plot(hx, hy, label="Convex Hull", color="red")

    plt.legend()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Convex Hull Test")
    plt.show()

if __name__ == "__main__":
    test()