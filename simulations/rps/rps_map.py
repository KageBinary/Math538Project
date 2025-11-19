import numpy as np


def rps_map_step(x: np.ndarray, h: float) -> np.ndarray:
    """
    One step of the discrete RPS map.
    x = [xR, xP, xS] (must sum to 1)
    h = step size (like the parameter in the logistic map)
    """
    xR, xP, xS = x

    # Replicator-type discrete update (Euler discretization of replicator dynamics)
    dR = xR * (xS - xP)
    dP = xP * (xR - xS)
    dS = xS * (xP - xR)

    # Forward-Euler map step
    x_next = np.array([
        xR + h * dR,
        xP + h * dP,
        xS + h * dS,
    ])

    # Renormalize to stay on simplex (numerical safety)
    total = x_next.sum()
    if total <= 0:
        # If something pathological happens, fall back to uniform
        return np.array([1/3, 1/3, 1/3])
    x_next /= total

    return x_next


def iterate_rps_map(x0: np.ndarray, h: float, steps: int) -> np.ndarray:
    """
    Iterate the RPS map for 'steps' iterations.
    x0: initial [xR, xP, xS], should sum to 1
    """
    xs = np.zeros((steps, 3), dtype=float)
    xs[0] = x0

    for n in range(steps - 1):
        xs[n + 1] = rps_map_step(xs[n], h)

    return xs


def plot_rps_map(xs: np.ndarray, title: str = "RPS Deterministic Map Trajectory"):
    """
    Quick plot of the discrete map's trajectory in the (xR, xP) plane.
    """
    import matplotlib.pyplot as plt

    xR = xs[:, 0]
    xP = xs[:, 1]

    plt.figure()
    plt.plot(xR, xP, marker=".", linewidth=0.8, markersize=2)
    plt.xlabel("x_R")
    plt.ylabel("x_P")
    plt.title(title)
    plt.tight_layout()
    plt.show()
