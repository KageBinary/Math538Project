# rps_bifurcation.py
import numpy as np
import matplotlib.pyplot as plt
from simulations.rps.rps_map import rps_map_step


def rps_bifurcation_diagram(
    h_min: float = 0.0,
    h_max: float = 0.6,
    num_h: int = 300,
    steps_total: int = 5000,
    steps_discard: int = 2000,
    coord: str = "R",
    show: bool = True,
):
    """
    Plot a bifurcation-like diagram for the RPS map.

    Parameters
    ----------
    h_min, h_max : float
        Range of step sizes h to scan.
    num_h : int
        Number of h values between h_min and h_max.
    steps_total : int
        Total number of iterations per h.
    steps_discard : int
        Number of initial iterations to discard (burn-in).
    coord : {"R", "P", "S"}
        Which coordinate to plot vs h (x_R, x_P, or x_S).
    show : bool
        If True, call plt.show() at the end. If False, leave it to caller.
    """
    coord = coord.upper()
    idx_map = {"R": 0, "P": 1, "S": 2}
    if coord not in idx_map:
        raise ValueError("coord must be one of 'R', 'P', 'S'")

    idx = idx_map[coord]

    hs_all = []
    x_all = []

    hs = np.linspace(h_min, h_max, num_h)

    for h in hs:
        x = np.array([0.4, 0.3, 0.3], dtype=float)

        # Burn-in
        for _ in range(steps_discard):
            x = rps_map_step(x, h)

        # Collect long-term points
        for _ in range(steps_total - steps_discard):
            x = rps_map_step(x, h)
            hs_all.append(h)
            x_all.append(x[idx])

    hs_all = np.array(hs_all)
    x_all = np.array(x_all)

    plt.figure(figsize=(6, 4))
    plt.plot(hs_all, x_all, ".", markersize=0.5)
    plt.xlabel("h (step size)")
    plt.ylabel(f"x_{coord} (long-term)")
    plt.title(f"Bifurcation-like Diagram for RPS Map (x_{coord})")
    plt.tight_layout()

    if show:
        plt.show()
