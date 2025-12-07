# simulations/competitive_pops/competitive_pops.py
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field


@dataclass
class CompetitivePopsParams:
    # Resource parameters
    r_R: float = 2.5      # intrinsic growth rate of resource
    K_R: float = 1.0      # carrying capacity of resource
    gamma: float = 0.5    # resource depletion by consumers

    # Species parameters (symmetric)
    a: float = 1.0        # benefit from resource
    d: float = 0.4        # baseline mortality
    b: float = 0.3        # self-competition (within species)
    c: float = 0.1        # competition from other species


def competitive_step(
    state: np.ndarray,
    params: CompetitivePopsParams,
) -> np.ndarray:
    """
    One step of the discrete competitive populations map.

    state = [R, X, Y, Z] (all >= 0)
    """
    R, X, Y, Z = state
    p = params

    # Resource update: logistic growth minus consumption
    cons = X + Y + Z
    R_next = R + p.r_R * R * (1 - R / p.K_R) - p.gamma * R * cons
    R_next = max(min(R_next, p.K_R), 0.0)
    # Growth term shared by all species
    growth = p.a * R
    # X
    comp_X = p.b * X + p.c * (Y + Z)
    X_next = X + X * (growth - p.d - comp_X)
    # Y
    comp_Y = p.b * Y + p.c * (X + Z)
    Y_next = Y + Y * (growth - p.d - comp_Y)
    # Z
    comp_Z = p.b * Z + p.c * (X + Y)
    Z_next = Z + Z * (growth - p.d - comp_Z)

    next_state = np.array([R_next, X_next, Y_next, Z_next], dtype=float)
    # keep nonnegative (no negative populations)
    next_state = np.maximum(next_state, 0.0)
    return next_state


def iterate_competitive_map(
    x0: np.ndarray,
    params: CompetitivePopsParams,
    steps: int,
) -> np.ndarray:
    """
    Iterate the competitive populations map for a given number of steps.

    Returns an array of shape (steps+1, 4) with [R, X, Y, Z] over time.
    """
    xs = np.empty((steps + 1, 4), dtype=float)
    xs[0] = x0

    x = x0.copy()
    for n in range(steps):
        x = competitive_step(x, params)
        xs[n + 1] = x

    return xs

@dataclass
class CompetitivePopsConfig:
    steps: int = 5000
    R0: float = 0.8
    X0: float = 0.2
    Y0: float = 0.25
    Z0: float = 0.15
    params: CompetitivePopsParams = field(default_factory=CompetitivePopsParams)


class CompetitivePopsSimulation:
    def __init__(self, config: CompetitivePopsConfig):
        self.config = config
        self.trajectory: np.ndarray | None = None

    def run(self):
        x0 = np.array(
            [self.config.R0, self.config.X0,
             self.config.Y0, self.config.Z0],
            dtype=float,
        )
        self.trajectory = iterate_competitive_map(
            x0, self.config.params, self.config.steps
        )

    # ---------- Plotting helpers ----------

    def plot_time_series(self, show: bool = True):
        if self.trajectory is None:
            raise RuntimeError("Run the simulation first.")

        t = np.arange(self.trajectory.shape[0])
        R = self.trajectory[:, 0]
        X = self.trajectory[:, 1]
        Y = self.trajectory[:, 2]
        Z = self.trajectory[:, 3]

        plt.figure()
        plt.plot(t, R, label="Resource R")
        plt.plot(t, X, label="Species X")
        plt.plot(t, Y, label="Species Y")
        plt.plot(t, Z, label="Species Z")
        plt.xlabel("n (time step)")
        plt.ylabel("Population / resource level")
        plt.title("Competitive Populations: Time Series")
        plt.legend()
        plt.tight_layout()
        if show:
            plt.show()

    def plot_phase_XY(self, show: bool = True):
        if self.trajectory is None:
            raise RuntimeError("Run the simulation first.")

        X = self.trajectory[:, 1]
        Y = self.trajectory[:, 2]

        plt.figure()
        plt.plot(X, Y, marker=".", linewidth=0.8, markersize=2)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Phase Plot: Species X vs Y")
        plt.tight_layout()
        if show:
            plt.show()
