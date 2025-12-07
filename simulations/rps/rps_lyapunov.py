import numpy as np
import matplotlib.pyplot as plt
from simulations.rps.rps_map import rps_map_step

def rps_lyapunov(h: float,
                 steps: int = 20_000,
                 discard: int = 2_000,
                 eps: float = 1e-8) -> float:
    """
    Estimate the maximal Lyapunov exponent for the RPS map at step size h.
    """
    # Start with some interior point on the simplex
    x = np.array([0.4, 0.3, 0.3], dtype=float)
    # Nearby initial condition (still sums to 1)
    y = x + np.array([eps, 0.0, -eps], dtype=float)

    lsum = 0.0
    count = 0

    for n in range(steps):
        x = rps_map_step(x, h)
        y = rps_map_step(y, h)

        if n < discard:
            continue

        diff = y - x
        dist = np.linalg.norm(diff[:2])  # 2D projection is enough

        if dist <= 1e-16:
            # trajectories collapsed; restart perturbation
            y = x + np.array([eps, 0.0, -eps])
            continue

        # Accumulate log stretching factor
        lsum += np.log(dist / eps)
        count += 1

        # Renormalize separation back to eps
        diff = diff * (eps / dist)
        y = x + diff
        # keep on simplex
        y[2] = 1.0 - y[0] - y[1]

    return lsum / count if count > 0 else np.nan


def scan_lyapunov(h_min=0.0, h_max=0.6, num_h=60, show=True):
    hs = np.linspace(h_min, h_max, num_h)
    lambdas = []

    for h in hs:
        lam = rps_lyapunov(h)
        lambdas.append(lam)
        print(f"h={h:.3f}, lambda={lam:.4f}")

    hs = np.array(hs)
    lambdas = np.array(lambdas)

    plt.figure()
    plt.plot(hs, lambdas, marker=".")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("h (step size)")
    plt.ylabel("Max Lyapunov exponent λ(h)")
    plt.title("Lyapunov Exponents for RPS Map")
    plt.tight_layout()

    if show:
        plt.show()

    return hs, lambdas