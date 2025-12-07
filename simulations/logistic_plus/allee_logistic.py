import numpy as np
import matplotlib.pyplot as plt


def allee_step_piecewise(x: float, r: float, A: float, s: float) -> float:
    """
    Piecewise logistic map with a strong Allee effect:

        if x < A:   x_{n+1} = s * x_n     (extinction region, 0 < s < 1)
        if x >= A:  x_{n+1} = r * x_n (1 - x_n)   (standard logistic)

    Here:
      - r in (0, 4] gives the usual logistic behaviour above threshold
      - A is the Allee threshold (0 < A < 1)
      - s in (0, 1) controls how fast populations below A die out
    """
    if x < A:
        x_next = s * x
    else:
        x_next = r * x * (1.0 - x)

    # keep in [0, 1] for numerical safety
    if not np.isfinite(x_next):
        return 0.0
    return np.clip(x_next, 0.0, 1.0)


def iterate_allee_piecewise(x0: float, r: float, A: float, s: float, steps: int) -> np.ndarray:
    xs = np.empty(steps + 1)
    xs[0] = x0
    x = x0
    for n in range(steps):
        x = allee_step_piecewise(x, r, A, s)
        xs[n + 1] = x
    return xs


def plot_allee_piecewise_time_series(x0=0.8, r=3.8, A=0.2, s=0.5, steps=200):
    xs = iterate_allee_piecewise(x0, r, A, s, steps)
    t = np.arange(steps + 1)

    plt.figure()
    plt.plot(t, xs, "-o", markersize=2)
    plt.axhline(A, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("n")
    plt.ylabel("x_n")
    plt.title(f"Piecewise Allee Logistic: r={r}, A={A}, s={s}, x0={x0}")
    plt.tight_layout()
    plt.show()
