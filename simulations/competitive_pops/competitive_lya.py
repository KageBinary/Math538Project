# simulations/competitive_pops/competitive_lya.py
import numpy as np
from .competitive_pops import competitive_step, CompetitivePopsParams

def trajectory_ok(
    params: CompetitivePopsParams,
    steps: int = 30_000,
    discard: int = 5_000,
    max_bound: float = 5.0,
    min_species: float = 1e-4,
) -> bool:
    """
    Check that the trajectory is bounded and not trivially extinct.

    - Reject if any component exceeds max_bound (blow-up).
    - Reject if all three species become ~0 (extinction).
    """
    x = np.array([0.8, 0.2, 0.25, 0.15], dtype=float)

    extinct_counter = 0

    for n in range(steps):
        x = competitive_step(x, params)

        if not np.isfinite(x).all():
            return False

        if np.any(x > max_bound):
            return False  # blow-up

        if n >= discard:
            # after burn-in, check if all species are basically gone
            X, Y, Z = x[1], x[2], x[3]
            if X < min_species and Y < min_species and Z < min_species:
                extinct_counter += 1
                # if extinct for many consecutive steps, call it extinct
                if extinct_counter > 500:
                    return False

    return True

def competitive_lyapunov(
    params: CompetitivePopsParams,
    steps: int = 20_000,
    discard: int = 2_000,
    eps: float = 1e-8,
    max_bound: float = 5.0,
) -> float:
    """
    Estimate maximal Lyapunov exponent for the 4D competitive pops map.
    Returns NaN if the trajectory blows up.
    """
    x = np.array([0.8, 0.2, 0.25, 0.15], dtype=float)

    rng = np.random.default_rng(0)
    v = rng.normal(size=4)
    v /= np.linalg.norm(v)
    y = x + eps * v
    y = np.maximum(y, 0.0)

    lsum = 0.0
    count = 0

    for n in range(steps):
        x = competitive_step(x, params)
        y = competitive_step(y, params)

        if (not np.isfinite(x).all()) or np.any(np.abs(x) > max_bound):
            # treat as divergence
            return float("nan")

        if n < discard:
            continue

        diff = y - x
        dist = np.linalg.norm(diff)
        if dist <= 1e-16:
            v = rng.normal(size=4)
            v /= np.linalg.norm(v)
            y = x + eps * v
            y = np.maximum(y, 0.0)
            continue

        lsum += np.log(dist / eps)
        count += 1

        diff = diff * (eps / dist)
        y = x + diff
        y = np.maximum(y, 0.0)

    return lsum / count if count > 0 else float("nan")

def random_search_for_chaos(
    num_samples: int = 50,
    lambda_threshold: float = 0.05,
) -> list[tuple[CompetitivePopsParams, float]]:
    rng = np.random.default_rng()
    chaotic_params: list[tuple[CompetitivePopsParams, float]] = []

    for i in range(num_samples):
        params = CompetitivePopsParams(
            r_R=rng.uniform(1.5, 3.5),
            K_R=1.0,
            gamma=rng.uniform(0.1, 0.7),
            a=rng.uniform(0.5, 1.8),
            d=rng.uniform(0.05, 0.4),
            b=rng.uniform(0.05, 0.4),
            c=rng.uniform(0.0, 0.3),
        )

        # First, reject clearly bad trajectories
        if not trajectory_ok(params):
            print(f"[{i+1}/{num_samples}] params rejected (blow-up/extinction).")
            continue

        lam = competitive_lyapunov(params, max_bound=5.0)
        print(f"[{i+1}/{num_samples}] λ = {lam:.4f} for params = {params}")

        if np.isnan(lam):
            continue

        if lam > lambda_threshold:
            print("  → candidate BOUNDED CHAOTIC parameter set found!\n")
            chaotic_params.append((params, lam))

    if not chaotic_params:
        print("\nNo bounded chaotic parameter sets found in this batch.")
    else:
        print("\nSummary of candidate bounded chaotic parameter sets:")
        for params, lam in chaotic_params:
            print(f"  λ = {lam:.4f}, params = {params}")

    return chaotic_params