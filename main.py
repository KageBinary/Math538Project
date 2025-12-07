import numpy as np
import matplotlib.pyplot as plt

from simulations.rps.rps_sim import RPSSimulation, RPSSimulationConfig
from simulations.rps.rps_map import iterate_rps_map, plot_rps_map
from simulations.rps.rps_bifurcation import rps_bifurcation_diagram
from simulations.rps.rps_lyapunov import scan_lyapunov
from simulations.competitive_pops.competitive_pops import (
    CompetitivePopsConfig,
    CompetitivePopsSimulation,
    CompetitivePopsParams,
)
from simulations.logistic_plus.allee_logistic import plot_allee_piecewise_time_series


def run_allee_map():
    print("\n=== Allee Logistic Map ===")
    r = float(input("Enter r (e.g., 3.0): ") or 3.0)
    A = float(input("Enter threshold A (e.g., 0.2): ") or 0.2)
    plot_allee_piecewise_time_series(x0=0.3, r=r, A=A, steps=300)


def run_competitive_pops():
    import matplotlib.pyplot as plt
    plt.close("all")
    print("\n=== Competitive Populations (Discrete Map) ===")

    # You can later ask user for parameters; for now just use defaults.
    params = CompetitivePopsParams(
        r_R=1.9836167132538598,
        K_R=1.0,
        gamma=0.3907389034212142,
        a=0.7224142677976857,
        d=0.3391697312887608,
        b=0.27240184065805156,
        c=0.10671376447711826
    )
    config = CompetitivePopsConfig(
        steps=5000,
        R0=0.8,
        X0=0.2,
        Y0=0.25,
        Z0=0.15,
        params=params,
    )
    sim = CompetitivePopsSimulation(config)
    sim.run()

    sim.plot_time_series(show=False)
    sim.plot_phase_XY(show=False)
    plt.show()

    print("Competitive populations simulation complete.\n")


def run_rps(seed_input: str):
    """Run stochastic RPS sim + deterministic map comparison."""
    plt.close("all")
    print("\n=== Rock–Paper–Scissors Simulation ===")

    # Seed handling
    if seed_input.lower() == "r":
        seed_in = np.random.randint(0, 1_000_000)
    elif seed_input == "":
        seed_in = None
    else:
        seed_in = int(seed_input)

    # Stochastic simulation config
    config = RPSSimulationConfig(seed=seed_in)
    sim = RPSSimulation(config)
    sim.run()

    # Stochastic plots
    sim.plot_counts(show=False)
    sim.plot_phase(show=False)

    # Deterministic map
    x0 = np.array([
        sim.rock_counts[0] / sim.N,
        sim.paper_counts[0] / sim.N,
        sim.scissors_counts[0] / sim.N,
    ])

    h = 0.05  # step size for the map
    map_steps = len(sim.times)
    xs_map = iterate_rps_map(x0, h=h, steps=map_steps)

    # Deterministic phase plot
    plot_rps_map(xs_map, title=f"Deterministic RPS Map (h={h})", show=False)

    # Comparison plot
    plt.figure()
    plt.plot(xs_map[:, 0], xs_map[:, 1],
             label="Map (xR vs xP)", linestyle="--")
    plt.plot(
        np.array(sim.rock_counts) / sim.N,
        np.array(sim.paper_counts) / sim.N,
        label="Simulation (xR vs xP)", alpha=0.7
    )
    plt.xlabel("x_R")
    plt.ylabel("x_P")
    plt.title("Simulation vs Deterministic Map (Phase Space)")
    plt.legend()
    plt.tight_layout()

    plt.show()
    print("Simulation complete.\n")


def run_rps_bif():
    """Show all three bifurcation diagrams: x_R, x_P, x_S vs h."""
    plt.close("all")
    print("\n=== RPS Map Bifurcation Diagrams ===")

    h_min_in = input("h_min (default 0.0): ").strip()
    h_max_in = input("h_max (default 0.6): ").strip()
    num_h_in = input("num_h (default 300): ").strip()

    h_min = float(h_min_in) if h_min_in else 0.0
    h_max = float(h_max_in) if h_max_in else 0.6
    num_h = int(num_h_in) if num_h_in else 300

    # One figure each for x_R, x_P, x_S
    rps_bifurcation_diagram(h_min=h_min, h_max=h_max,
                            num_h=num_h, coord="R", show=False)
    rps_bifurcation_diagram(h_min=h_min, h_max=h_max,
                            num_h=num_h, coord="P", show=False)
    rps_bifurcation_diagram(h_min=h_min, h_max=h_max,
                            num_h=num_h, coord="S", show=False)

    plt.show()
    print("Bifurcation diagrams finished.\n")


def run_rps_lya():
    """Run Lyapunov scan for the RPS map."""
    plt.close("all")
    print("\n=== RPS Map Lyapunov Scan ===")

    h_min_in = input("h_min (default 0.0): ").strip()
    h_max_in = input("h_max (default 0.6): ").strip()
    num_h_in = input("num_h (default 60): ").strip()

    h_min = float(h_min_in) if h_min_in else 0.0
    h_max = float(h_max_in) if h_max_in else 0.6
    num_h = int(num_h_in) if num_h_in else 60

    scan_lyapunov(h_min=h_min, h_max=h_max, num_h=num_h, show=False)

    plt.show()
    print("Lyapunov scan finished.\n")


def main():
    while True:
        print("===== Simulation Menu =====")
        print("1. Rock–Paper–Scissors Simulation")
        print("2. RPS Map Bifurcation Diagrams (x_R, x_P, x_S)")
        print("3. RPS Map Lyapunov Scan")
        print("4. Competitive Populations Simulation")
        print("5. Allee Logistic Map")
        print("0. Exit")
        print("===========================")

        choice = input("Choose an option: ").strip()

        match choice:
            case "1":
                seed = input(
                    "Enter seed (blank=default, r=random): "
                ).strip()
                run_rps(seed)
            case "2":
                run_rps_bif()
            case "3":
                run_rps_lya()
            case "4":
                run_competitive_pops()
            case "5":
                run_allee_map()
            case "0":
                print("Exiting program.")
                break
            case _:
                print("Invalid option. Try again.\n")


if __name__ == "__main__":
    main()
