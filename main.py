from simulations.rps.rps_sim import RPSSimulation, RPSSimulationConfig
from simulations.rps.rps_map import iterate_rps_map, plot_rps_map

import numpy as np
import matplotlib.pyplot as plt
def run_rps(seed_input: str):
    print("\n=== Rock–Paper–Scissors Simulation ===")

    # Seed handling
    if seed_input.lower() == 'r':
        seed_in = np.random.randint(0, 1_000_000)
    elif seed_input == '':
        seed_in = None
    else:
        seed_in = int(seed_input)

    # Stochastic simulation config
    config = RPSSimulationConfig(seed=seed_in)
    sim = RPSSimulation(config)
    sim.run()

    # -------------------------------
    #     STOCHASTIC PLOTS
    # -------------------------------
    sim.plot_counts(show=False)
    sim.plot_phase(show=False)

    # -------------------------------
    #     DETERMINISTIC MAP
    # -------------------------------
    # Initial condition matching the simulation start
    x0 = np.array([
        sim.rock_counts[0] / sim.N,
        sim.paper_counts[0] / sim.N,
        sim.scissors_counts[0] / sim.N,
    ])

    # Choose a step-size parameter h for the map
    h = 0.05  # (you can experiment with 0.01, 0.1, 0.2 later)

    # Iterate the deterministic map
    map_steps = len(sim.times)  # match sim length
    xs_map = iterate_rps_map(x0, h=h, steps=map_steps)

    # -------------------------------
    #     DETERMINISTIC PHASE PLOT
    # -------------------------------
    plot_rps_map(xs_map, title=f"Deterministic RPS Map (h={h})")

    # -------------------------------
    #     COMPARISON PLOT (OPTIONAL)
    # -------------------------------
    plt.figure()
    plt.plot(xs_map[:,0], xs_map[:,1], label="Map (xR vs xP)", linestyle="--")
    plt.plot(
        np.array(sim.rock_counts)/sim.N,
        np.array(sim.paper_counts)/sim.N,
        label="Simulation (xR vs xP)", alpha=0.7
    )
    plt.xlabel("x_R")
    plt.ylabel("x_P")
    plt.title("Simulation vs Deterministic Map (Phase Space)")
    plt.legend()
    plt.tight_layout()

    # Show everything at once
    plt.show()

    print("Simulation complete.\n")



def main():
    while True:
        print("===== Simulation Menu =====")
        print("1. Rock–Paper–Scissors")
        print("0. Exit")
        print("===========================")

        choice = input("Choose a simulation: ").strip()

        match choice:
            case "1":
                seed = input("Enter seed (or leave blank for default, r for random): ").strip()
                run_rps(seed)
            case "0":
                print("Exiting program.")
                break
            case _:
                print("Invalid option. Try again.\n")


if __name__ == "__main__":
    main()
