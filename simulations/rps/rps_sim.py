import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

ROCK = 0
PAPER = 1
SCISSORS = 2

def winner(a: int, b: int) -> int | None:
    #if equal: return None

    # Rock vs Scissors -> Rock
    # Paper vs Rocker -> Paper
    # Scissors vs Paper -> Scissors
    # and vice versa
    if a == b:
        return None
    if (a == ROCK and b == SCISSORS) or (b == ROCK and a == SCISSORS):
        return ROCK
    if (a == PAPER and b == ROCK) or (b == PAPER and a == ROCK):
        return PAPER
    if (a == SCISSORS and b == PAPER) or (b == SCISSORS and a == PAPER):
        return SCISSORS
    

@dataclass
class RPSSimulationConfig:
    N: int = 500
    steps: int = 10_000
    p_rock: float = 1/3
    p_paper: float = 1/3
    p_scissors: float = 1/3
    record_every: int = 10
    seed: int | None = None

class RPSSimulation:
    def __init__(self, config: RPSSimulationConfig):
        self.config = config

        if config.seed is not None:
            np.random.seed(config.seed)

        assert abs(config.p_rock + config.p_paper + config.p_scissors - 1.0) < 1e-8, "Probabilities must sum to 1."
        self.N = config.N
        self.state = self._init_state(config)

        self.times = []
        self.rock_counts = []
        self.paper_counts = []
        self.scissors_counts = []

    def _init_state(self, config: RPSSimulationConfig) -> np.ndarray:
        choices = np.random.choice(
            [ROCK, PAPER, SCISSORS],
            size=self.N,
            p=[config.p_rock, config.p_paper, config.p_scissors]
        )
        return choices
    
    def _interaction_step(self):
        i = np.random.randint(0, self.N)
        j = np.random.randint(0, self.N)
        if i == j:
            return
        
        a = self.state[i]
        b = self.state[j]
        w = winner(a, b)
        if w is None:
            return
        if w == a:
            self.state[j] = w
        else:
            self.state[i] = w

    def _record_state(self, t: int):
        values, counts = np.unique(self.state, return_counts=True)

        count_dict = {v: c for v, c in zip(values, counts)}

        rock_count = count_dict.get(ROCK, 0)
        paper_count = count_dict.get(PAPER, 0)
        scissors_count = count_dict.get(SCISSORS, 0)

        self.times.append(t)
        self.rock_counts.append(rock_count)
        self.paper_counts.append(paper_count)
        self.scissors_counts.append(scissors_count)

    def run(self):
        self._record_state(t=0)

        for step in range(1, self.config.steps + 1):
            self._interaction_step()

            if step % self.config.record_every == 0:
                self._record_state(t=step)

    def plot_counts(self, normalized: bool = True, show: bool = True):
        times = np.array(self.times, dtype=float)
        r = np.array(self.rock_counts, dtype=float)
        p = np.array(self.paper_counts, dtype=float)
        s = np.array(self.scissors_counts, dtype=float)

        if normalized:
            r /= self.N
            p /= self.N
            s /= self.N
            ylabel = "Proportion"
        else:
            ylabel = "Count"

        plt.figure()
        
        plt.plot(times, r, label="Rock", color="red")
        plt.plot(times, p, label="Paper", color="blue")
        plt.plot(times, s, label="Scissors", color="green")
        plt.xlabel("Interaction Steps")
        plt.ylabel(ylabel)
        plt.title("Rock-Paper-Scissors Population Dynamics")
        plt.legend()
        plt.tight_layout()
        if show:
            plt.show()
        
    
    def plot_phase(self, show: bool = True):
        xR = np.array(self.rock_counts, dtype=float) / self.N
        xP = np.array(self.paper_counts, dtype=float) / self.N

        plt.figure()
        plt.plot(xR, xP, marker=".", linewidth=0.8, markersize=2)
        plt.xlabel("x_R (Rock Proportion)")
        plt.ylabel("x_P (Paper Proportion)")
        plt.title("Phase Plot of Rock-Paper-Scissors Dynamics")
        plt.tight_layout()
        if show:
            plt.show()
