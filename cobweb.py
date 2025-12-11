import numpy as np
import matplotlib.pyplot as plt

def logistic(x, r):
    return r * x * (1 - x)

def cobweb(r, x0=0.2, steps=100, filename="cobweb.png"):
    xs = [x0]
    for _ in range(steps):
        xs.append(logistic(xs[-1], r))

    xline = np.linspace(0, 1, 400)

    plt.figure(figsize=(5,5))
    plt.plot(xline, logistic(xline, r), 'k', linewidth=1)
    plt.plot(xline, xline, 'k--', linewidth=1)

    x = x0
    for _ in range(steps):
        y = logistic(x, r)
        plt.plot([x, x], [x, y], 'r', linewidth=0.7)
        plt.plot([x, y], [y, y], 'r', linewidth=0.7)
        x = y

    plt.title(f"Cobweb diagram for r = {r}")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

cobweb(2.5, filename="cobweb_r25.png")
cobweb(3.2, filename="cobweb_r32.png")
cobweb(4.0, filename="cobweb_r40.png")
