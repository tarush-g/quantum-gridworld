import matplotlib.pyplot as plt
import numpy as np


def plot_heatmaps(classical_pi, quantum_probs, states, size):
    q = quantum_probs.sum(axis=1)

    cgrid = np.full((size, size), np.nan)
    qgrid = np.full((size, size), np.nan)

    for (r, c), i in states.items():
        cgrid[r, c] = classical_pi[i]
        qgrid[r, c] = q[i]

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    ax[0].imshow(cgrid)
    ax[0].set_title("Classical")

    ax[1].imshow(qgrid)
    ax[1].set_title("Szegedy")

    plt.show()