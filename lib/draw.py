from matplotlib import pyplot as plt
import numpy as np


def plot_3d_with_colors(x, y, z, x_label, y_label, z_label, title):
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection='3d')

    j = 100
    steps = len(x)
    colors = plt.cm.jet(np.linspace(0, 1, int(steps//j) + 1))

    for i in range(0, steps, j):
        ax.plot(x[i:i+j], y[i:i+j], z[i:i+j],
                color=colors[int(i//j)], alpha=0.4)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    plt.title(title)
