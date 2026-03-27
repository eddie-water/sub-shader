import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os

ASCII_OFFSET = 97

def plot_vectors(vectors, filename):
    labels = [
        rf'$\vec{{{chr(ASCII_OFFSET + i)}}} = ({x}, {y})$'
        for i, (x, y) in enumerate(vectors)
    ]

    nrows = len(vectors) // 2
    ncols = 2

    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(
            nrows=nrows, 
            ncols=ncols, 
            figure=fig, 
            wspace=0.5,
            hspace=0.2)

    for i, (v, label) in enumerate(zip(vectors, labels)):
        row = i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col])
        
        ax.axhline(0, color='black', linewidth=2)
        ax.axvline(0, color='black', linewidth=2)

        ax.quiver(0, 0, v[0], v[1], 
            angles='xy', scale_units='xy', scale=1,
            color='steelblue',
            alpha=1,
            width=0.02,
            headwidth=3, 
            headlength=3, 
            headaxislength=3)

        ax.set_title(label, fontsize=10, color='royalblue')
        # ax.set_aspect('equal')
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.suptitle('2D Vectors', fontsize=13, fontweight='bold')

    figures_dir = './figures'
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, f'{filename}.png'), dpi=150)

    plt.show()