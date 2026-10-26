from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def heatmap(ax, matrix, xlabels, ylabels, fmt="{:d}", cmap="Blues"):
    matrix = np.asarray(matrix)
    im = ax.imshow(matrix, cmap=cmap)
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=45, ha="right")
    ax.set_yticks(range(len(ylabels)), ylabels)
    threshold = (matrix.max() + matrix.min()) / 2
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, fmt.format(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black", fontsize=8)
    return im
