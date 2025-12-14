from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plots


def read_file(filename):
    """Read trajectory file and return a dict of vertices and list of edges."""
    vertices = {}
    edges = []

    with open(filename, "r") as file:
        for line in file:
            parts = line.split()
            if parts[0] == "v":
                id = int(parts[1])
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                b = float(parts[5])
                c1, c2 = float(parts[6]), float(parts[7])
                vertices[id] = (x, y, z, b, c1, c2)
            elif parts[0] == "e":
                id1, id2 = int(parts[1]), int(parts[2])
                edges.append((id1, id2))

    return vertices, edges


def latexify_doublecolumn(width="column", fontsize=10, projection="2d"):
    """Set LaTeX-style figure parameters."""
    if width == "column":
        fig_width = 3.4
    elif width == "full":
        fig_width = 7.0
    else:
        raise ValueError("width must be 'column' or 'full'")

    if projection == "3d":
        aspect = 1.1
    else:
        golden_ratio = (5**0.5 - 1) / 2
        aspect = golden_ratio

    fig_height = fig_width * aspect

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.size": fontsize,
            "axes.labelsize": fontsize,
            "legend.fontsize": fontsize * 0.9,
            "xtick.labelsize": fontsize * 0.9,
            "ytick.labelsize": fontsize * 0.9,
            "figure.figsize": (fig_width, fig_height),
        }
    )


def main(path, colors=None, label=None):

    color = "blue"
    latexify_doublecolumn(fontsize=9, projection="2d")

    vertices, edges = read_file(path)

    es = np.array([v[0] for v in vertices.values()])
    ns = np.array([v[1] for v in vertices.values()])
    us = np.array([v[2] for v in vertices.values()])
    bs = np.array([v[3] for v in vertices.values()])

    # target = np.array([0.0, 0.0, 0.0])
    # r = 1.5  # keep points within this radius
    # pts = np.column_stack([es, ns, us])
    # dist = np.linalg.norm(pts - target, axis=1)
    # mask = dist <= r

    mask = (3.5 <= bs) & (bs <= 3.6)

    es_f = es[mask]
    ns_f = ns[mask]
    us_f = us[mask]

    # start = 0
    # end = start + 10
    # es = es[start:end]
    # ns = ns[start:end]
    # us = us[start:end]

    fig = plt.figure()
    axes = [
        # fig.add_subplot(111),
        fig.add_subplot(111, projection="3d"),
        # fig.add_subplot(131),
        # fig.add_subplot(132),
        # fig.add_subplot(133),
    ]

    # ax.set_xlim(-75, 75)
    # ax.set_ylim(-75, 75)

    # ax.set_xlabel("East [km]", labelpad=1)
    # ax.set_ylabel("North [km]", labelpad=1)

    # XY projection
    axes[0].scatter([0], [0], [0], marker="o", c="red", s=10, depthshade=False)
    axes[0].scatter(es_f, ns_f, us_f, marker="o", s=3, depthshade=False)
    axes[0].scatter(es_f, ns_f, [0], marker=".", depthshade=False)

    axes[0].set_xlabel("East [km]", labelpad=1)
    axes[0].set_ylabel("North [km]", labelpad=1)
    axes[0].set_zlabel("Up [km]", labelpad=1)

    for e, n, u in zip(es_f, ns_f, us_f):
        axes[0].plot([e, e], [n, n], [0, u], c="gray", alpha=0.3)

    # for edge in edges:
    #     id1, id2 = edge
    #     if start < id1 and id1 < end and start < id2 and id2 < end:
    #         e0, n0, u0, _, _, _ = vertices[id1]
    #         e1, n1, u1, _, _, _ = vertices[id2]
    #         axes[0].plot([e0, e1], [n0, n1], [u0, u1], color="black")

    # # XZ projection
    # axes[1].scatter(e, u, c=n, cmap="plasma", marker=".")
    # axes[1].set_xlabel("X")
    # axes[1].set_ylabel("Z")
    # axes[1].set_title("XZ Projection")
    #
    # # YZ projection
    # axes[2].scatter(n, u, c=e, cmap="inferno", marker=".")
    # axes[2].set_xlabel("Y")
    # axes[2].set_ylabel("Z")
    # axes[2].set_title("YZ Projection")

    fig.legend()
    # plt.tight_layout(pad=0.2)
    # fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    # plt.savefig("rrt_tree.pdf", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    file = "../atcrs/rrt_tree.g"
    main(file, label="PL-RRT$^*$")
