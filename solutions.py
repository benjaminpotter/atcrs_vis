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
                id = parts[1]
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                b = float(parts[5])
                c1, c2 = float(parts[6]), float(parts[7])
                original = parts[8] == "true"
                vertices[id] = (x, y, z, b, c1, c2, original)
            elif parts[0] == "e":
                id1, id2 = int(parts[1]), int(parts[2])
                edges.append((id1, id2))

    return vertices, edges


def compute_spline(df):
    """Compute a 3D Hermite spline for a trajectory DataFrame."""
    tx = np.cos(df["b"])
    ty = np.sin(df["b"])
    tz = np.gradient(df["z"])  # Simple z tangent

    # Optional scale for tangents
    scale = np.gradient(
        np.sqrt(
            np.diff(df["x"], prepend=df["x"][0]) ** 2
            + np.diff(df["y"], prepend=df["y"][0]) ** 2
            + np.diff(df["z"], prepend=df["z"][0]) ** 2
        )
    )
    tx *= scale
    ty *= scale

    t = np.linspace(0, 1, len(df))
    sx = CubicHermiteSpline(t, df["x"], tx)
    sy = CubicHermiteSpline(t, df["y"], ty)
    sz = CubicHermiteSpline(t, df["z"], tz)

    t_fine = np.linspace(0, 1, 400)
    x_s, y_s, z_s = sx(t_fine), sy(t_fine), sz(t_fine)
    return x_s, y_s, z_s


def plot_trajectory(ax, df, spline_coords, color="blue", label=None):
    """Plot a trajectory on given axes."""
    x_s, y_s, z_s = spline_coords
    min_z = df["z"].min()

    # Vertical lines to ground plane
    # skip = 6
    # for xi, yi, zi in zip(df["x"][::skip], df["y"][::skip], df["z"][::skip]):
    #     ax.plot(
    #         [xi, xi],
    #         [yi, yi],
    #         [min_z, zi],
    #         color=color,
    #         linestyle="-",
    #         linewidth=1,
    #         alpha=0.3,
    #     )

    # Plot smooth spline
    ax.plot(x_s, y_s, z_s, label=label, c=color)
    ax.plot(x_s, y_s, [0], c=color, alpha=0.3)

    # Original points
    ax.scatter(
        df[df["original"]]["x"],
        df[df["original"]]["y"],
        df[df["original"]]["z"],
        s=3,
        c=color,
        depthshade=False,
    )


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


def main(file_list, output, colors=None, labels=None):
    """Plot multiple trajectories on the same axes."""
    latexify_doublecolumn(fontsize=9, projection="2d")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # ax.set_xlim(-75, 75)
    # ax.set_ylim(-75, 75)
    ax.set_zlim(0, 4)
    ax.set_xlabel("East [km]", labelpad=1)
    ax.set_ylabel("North [km]", labelpad=1)
    ax.set_zlabel("Up [km]", labelpad=1)
    ax.set_box_aspect((1, 1, 0.7))

    if colors is None:
        colors = ["blue", "red", "green", "orange", "purple"]
    if labels is None:
        labels = [None] * len(file_list)

    for i, path in enumerate(file_list):
        vertices, _ = read_file(path)
        df = pd.DataFrame(
            {
                "x": [v[0] for v in vertices.values()],
                "y": [v[1] for v in vertices.values()],
                "z": [v[2] for v in vertices.values()],
                "b": [v[3] for v in vertices.values()],
                "original": [v[6] for v in vertices.values()],
            }
        )

        spline_coords = compute_spline(df)
        plot_trajectory(
            ax, df, spline_coords, color=colors[i % len(colors)], label=labels[i]
        )

    fig.legend()
    # plt.tight_layout(pad=0.2)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    plt.savefig(output, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    # Example usage: list of trajectory files
    # files = ["paths/ara_plan.g", "paths/pl_rrt_plan.g", "paths/human_plan.g"]
    # main(files, "pl_search_trees.pdf", labels=["PL-ARA$^*$", "PL-RRT$^*$", "Human"])

    files = ["paths/mc_ara_plan.g", "paths/mc_rrt_plan.g", "paths/human_plan.g"]
    main(files, "mc_search_trees.pdf", labels=["MC-ARA$^*$", "MC-RRT$^*$", "Human"])

    # files = ["paths/mc_ara_plan.g", "paths/human_plan.g"]
    # main(files, labels=["MC-ARA$^*$", "Human"])
