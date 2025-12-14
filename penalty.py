import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    path = "../atcrs/penalty.csv"

    df = pd.read_csv(path)
    df["x"] = df["x"] * 0.5
    df["y"] = df["y"] * 0.5

    df_avg = df.groupby(["x", "y"], as_index=False)["penalty"].mean()
    df_hm = df_avg.pivot(index="y", columns="x", values="penalty").fillna(1.0)

    latexify_doublecolumn()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    xs, ys = np.meshgrid(df_hm.columns, df_hm.index)
    im = ax.pcolormesh(xs, ys, df_hm.values, shading="auto", cmap="grey")

    ax.set_aspect("equal")
    ax.set_xlabel("East [km]")
    ax.set_ylabel("North [km]")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean Penalty Factor")

    plt.savefig("penalty.eps", dpi=300, bbox_inches="tight")


def latexify_doublecolumn(width="column", fontsize=9):
    # Typical dimensions (inches)
    if width == "column":
        fig_width = 3.4  # approximation of \columnwidth
    elif width == "full":
        fig_width = 7.0  # approximation of \textwidth
    else:
        raise ValueError("width must be 'column' or 'full'")

    golden_ratio = (5**0.5 - 1) / 2
    fig_height = fig_width * golden_ratio

    params = {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "legend.fontsize": fontsize * 0.9,
        "xtick.labelsize": fontsize * 0.9,
        "ytick.labelsize": fontsize * 0.9,
        "figure.figsize": (fig_width, fig_height),
    }

    plt.rcParams.update(params)


if __name__ == "__main__":
    main()
