import matplotlib.pyplot as plt
import pandas as pd
from argparse import ArgumentParser
import numpy as np


def main():
    import matplotlib

    matplotlib.use("QtAgg")
    latexify_doublecolumn("column", fontsize=9)

    parser = ArgumentParser()
    parser.add_argument("ara_path")
    parser.add_argument("rrt_path")
    parser.add_argument("human_path")
    args = parser.parse_args()

    df_ara = prep(pd.read_csv(args.ara_path))
    df_rrt = prep(pd.read_csv(args.rrt_path))
    df_human = pd.read_csv(args.human_path)

    print(df_human.head())
    df_human["best_path_length"] = df_human["path_length_ecld"]

    def make_medians(df, name, cols):
        _df = df[cols].median().to_frame()
        _df.columns = [name]
        _df.index = ["30", "60", "180"]
        return _df

    def make_table(df):
        return pd.concat(
            [
                make_medians(df, name, to_cols(name))
                for name in ["iters", "states", "path_length"]
            ],
            axis=1,
        )

    def numprint_formatter(x):
        if pd.isna(x):
            return r"\text{NaN}"
        return f"\\numprint{{{x:.1f}}}"

    df_tab = pd.concat([make_table(df_ara), make_table(df_rrt)], axis=0)
    print(df_tab.to_latex(escape=False, formatters=[numprint_formatter] * 3))
    print(df_human["best_path_length"].median())

    # Choose the column you want to plot
    # cols = ["path_length_30s", "path_length_1m", "path_length_3m"]
    dfs = [df_ara, df_rrt]
    names = ["PL-ARA$^*$", "PL-RRT$^*$"]

    cols = ["best_path_length"]
    fig, ax = plot_ecdf(dfs + [df_human], names + ["Human"], cols)
    ax.set_xlabel("Path Length")
    ax.set_ylabel("Cumulative Probability")
    plt.tight_layout()
    plt.savefig("path_length_ecdf.pdf")

    cols = ["max_states"]
    fig, ax = plot_ecdf(dfs, names, cols)
    ax.set_xscale("log")
    ax.set_xlabel("Number of States")
    ax.set_ylabel("Cumulative Probability")
    plt.tight_layout()
    plt.savefig("states_ecdf.pdf")


def plot_ecdf(dfs, names, cols):

    mi = np.inf
    ma = -np.inf
    for df in dfs:
        for col in cols:
            mi = min(mi, df[col].min())
            ma = max(ma, df[col].max())

    print(f"[{mi}, {ma}]")

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for name, df in zip(names, dfs):
        for col in cols:
            x, y = ecdf(df, col)

            # Pad at the beginning with global_min and 0
            x = np.insert(x, 0, mi)
            y = np.insert(y, 0, 0.0)

            # Pad at the end with global_max and 1
            x = np.append(x, ma)
            y = np.append(y, 1.0)

            ax.step(x, y, where="post", label=name)

    # ax.set_xlim(mi, ma)
    ax.grid(True)
    ax.legend()

    return fig, ax


def latexify_doublecolumn(width="column", fontsize=10):
    """
    Configure matplotlib for LaTeX double-column figure production.

    Parameters
    ----------
    width : {"column", "full"}
        'column' for \columnwidth, 'full' for \textwidth
    fontsize : int
        Base font size to match your document (IEEE typically uses 10 pt).
    """

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


def to_cols(name):
    times = ["30s", "1m", "3m"]
    return [f"{name}_{time}" for time in times]


def prep(df):
    def replace_missing(cols, missing):
        df[cols] = df[cols].replace(missing, pd.NA)
        df[cols] = df[cols].ffill(axis=1)

    path_cols = to_cols("path_length")
    replace_missing(path_cols, np.inf)

    state_cols = to_cols("states")
    replace_missing(state_cols, 0)

    replace_missing(to_cols("iters"), 0)

    df["best_path_length"] = df[path_cols].min(axis=1).replace(pd.NA, np.inf)
    df["max_states"] = df[state_cols].max(axis=1).replace(pd.NA, 0)

    return df


def ecdf(df, col):

    # Extract the data as a sorted array
    x = np.sort(df[col].values)

    # Compute empirical CDF values
    y = np.arange(1, len(x) + 1) / len(x)

    return x, y


if __name__ == "__main__":
    main()
