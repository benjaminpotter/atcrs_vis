from argparse import ArgumentParser
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def main():
    parser = ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    path = Path(args.path)

    benchmark_paths = [
        (path / file, label)
        for file, label in [
            ("rrt_v2_benchmark.csv", "Euclidean"),
            ("rrt_v3_benchmark.csv", "Dubins"),
            ("rrt_v4_benchmark.csv", "Dubins + Filter"),
        ]
    ]

    benchmarks = [
        (bm_path, pd.read_csv(bm_path), label) for bm_path, label in benchmark_paths
    ]
    print(f"found {len(benchmarks)} benchmarks")

    latexify_doublecolumn()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel("Number of States")
    ax.set_ylabel("Iterations per Millisecond")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)

    for bm_path, df, label in benchmarks:
        print(bm_path)
        ax.plot(df["state_count"], df["iters_per_ms"], label=label)

    fig.legend()

    save_path = f"{path.stem}.pdf"
    plt.tight_layout()
    plt.savefig(save_path)


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


if __name__ == "__main__":
    main()
