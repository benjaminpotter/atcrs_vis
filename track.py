from argparse import ArgumentParser
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def main():
    parser = ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("fa_flight_id")
    args = parser.parse_args()
    path = Path(args.path)

    df = pd.read_csv(path)
    df = df[df["fa_flight_id"] == args.fa_flight_id]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by="timestamp")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlim(-75, 75)
    ax.set_ylim(-75, 75)
    ax.set_zlim(0, 7)

    ax.scatter(
        df["enu_east_km"],
        df["enu_north_km"],
        df["enu_up_km"],
        s=10,
    )

    # for edge in edges:
    #     id1, id2 = edge
    #     x1, y1, z1, _ = vertices[id1]
    #     x2, y2, z2, _ = vertices[id2]
    #     ax.plot([x1, x2], [y1, y2], [z1, z2], color="black", lw=0.1)

    fig.legend()

    save_path = f"{args.fa_flight_id}.png"
    plt.savefig(save_path, dpi=300)


if __name__ == "__main__":
    main()
