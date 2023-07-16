import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics.pairwise import pairwise_distances
import torch
from pathlib import Path as P
from src.utils.crack_number import crack
import numpy as np
import logging
import pandas as pd
from sklearn.manifold import TSNE
from ..data import MultiviewDataset

logging.basicConfig(level=logging.INFO)
from src.vis.plots import plot_gnd_hat
from sklearn.model_selection import train_test_split


# matplotlib.use(backend="pgf")
# logging.info(f"use backend {matplotlib.get_backend()}")


def plot_scatters(H, datapath: P, size=10, palette='viridis', markers='o'):
    H = TSNE().fit_transform(H)
    data = MultiviewDataset(datapath=datapath)
    df = pd.DataFrame.from_dict(dict(x=H[:, 0], y=H[:, 1], label=data.Y))
    pal = sns.color_palette(palette, n_colors=data.clusterNum)
    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="label",
        palette=pal,
        markers=markers,
        edgecolor=None,
        s=size,
        legend=False,
    )
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("")
    plt.ylabel("")


def plot_heatmap(X, title=None):
    sns.heatmap(X, cbar=None, xticklabels=[], yticklabels=[])
    if title:
        plt.title(title)


def savefig_show(save: P):
    plt.tight_layout()
    plt.savefig(save)
    plt.close()
    logging.info(f"savefig {save}")


def reshape_features(x, transpose=True):
    n, d = x.shape
    h, w = crack(d)
    x = x.reshape((n, w, h))
    if transpose:
        x = np.transpose(x, (0, 2, 1))
    return x


def plot_completion(X_gt, X_hat, M, view, num):
    x_pred = X_hat[view][~M[:, view]]
    x_true = X_gt[view][~M[:, view]]
    x_pred, _, x_true, _ = train_test_split(
        x_pred, x_true, train_size=num, shuffle=True
    )

    x_pred = reshape_features(x_pred)
    x_true = reshape_features(x_true)

    plot_gnd_hat(x_true, x_pred, cmap="viridis")


def visualize_completion(taskdir: P, args: dict, num=8, view=0, dpi=200, visdir=None):
    prefix = taskdir.parent.name
    logging.info("visualize_completion for {}".format(prefix))
    if visdir is None:
        visdir = taskdir
    visdir.mkdir(parents=1, exist_ok=1)

    H_common = torch.load(taskdir.joinpath("H_common.pt"), "cpu")
    plot_heatmap(pairwise_distances(H_common))
    savefig_show(visdir.joinpath(f"{prefix}-H_common.png"))
    logging.info("Done H_common")

    X_hat = torch.load(taskdir / "X_hat.pt", "cpu")
    X_gt = torch.load(taskdir / "X_gt.pt", "cpu")
    M = torch.load(taskdir / "M.pt", "cpu")

    plot_completion(X_gt, X_hat, M, 0, num)
    savefig_show(visdir.joinpath(f"{prefix}-gnd_hat.png"))
    logging.info("Done gnd hat")

    plot_history(
        taskdir.joinpath("history.pt"),
        keys=["loss", "mse"],
        dpi=dpi,
    )
    savefig_show(visdir.joinpath(f"{prefix}-history-mse.png"))

    plot_history(
        taskdir.joinpath("history.pt"),
        keys=["loss", "ACC", "NMI", "PUR", "F1"],
        dpi=dpi,
    )
    savefig_show(visdir.joinpath(f"{prefix}-history-cluster.png"))
    logging.info("Done history")

    plot_scatters(
        H=H_common,
        args=args,
    )
    savefig_show(visdir.joinpath(f"{prefix}-scatters.png"))
    logging.info("Done scatters")


def plot_history(
    # infile: P,
    data: list,
    keys: list,
    dpi: int = 200,
    major: str = None,
):
    if major is None:
        major = keys[0]
    # Process data.
    # data = torch.load(infile, "cpu")
    # print(data)
    x_list = range(len(data))
    # data is a list of dict
    df = pd.DataFrame.from_records(data)
    df = df.applymap(float)
    data = df.to_dict("list")  # ACC => [...], loss => [...]

    # Begin plotting.
    plt.figure(dpi=dpi)
    plt.grid("--")
    ax1 = plt.subplot()
    ax2 = ax1.twinx()
    ln1 = ax1.plot(x_list, data[major], color="C0", label=major.capitalize())
    lns = ln1
    i = 1
    for key, value in data.items():
        if key == major or key not in keys:
            continue
        ln_now = ax2.plot(x_list, value, color=f"C{i}", label=key.upper())
        i += 1
        lns = lns + ln_now

    plt.legend(lns, [l.get_label() for l in lns], loc="best")
