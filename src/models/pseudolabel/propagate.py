import numpy as np
import torch

from .autoencoder import Model_GCNTSNE_AllSamples, Model_GCN_SemiNodeClass
from typing import List, Literal

from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import rbf_kernel

from src.data.dataset import PartialMultiviewDataset
from src.utils.torch_utils import *
from src.vis.visualize import *

from .metrics import get_all_metrics
from .subspace import Model_Subspace_SingleView

BUILD_GRAPH_METHOD = Literal["Subspace", "RBF", "KNN"]


def build_graphs_within_view(
    X_all: List[Tensor],  # [X_paired, X_single] for all views
    data: PartialMultiviewDataset,
    method: BUILD_GRAPH_METHOD,
    device,
    k: int = 7,
    gamma: int = 20,
) -> List[Tensor]:
    m = len(X_all)
    A_all = [None] * m

    for v in range(m):
        logging.info(f"Build graph for view {v}")
        if method == "Subspace":
            X = convert_tensor(X_all[v], dev=device)
            model = Model_Subspace_SingleView(
                sampleNum=X.shape[0],
                X=X,
                data=data,
                verbose=True,
            ).to(device)
            model.fit()
            A_all[v] = model.S
        elif method == "RBF":
            A_all[v] = rbf_kernel(X_all[v], X_all[v], gamma=gamma)
        elif method == "KNN":
            A_all[v] = kneighbors_graph(
                X_all[v], n_neighbors=k, mode="distance"
            ).toarray()
        else:
            raise ValueError(method)
    return A_all


def propagate_within_view(
    data: PartialMultiviewDataset,
    X_all: List[Tensor],  # [X_paired, X_single] for all views
    A_all: List[Tensor],
    Y_paired: Tensor,
    device,
) -> List[Tensor]:
    m = len(X_all)
    Y_all = [None] * m
    Y_paired_onehot = np.eye(data.clusterNum)[convert_numpy(Y_paired)]
    for v in range(m):
        logging.info(f"Propagate view {v}")
        X = convert_tensor(X_all[v], dev=device)
        A = convert_tensor(A_all[v], dev=device)
        model = Model_GCN_SemiNodeClass(
            in_feats=X.shape[1],
            out_feats=data.clusterNum,
            verbose=True,
        ).to(device)
        model.fit(
            X=X,
            A=A,
            Y=Y_paired,
        )
        Y_all[v] = np.concatenate([Y_paired_onehot, convert_numpy(model.preds_test)], 0)

    return Y_all


def assemble_labels(
    data: PartialMultiviewDataset,
    Y_all: List[Tensor],
    idx_all: List[Tensor],
):
    n, m, c = data.sampleNum, data.viewNum, data.clusterNum
    Y = np.zeros((n, c))
    Y_all = convert_numpy(Y_all)
    idx_all = convert_numpy(idx_all)

    for v in range(m):
        Y[idx_all[v]] += Y_all[v]

    D = np.expand_dims(data.mask_sum, 1)
    ypreds = Y / D
    assert np.allclose(np.sum(ypreds, 1), 1)

    ypreds = np.argmax(ypreds, 1)
    metrics = get_all_metrics(
        label=data.Y,
        ypred=ypreds,
    )
    return metrics, ypreds


def propagate_labels_local(
    build_graphs_method: BUILD_GRAPH_METHOD,
    X_all,
    data: PartialMultiviewDataset,
    k: int,
    device: int,
    Y_paired: Tensor,
    idx_all,
):
    logging.info(f"Build graphs within view {build_graphs_method}")
    A_all = build_graphs_within_view(
        X_all=X_all,
        method=build_graphs_method,
        data=data,
        k=k,
        device=device,
    )

    logging.info(f"Propagate labels within view")
    Y_all = propagate_within_view(
        data=data,
        X_all=X_all,
        A_all=A_all,
        Y_paired=Y_paired,
        device=device,
    )

    logging.info(f"Assemble labels")
    metrics, ypred = assemble_labels(
        data=data,
        Y_all=Y_all,
        idx_all=idx_all,
    )
    outputs = dict(ypred=ypred, A_all=A_all)

    return outputs, metrics


def propagate_labels_global(
    data: PartialMultiviewDataset,
    X: List[Tensor],
    M: Tensor,
    hidden_dims: int,
    Y_paired: Tensor,
    device: int,
    ppl: int,
    lamda: float,
):
    model = Model_GCNTSNE_AllSamples(
        in_channels=data.view_dims,
        clusterNum=data.clusterNum,
        hidden_dims=hidden_dims,
        perplexity=ppl,
        lamda=lamda,
        verbose=True,
    ).to(device)

    outputs = model.fit(X=X, M=M, Y_paired=Y_paired, data=data)
    return outputs, model.metrics
