"""

"""
import numpy as np

from .autoencoder import Model_GCNTSNE_PairedSamples
import logging
from pathlib import Path as P
from typing import List, Literal

from src.data.dataset import PartialMultiviewDataset
from src.utils.io_utils import *
from src.utils.torch_utils import *
from src.vis.visualize import *

from .propagate import (
    propagate_labels_global,
    propagate_labels_local,
    BUILD_GRAPH_METHOD,
)

from .subspace import Model_Simple_MVC, Model_Subspace_MVC

INIT_LABELS_METHOD = Literal["Concat", "SC-C", "SC-A", "Subspace", "GCNTSNE"]


def initialize_labels(
    X_paired: List[Tensor],
    data: PartialMultiviewDataset,
    init_method: INIT_LABELS_METHOD,
    k: int,
    device,
    **kwargs,
):
    X_paired = convert_tensor(X_paired, dev=device)
    if init_method == "Subspace":
        model = Model_Subspace_MVC(
            data=data,
            X_list=X_paired,
            verbose=True,
        ).to(device)
        model.fit()
    elif init_method == "GCNTSNE":
        model = Model_GCNTSNE_PairedSamples(
            in_channels=data.view_dims,
            verbose=True,
            **kwargs,
        ).to(device)
        model.fit(X_paired=X_paired, data=data)
    else:
        # simple
        model = Model_Simple_MVC(
            X_list=X_paired,
            data=data,
            method=init_method,
            verbose=True,
            k=k,
        ).to(device)
        model.fit()
    return model.ypred


def train_main(
    datapath=P("./data/ORL-40.mat"),
    eta=0.5,
    views=None,
    pp_type: Literal["L", "G"] = "L",
    init: INIT_LABELS_METHOD = "Subspace",
    build_graph: BUILD_GRAPH_METHOD = "KNN",
    k: int = 7,
    ppl: int = 10,
    lamda: float = 0.1,
    hidden_dims: int = 128,
    device="cpu",
    savedir: P = P("output/pseudolabel"),
    save_vars: bool = False,
):
    method_name = f"PLP-{pp_type}"
    config = dict(
        datapath=datapath,
        eta=eta,
        views=views,
        method=method_name,
        init=init,
        build_graph=build_graph,
        hidden_dims=hidden_dims,
        lamda=lamda,
        ppl=ppl,
        k=k,
    )
    train_begin(savedir, config, f"Begin train {method_name}")

    data = PartialMultiviewDataset(
        datapath=datapath,
        paired_rate=1 - eta,
        view_ids=views,
        normalize="minmax",
    )

    X_paired = convert_tensor(data.X_paired_list, dev=device)
    X = convert_tensor(data.X, dev=device)
    M = convert_tensor(data.mask, dev=device, dtype=torch.bool)

    logging.info(f"Initialize labels")
    Y_paired = initialize_labels(
        device=device,
        X_paired=X_paired,
        data=data,
        init_method=init,
        k=k,
    )

    logging.info(f"Propagate labels")

    if pp_type == "G":
        # Global
        outputs, metrics = propagate_labels_global(
            data=data,
            hidden_dims=hidden_dims,
            X=X,
            M=M,
            Y_paired=Y_paired,
            ppl=ppl,
            lamda=lamda,
            device=device,
        )
    else:
        # Local
        X_all = data.X_all_list
        idx_all = data.idx_all_list

        outputs, metrics = propagate_labels_local(
            build_graphs_method=build_graph,
            X_all=X_all,
            data=data,
            Y_paired=Y_paired,
            idx_all=idx_all,
            k=k,
            device=device,
        )

    if save_vars:
        outputs = convert_numpy(outputs)
        H_common = outputs['H_common']
        save_var(savedir, H_common, "H_common")

    train_end(savedir, metrics)
