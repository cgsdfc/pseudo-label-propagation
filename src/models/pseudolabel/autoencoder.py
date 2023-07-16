# MIT License

# Copyright (c) 2023 Ao Li, Cong Feng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
import math
from typing import List

from src.utils.torch_utils import *

from .backbone import GCN_Block_GINN, GCN_Encoder_SDIMC, Imputer
from .metrics import KMeans_Evaluate, get_all_metrics, MaxMetrics
from .ptsne_training import (
    calculate_optimized_p_cond,
    get_q_joint,
    loss_function,
    make_joint,
)


class PairedSampleLoss(nn.Module):
    """
    The loss imposed on the paired samples to initialize pseudo labels.
    """

    def __init__(self, use_tsne=1, use_recon=1):
        super().__init__()
        self.use_tsne = use_tsne
        self.use_recon = use_recon
        self.loss_manifold_reg = ManifoldRegLoss()

    def forward(self, inputs: dict):
        loss = self.loss_manifold_reg(inputs)
        return loss


class AllSampleLoss(nn.Module):
    """
    The loss imposed on all samples to learn from pseudo labels and other parts.
    """

    def __init__(self, lamda: float):
        super().__init__()
        self.loss_pseudolabel = PseudoLabelLoss()
        self.loss_manifold_reg = ManifoldRegLoss()
        self.lamda = lamda

    def forward(self, inputs: dict):
        loss = self.loss_manifold_reg(inputs)
        loss = loss + self.loss_pseudolabel(inputs) * self.lamda
        return loss


class ManifoldRegLoss(nn.Module):
    """
    t-SNE based manifold regularization loss.
    """

    def forward(self, inputs: dict):
        P_view: List[Tensor] = inputs["P_view"]
        H_common: Tensor = inputs["H_common"]
        M: Tensor = inputs["M"]
        viewNum: int = inputs["viewNum"]
        loss = 0

        for v in range(viewNum):
            h_common = H_common[M[:, v]]
            q_common = get_q_joint(h_common)
            loss += loss_function(p_joint=P_view[v], q_joint=q_common)
        loss = loss / viewNum
        return loss


class ReconstructionLoss(nn.Module):
    """
    MSE-based view completion loss
    """

    def forward(self, inputs: dict):
        X_hat: Tensor = inputs["X_hat"]
        M: Tensor = inputs["M"]
        X_view: Tensor = inputs["X_view"]
        loss = 0
        for v in range(inputs["data"].viewNum):
            loss += F.mse_loss(X_hat[v][M[:, v]], X_view[v])
        loss = loss / inputs["data"].viewNum
        return loss


class PseudoLabelLoss(nn.Module):
    """
    The masked label loss to learn from pseudo labels.
    """

    def forward(self, inputs: dict):
        Y_pred: Tensor = inputs["Y_pred"]
        M_paired: Tensor = inputs["M_paired"]
        Y_paired: Tensor = inputs["Y_paired"]
        loss = F.cross_entropy(Y_pred[M_paired], Y_paired)
        return loss


class MultiviewEncoder(nn.Module):
    """
    The multi-view encoder part of the model.
    """

    def __init__(self, hidden_dims: int, in_channels: List[int]):
        super().__init__()
        self.hidden_dims = hidden_dims

        self.encoder_view = nn.ModuleList()
        for in_channel in in_channels:
            encoder = GCN_Encoder_SDIMC(
                view_dim=in_channel, clusterNum=self.hidden_dims
            )

            self.encoder_view.append(encoder)

    def forward(self, inputs: dict):
        X_view: List[Tensor] = inputs["X_view"]
        M: Tensor = inputs["M"]
        S_view: List[Tensor] = inputs["S_view"]

        # Encoding
        sampleNum, viewNum = M.shape
        H_view = [None] * viewNum
        for v in range(viewNum):
            h_tilde = self.encoder_view[v](X_view[v], S_view[v])
            H_view[v] = h_tilde

        # Fusion
        H_common = torch.zeros(sampleNum, self.hidden_dims).to(M.device)
        for v in range(viewNum):
            H_common[M[:, v]] += H_view[v]
        H_common = H_common / torch.sum(M, 1, keepdim=True)
        H_common = F.normalize(H_common)

        inputs["H_common"] = H_common
        inputs["H_view"] = H_view
        return inputs


class MultiviewDecoder(nn.Module):
    """
    The decoder for view completion, i.e., completion-pretraining stage.
    """

    def __init__(self, hidden_dims: int, in_channels: List[int]):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.decoder_view = nn.ModuleList()
        for v, in_channel in enumerate(in_channels):
            decoder = Imputer(in_channel, self.hidden_dims)
            self.decoder_view.append(decoder)

    def forward(self, inputs: dict):
        H_common: List[Tensor] = inputs["H_common"]
        X_hat = [None] * len(self.decoder_view)
        for v in range(len(self.decoder_view)):
            X_hat[v] = self.decoder_view[v](H_common)
        inputs["X_hat"] = X_hat
        return inputs


class Model_GCN_Base(nn.Module):
    """
    The encoder-decoder model for representation learning & view completion.
    """

    def __init__(
        self,
        in_channels: List[int],
        hidden_dims: int = 128,
        perplexity: int = 10,
        lr=0.001,
        epochs=200,
        valid_freq=10,
        verbose=True,
    ):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.in_channels = in_channels
        self.perplexity = perplexity
        self.viewNum = len(in_channels)
        self.lr = lr
        self.epochs = epochs
        self.eval_epochs = valid_freq
        self.verbose = verbose

        self.encoder = MultiviewEncoder(
            hidden_dims=hidden_dims,
            in_channels=in_channels,
        )
        self.decoder = MultiviewDecoder(hidden_dims, in_channels)

    def forward(self, inputs: dict):
        inputs = self.encoder(inputs)
        inputs = self.decoder(inputs)
        return inputs


class Model_GCNTSNE_PairedSamples(Model_GCN_Base):
    """
    The encoder-decoder model for representation learning & view completion.
    """

    def __init__(
        self,
        in_channels: List[int],
        hidden_dims: int = 128,
        perplexity: int = 10,
        lamda=0.1,
        lr=0.001,
        epochs=200,
        valid_freq=10,
        verbose=True,
        use_tsne=1,
        use_recon=1,
    ):
        super().__init__(
            in_channels=in_channels,
            hidden_dims=hidden_dims,
            perplexity=perplexity,
            lr=lr,
            epochs=epochs,
            valid_freq=valid_freq,
            verbose=verbose,
        )
        self.lamda = lamda
        self.use_tsne = use_tsne
        self.loss = PairedSampleLoss(use_recon=use_recon, use_tsne=use_tsne)

    def preprocess(self, X_paired: List[Tensor], data):
        # X_list is minmax-scaled.
        device = X_paired[0].device
        n, m = X_paired[0].shape[0], len(X_paired)
        S_view = [
            calculate_optimized_p_cond(
                X_paired[v], math.log2(self.perplexity), dev=device
            )
            for v in range(m)
        ]
        P_view = [make_joint(s) for s in S_view] if self.use_tsne else None

        return dict(
            X_view=X_paired,
            S_view=S_view,
            P_view=P_view,
            M=torch.ones(n, m, device=device, dtype=torch.bool),
            data=data,
            viewNum=data.viewNum,
        )

    def postprocess(self, inputs: dict):
        model = self
        model.eval()
        with torch.no_grad():
            x = model(inputs)
        self.ypred = KMeans_Evaluate(
            X=x["H_common"],
            data=inputs["data"],
            return_ypred=True,
            return_metrics=False,
        )[0]

    def fit(self, X_paired: List[Tensor], data):
        args = self
        model = self
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)
        inputs = self.preprocess(X_paired, data)

        x = inputs
        for epoch in range(args.epochs):
            model.train()
            x = model(x)
            loss = self.loss(x)
            optim.zero_grad()
            loss.backward()
            optim.step()

            if (1 + epoch) % args.eval_epochs == 0 and self.verbose:
                model.eval()
                with torch.no_grad():
                    x = model(x)
                logging.info(f"epoch {epoch:04} loss {loss.item():.4f}")

        self.postprocess(inputs)


class Model_GCNTSNE_AllSamples(Model_GCN_Base):
    """
    The encoder-decoder model for representation learning & view completion.
    """

    def __init__(
        self,
        in_channels: List[int],
        clusterNum: int,
        hidden_dims: int = 128,
        perplexity: int = 10,
        lamda: float = 0.1,
        lr=0.001,
        epochs=200,
        valid_freq=10,
        verbose=True,
    ):
        super().__init__(
            in_channels=in_channels,
            hidden_dims=hidden_dims,
            perplexity=perplexity,
            lr=lr,
            epochs=epochs,
            valid_freq=valid_freq,
            verbose=verbose,
        )
        self.classifier = nn.Linear(hidden_dims, clusterNum)
        self.loss = AllSampleLoss(lamda)

    def forward(self, inputs: dict):
        inputs = super().forward(inputs)
        inputs["Y_pred"] = self.classifier(inputs["H_common"])
        return inputs

    def preprocess(self, X: List[Tensor], M: Tensor, data):
        # X_list is minmax-scaled.
        device = X[0].device
        n, m = X[0].shape[0], len(X)
        X_view = [X[v][M[:, v]] for v in range(m)]
        M_paired = M.sum(1) == m
        S_view = [
            calculate_optimized_p_cond(
                X_view[v], math.log2(self.perplexity), dev=device
            )
            for v in range(m)
        ]
        P_view = [make_joint(s) for s in S_view]
        return dict(
            X_view=X_view,
            S_view=S_view,
            P_view=P_view,
            M=M,
            M_paired=M_paired,
            data=data,
            viewNum=data.viewNum,
        )

    def evaluate(self, inputs: dict):
        self.eval()
        model = self
        model.eval()
        with torch.no_grad():
            x = model(inputs)
        metrics = KMeans_Evaluate(X=x["H_common"], data=inputs["data"])[0]
        return metrics

    def fit(self, X: List[Tensor], M: Tensor, Y_paired: Tensor, data):
        args = self
        model = self
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)
        inputs = self.preprocess(X, M, data)
        inputs["Y_paired"] = Y_paired

        mm = MaxMetrics()
        x = inputs
        outputs = {}
        history = []

        for epoch in range(args.epochs):
            model.train()
            x = model(x)
            loss = self.loss(x)
            optim.zero_grad()
            loss.backward()
            optim.step()

            if (1 + epoch) % args.eval_epochs == 0 and self.verbose:
                metrics = self.evaluate(x)
                if mm.update(**metrics)["ACC"]:
                    outputs.update(H_common=x["H_common"])
                loss = loss.item()
                logging.info(f"epoch {epoch:04} loss {loss:.4f} {metrics}")
                metrics["loss"] = loss
                history.append(metrics)

        self.metrics = mm.report(current=False)
        return outputs


class Model_GCN_SemiNodeClass(nn.Module):
    """
    两层GCN网络，用于节点分类或者回归。
    """

    def __init__(
        self,
        in_feats,
        out_feats,
        dropout=0.2,
        lr=0.01,
        epochs=200,
        valid_freq=10,
        verbose=True,
    ) -> None:
        super().__init__()
        self.dim_emb = out_feats
        self.dim_input = in_feats
        self.dropout = dropout
        self.dim_hidden = int(round(0.8 * self.dim_input))
        self.lr = lr
        self.epochs = epochs
        self.valid_freq = valid_freq
        self.verbose = verbose

        self.conv1 = GCN_Block_GINN(
            in_channels=self.dim_input,
            out_channels=self.dim_hidden,
            activation=nn.ReLU,
            dropout=self.dropout,
        )
        self.conv2 = GCN_Block_GINN(
            in_channels=self.dim_hidden,
            out_channels=out_feats,
            activation=nn.Identity,
            dropout=self.dropout,
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, a):
        x = self.conv1(x, a)
        x = self.conv2(x, a)
        return x

    def fit(self, X: Tensor, A: Tensor, Y: Tensor):
        n, d = X.shape
        assert A.shape[0] == n
        n_train = len(Y)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        for epoch in range(self.epochs):
            self.train()
            preds = self.forward(X, A)
            loss = self.loss(preds[:n_train], Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self.verbose and (1 + epoch) % self.valid_freq == 0:
                logging.info(f"epoch {epoch:04} loss {loss:.4f}")

        self.eval()
        self.preds = self.forward(X, A)
        self.preds_test = self.preds[n_train:]
        self.preds_test = F.softmax(self.preds_test, 1)
