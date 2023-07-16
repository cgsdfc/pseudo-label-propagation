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

from torch_geometric.nn.dense import DenseGCNConv

from ...utils.torch_utils import F, Tensor, nn, torch


class GCN_Decoder_SDIMC(nn.Module):
    """
    仿照 SDIMC 的解码器(GCN)。
    架构如下：
    GCN_Decoder(
        BN(c),
        GCN(c, 0.8mv),
        ReLU(),
        BN(0.8mv),
        GCN(0.8mv, mv),
    ), where mv is view_dim, c is clusterNum.
    """

    def __init__(self, view_dim, clusterNum):
        super().__init__()
        self.mv = view_dim
        self.c = clusterNum
        self.mv_ = int(round(0.8 * self.mv))
        self.bn1 = nn.BatchNorm1d(self.c)
        self.bn2 = nn.BatchNorm1d(self.mv_)
        self.conv1 = DenseGCNConv(self.c, self.mv_)
        self.conv2 = DenseGCNConv(self.mv_, self.mv)

    def forward(self, x, s):
        x = self.bn1(x)
        x = F.relu(self.conv1(x, s)).squeeze()
        x = self.conv2(self.bn2(x), s)
        x = torch.sigmoid(x).squeeze()
        return x


class Imputer(nn.Module):
    """
    解码器，不来自任何文献，只是好使。
    """

    def __init__(self, dim_output, dim_emb):
        super().__init__()
        hiddens = int(round(0.8 * dim_output))
        self.head = nn.Sequential(
            nn.Linear(dim_emb, hiddens),
            nn.ReLU(),
            nn.Linear(hiddens, dim_output),
            nn.Sigmoid(),
        )

    def forward(self, h):
        xbar = self.head(h)
        return xbar


class NeuralMapper(nn.Module):
    """
    来自 Parametric t-SNE （PTSNE）的编码器。
    """

    def __init__(self, dim_input, dim_emb=2):
        super().__init__()
        # 这个4层MLP的结构和CDIMC的是一样的，不过多了BN。
        dim1 = int(round(dim_input * 0.8))
        dim2 = int(round(dim_input * 0.5))
        self.linear_1 = nn.Linear(dim_input, dim1)
        self.bn_1 = nn.BatchNorm1d(dim1)
        self.linear_2 = nn.Linear(dim1, dim1)
        self.bn_2 = nn.BatchNorm1d(dim1)
        self.linear_3 = nn.Linear(dim1, dim2)
        self.bn_3 = nn.BatchNorm1d(dim2)
        self.linear_4 = nn.Linear(dim2, dim_emb)
        self.relu = nn.ReLU()
        self.apply(weights_init)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.bn_1(x)
        x = self.linear_2(self.relu(x))
        x = self.bn_2(x)
        x = self.linear_3(self.relu(x))
        x = self.bn_3(x)
        x = self.linear_4(self.relu(x))
        return x


class MLP_Decoder_SDIMC(nn.Module):
    """
    来自 SDIMC 的解码器(三层 MLP)。
    架构如下：
    MLP_Decoder(
        Linear(c, c),
        ReLU(),
        Linear(c, 0.8mv),
        ReLU(),
        Linear(0.8mv, mv),
    )
    """

    def __init__(self, view_dim, clusterNum):
        super().__init__()
        self.mv = view_dim
        self.c = clusterNum
        self.mv_ = int(round(0.8 * self.mv))
        self.network = nn.Sequential(
            nn.Linear(in_features=self.c, out_features=self.c),
            nn.ReLU(),
            nn.Linear(in_features=self.c, out_features=self.mv_),
            nn.ReLU(),
            nn.Linear(in_features=self.mv_, out_features=self.mv),
        )

    def forward(self, h):
        xbar = self.network(h)
        return xbar


class GCN_Encoder_SDIMC(nn.Module):
    """
    来自 SDIMC（Structural Deep Incomplete Multi-view Clustering Network）的 GCN 编码器（双层）
    架构如下：
    GCN_Encoder(
        GCN(mv, 0.8mv),
        BN(),
        ReLU(),
        GCN(0.8mv, c),
        BN(),
    ), where mv is view_dim, c is clusterNum.
    """

    def __init__(self, view_dim, clusterNum):
        super().__init__()
        self.mv = view_dim
        self.c = clusterNum
        self.mv_ = int(round(0.8 * self.mv))
        self.conv1 = DenseGCNConv(in_channels=self.mv, out_channels=self.mv_)
        # self.bn1 = nn.BatchNorm1d(num_features=self.mv_)
        self.relu = nn.ReLU()
        self.conv2 = DenseGCNConv(in_channels=self.mv_, out_channels=self.c)
        # self.bn2 = nn.BatchNorm1d(num_features=self.c)

    def forward(self, x, a):
        x = self.conv1(x, a).squeeze()
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x, a).squeeze()
        # x = self.bn2(x)
        return x


class GCN_Block_GINN(nn.Module):
    """
    GINN 单块。一般两个这样的单块构成一个GCN编码器。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.2,
        activation=nn.ReLU,
    ) -> None:
        super().__init__()
        self.conv = DenseGCNConv(in_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation()

    def forward(self, x, adj):
        x = self.dropout(x)
        # 务必记住！DenseGCN 需要squeeze！！
        x = self.conv(x, adj).squeeze()
        x = self.activation(x)
        return x


class GCN_Encoder_GINN(nn.Module):
    """
    来自 GINN（Missing Data Imputation with Adversarially-trained Graph Convolutional Networks）
    的 GCN 编码器（单层）。
    """

    def __init__(self, dim_emb, dim_input, dropout=0.2) -> None:
        super().__init__()
        self.dim_emb = dim_emb
        self.dim_input = dim_input
        self.dropout = dropout
        self.dim_hidden = int(round(0.8 * self.dim_input))
        self.conv1 = GCN_Block_GINN(
            in_channels=self.dim_input,
            out_channels=self.dim_hidden,
            activation=nn.ReLU,
            dropout=self.dropout,
        )
        self.conv2 = GCN_Block_GINN(
            in_channels=self.dim_hidden,
            out_channels=dim_emb,
            activation=nn.Identity,
            dropout=self.dropout,
        )

    def forward(self, x, a):
        x = self.conv1(x, a)
        x = self.conv2(x, a)
        return x


class ViewAE(nn.Module):
    """
    来自 CDIMC 的自编码器。

    encoder0 每个编码器是4层fc，解码器也是4层fc，只有relu，没有bn和dropout。

    [mv,0.8mv], [0.8mv, 0.8mv], [0.8mv, 1500], [1500, k]

    decoder0 解码器的结构是和编码器相反的，
    但是注意，解码器有5层，而编码器只有4层。

    [k, 1500, 0.8mv, 0.8mv]

    [k, k], [k, 1500], [1500, 0.8mv], [0.8mv, 0.8mv], [0.8mv, mv]
    """

    def __init__(self, n_input, n_z):
        super(ViewAE, self).__init__()
        self.dim1 = int(round(0.8 * n_input))
        self.dim2 = 1500
        self.n_input = n_input
        self.n_z = n_z

        self.encoder = nn.Sequential(
            nn.Linear(n_input, self.dim1),
            nn.ReLU(),
            nn.BatchNorm1d(self.dim1),
            nn.Linear(self.dim1, self.dim1),
            nn.ReLU(),
            nn.BatchNorm1d(self.dim1),
            nn.Linear(self.dim1, self.dim2),
            nn.ReLU(),
            nn.BatchNorm1d(self.dim2),
            nn.Linear(self.dim2, n_z),
        )

        self.decoder = nn.Sequential(
            nn.Linear(n_z, n_z),  # [k,k]
            nn.ReLU(),
            nn.BatchNorm1d(n_z),
            nn.Linear(n_z, self.dim2),  # [k,1500]
            nn.ReLU(),
            nn.BatchNorm1d(self.dim2),
            nn.Linear(self.dim2, self.dim1),
            nn.ReLU(),
            nn.BatchNorm1d(self.dim1),
            nn.Linear(self.dim1, n_input),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)
                nn.init.xavier_uniform_(m.weight)


def multiview_mse_loss(inputs, targets) -> Tensor:
    "多视角 MSE-loss"
    loss = [F.mse_loss(xbar, x) for xbar, x in zip(inputs, targets)]
    return sum(loss) / len(loss)


def imputation_loss(inputs, targets, mask) -> Tensor:
    "多视角 MSE-loss with mask"
    loss = [F.mse_loss(xbar[m], x[m]) for xbar, x, m in zip(inputs, targets, mask.T)]
    return sum(loss) / len(loss)


def weights_init(m: nn.Module) -> None:
    if isinstance(m, (nn.Linear,)):
        nn.init.kaiming_normal_(m.weight.data)
        m.bias.data.fill_(0.01)
    elif isinstance(m, (nn.BatchNorm1d,)):
        m.weight.data.fill_(1.0)
        m.bias.data.fill_(0)


class GCN_Model(nn.Module):
    """
    来自 GINN（Missing Data Imputation with Adversarially-trained Graph Convolutional Networks）
    的 GCN 编码器（单层）。
    """

    def __init__(
        self,
        in_feats,
        out_feats,
        dropout=0.2,
        classify=True,
        lr=0.01,
        epochs=100,
        valid_freq=10,
    ) -> None:
        super().__init__()
        self.dim_emb = out_feats
        self.dim_input = in_feats
        self.dropout = dropout
        self.dim_hidden = int(round(0.8 * self.dim_input))
        self.lr = lr
        self.epochs = epochs
        self.valid_freq = valid_freq

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
        if classify:
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.MSELoss()

    def forward(self, x, a):
        x = self.conv1(x, a)
        x = self.conv2(x, a)
        return x

    def fit(self, X: Tensor, A: Tensor, M: Tensor, Y: Tensor):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        for epoch in range(self.epochs):
            preds = self.forward(X, A)
            loss = self.loss(preds[M], Y[M])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (1 + epoch) % self.valid_freq:
                logging.info(f"epoch {epoch:04} loss {loss:.4f}")
