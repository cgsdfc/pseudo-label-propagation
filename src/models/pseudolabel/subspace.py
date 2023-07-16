import logging
from typing import List

from src.data.dataset import PartialMultiviewDataset
from src.utils.torch_utils import *

from .AnchorGraph import Gaussian_Kernel, Make_DiagMask, multiview_mse_loss
from .metrics import Evaluate_Graph, KMeans_Evaluate


class Model_Subspace_SingleView(nn.Module):
    """
    （稀疏）子空间学习模型
    """

    def __init__(
        self,
        sampleNum: int,
        data,
        X: Tensor = None,
        lr: float = 0.1,
        epochs: int = 100,
        valid_freq=10,
        verbose=False,
    ):
        super().__init__()
        self.lr = lr
        self.epochs = epochs
        self.data = data
        self.X = X
        self.verbose = verbose
        self.valid_freq = valid_freq
        self.params = nn.Parameter(torch.empty(sampleNum, sampleNum))
        # 只有Parameter才会被move。
        # 只保留一份副本，节省内存。
        self.diag_mask = nn.Parameter(Make_DiagMask(sampleNum), requires_grad=False)
        nn.init.normal_(self.params)

    def forward_subspace(self):
        logits = torch.exp(self.params)
        logits = logits * self.diag_mask
        normalization = EPS_max(logits.sum(1)).unsqueeze(1)
        S = logits / normalization
        return S

    def forward(self, x):
        S = self.forward_subspace()
        xbar = S @ x
        loss = F.mse_loss(xbar, x)
        return loss

    def fit(self):
        """
        训练子空间学习模型，学到一个概率图S，本身可直接用于谱聚类，
        也可以作为PTSNE的输入图，进一步学习，提升性能。
        """
        config = self
        X = self.X
        model = self
        optim = Adam(model.parameters(), lr=config.lr)

        for epoch in range(config.epochs):
            loss = model.forward(X)
            optim.zero_grad()
            loss.backward()
            optim.step()
            if (1 + epoch) % config.valid_freq == 0 and config.verbose:
                logging.info(f"epoch {epoch:04d} loss {loss:.6f}")

        with torch.no_grad():
            model.eval()
            self.S = model.forward_subspace()


class Model_Simple_MVC(nn.Module):
    """
    几种简单IMC方法：
    将数据进行简单补全后（均值补全），
    1. Best Single View（BSV）：最佳单视角性能；
    2. Concat：全部特征连接一起后进行KMeans；
    3. SC-A：各视角的高斯核图的平均值做谱聚类；
    4. SC-C：各视角的特征连接一起后得到高斯核图，然后谱聚类；
    S_global is row normalized.
    """

    def __init__(
        self,
        *,
        X_list: List[Tensor],
        data: PartialMultiviewDataset,
        method: str,
        k: int = 5,
        verbose,
    ):
        super().__init__()
        self.data = data
        self.x_list = X_list
        self.concat = self.spectral_concat = self.spectral_average = 0
        self.verbose = verbose

        if method == "Concat":
            self.concat = 1
        elif method == "SC-A":
            self.spectral_average = 1
        elif method == "SC-C":
            self.spectral_concat = 1
        else:
            raise ValueError(method)

        self.x_concat = None
        self.S = None
        self.k = k
        self.embeddings = None  # 嵌入，表示。
        self.metrics = None

    def _concat_features(self):
        self.x_concat = torch.cat(self.x_list, 1)

    def _compute_graph(self):
        if self.spectral_average:
            S_list = []
            for x in self.x_list:
                S_list.append(Gaussian_Kernel(x, x, k=self.k))
            self.S = sum(S_list) / len(S_list)
        else:
            # 连接图
            self.S = Gaussian_Kernel(self.x_concat, self.x_concat, k=self.k)
        self.S_global = self.S

    def fit(self):
        if self.concat or self.spectral_concat:
            self._concat_features()

        if self.concat:
            self.ypred = KMeans_Evaluate(
                self.x_concat,
                self.data,
                return_ypred=True,
                return_metrics=False,
            )[0]
            self.embeddings = self.x_concat

        elif self.spectral_concat or self.spectral_average:
            self._compute_graph()
            self.embeddings, self.ypred = Evaluate_Graph(
                self.data,
                S=self.S,
                type="regular",
                return_spectral_embedding=True,
                return_ypred=True,
                return_metrics=False,
            )


class Model_Subspace_MVC(nn.Module):
    """
    多视角稀疏子空间模型。

    局部子空间（local subspace）：即各个视角分配一个独立的子空间矩阵。
    全局子空间（global subspace）：即所有视角共享的一个全局子空间矩阵。

    损失函数分为两部分：

    1. loss-local：利用局部子空间的自表达性质。
        loss_local := 1/N_v * \sum_{v=1}^N_v ||S_v @ X_v - X_v||_F^2

    2. loss-global：利用全局子空间的自表达性质。
        loss_global := 1/N_v * \sum_{v=1}^N_v ||S @ X_v - X_v||_F^2
        s.t., S = 1/N_v * \sum_{v=1}^N_v S_v

    总的损失函数：
        loss := loss_local + global_loss_weight * loss_global
    """

    def __init__(
        self,
        data,  # Metainfo of dataset.
        X_list: "List[Tensor]",
        global_loss_weight: float = 0.1,
        epochs: int = 200,
        lr: float = 0.1,
        valid_freq: int = 10,
        verbose=True,
    ):
        super().__init__()
        self.data = data
        self.sampleNum = X_list[0].shape[0]
        self.viewNum = data.viewNum
        self.lr = lr
        self.epochs = epochs
        self.valid_freq = valid_freq
        self.verbose = verbose

        self.x_list = X_list
        self.global_loss_weight = global_loss_weight
        self.subspace_model_list = nn.ModuleList()
        for _ in range(self.viewNum):
            model = Model_Subspace_SingleView(
                sampleNum=self.sampleNum,
                data=data,
            )
            self.subspace_model_list.append(model)

    def forward_subspace(self):
        """
        得到各视角的local subspace和公共的global subspace。
        Returns S_global, S_local
        """
        S_local = []
        for (
            subspace_model
        ) in self.subspace_model_list:  # type: Model_Subspace_SingleView
            S_local.append(subspace_model.forward_subspace())
        S_global = sum(S_local) / self.viewNum
        return S_global, S_local

    def forward(self):
        x_list = self.x_list
        loss = 0

        S_global, S_local = self.forward_subspace()
        xbar_local = []
        xbar_global = []
        for S_loc, x in zip(S_local, x_list):
            xbar_local.append(S_loc @ x)
            xbar_global.append(S_global @ x)

        loss_local = multiview_mse_loss(xbar_local, x_list)
        loss_global = multiview_mse_loss(xbar_global, x_list) * self.global_loss_weight
        loss = loss_local + loss_global

        return loss

    def fit(self):
        """
        训练子空间学习模型，学到一个概率图S，本身可直接用于谱聚类，
        也可以作为PTSNE的输入图，进一步学习，提升性能。
        """
        print(f"fitting multiview subspace model")
        data = self.data
        config = self
        model = self
        optim = Adam(model.parameters(), lr=config.lr)

        for epoch in range(config.epochs):
            model.train()
            loss = model.forward()
            optim.zero_grad()
            loss.backward()
            optim.step()
            if self.verbose and (1 + epoch) % config.valid_freq == 0:
                model.eval()
                with torch.no_grad():
                    S_global, S_local = model.forward_subspace()
                logging.info(f"epoch {epoch:04d} {loss:.6f}")

        with torch.no_grad():
            model.eval()
            self.S_global, S_local = model.forward_subspace()
            self.embeddings, self.ypred = Evaluate_Graph(
                data,
                S=S_global,
                type="regular",
                return_spectral_embedding=True,
                return_metrics=False,
                return_ypred=True,
            )
