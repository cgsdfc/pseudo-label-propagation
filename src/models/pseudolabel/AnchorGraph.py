import torch
from torch import max as EPS_max
from torch import tensor

from src.utils.torch_utils import *


def Kernel_To_DoubleRandom(K: Tensor):
    """
    Input: K, symmetric, non-neg.
    Output: Double random matrix.
    """
    return K / torch.sum(K) * K.shape[0]


def multiview_mse_loss(inputs, targets) -> Tensor:
    "多视角 MSE-loss"
    loss = [F.mse_loss(xbar, x) for xbar, x in zip(inputs, targets)]
    return sum(loss) / len(loss)


def EuclideanDistances(a, b, squared=True):
    sq_a = a**2
    sum_sq_a = torch.sum(sq_a, dim=1).unsqueeze(1)  # m->[m, 1]
    sq_b = b**2
    sum_sq_b = torch.sum(sq_b, dim=1).unsqueeze(0)  # n->[1, n]
    bt = b.t()
    ans = sum_sq_a + sum_sq_b - 2 * a.mm(bt)
    if squared:
        return ans
    return torch.sqrt(ans)


def Gaussian_Kernel(x, y, k):
    D = EuclideanDistances(x, y, squared=True)
    sigma = torch.mean(D)
    K = torch.exp(-D / (EPS_max(sigma)))
    K = KNN_pruning(K, k)
    return K


def Student_Kernel(x, y, alpha=1):
    D = EuclideanDistances(x, y)
    K = torch.pow(1 + D / alpha, -(1 + alpha) / 2)
    return K


def Fuse_AnchorGraph(Z1, Z2, nPaired):
    Z = torch.cat(
        [(Z1[:nPaired, :] + Z2[:nPaired, :]) / 2, Z1[nPaired:, :], Z2[nPaired:, :]],
        dim=0,
    )
    return Z


def Make_AnchorGraph(allSmp, anchor, kernel, **kwds):
    K = kernel(allSmp, anchor, **kwds)
    return RowNorm(K) @ K


def KNN_pruning(Z, k):
    Z = torch.clone(Z)
    idx = torch.argsort(-Z, 1)
    for i in range(Z.shape[0]):
        Z[i, idx[i, k + 1 :]] = 0
    Z = RowNorm(Z) @ Z
    return Z


def Anchor_To_Full(Z):
    """
    Returns double random matrix
    """
    S = Z @ ColNorm(Z) @ Z.T
    return S


def RowNorm(x):
    "Row normalizer"
    return torch.diag(1 / EPS_max(torch.sum(x, 1)))


def ColNorm(x):
    "Column normalizer"
    return torch.diag(1 / EPS_max(torch.sum(x, 0)))


def Make_DiagMask_like(x: tensor):
    """
    产生一个对角Mask矩阵，即对角线为0，其他为1，与之相乘能把对角线元素置为零。
    """
    return Make_DiagMask(x.shape[0]).to(x.device)


def Make_DiagMask(n):
    return 1 - torch.eye(n)


def DoubleRandom_To_Joint(S, mask_diag=True):
    if mask_diag:
        Mask = Make_DiagMask_like(S)
        S = S * Mask
    S = S / torch.sum(S)
    return EPS_max(S)


def Make_Joint(allSmp, anchor, kernel, **kwds):
    Z = Make_AnchorGraph(allSmp, anchor, kernel, **kwds)
    S = Anchor_To_Full(Z)
    assert Is_Double_Random(S)
    S = DoubleRandom_To_Joint(S)
    assert Is_Joint_Distribution(S)
    return S


def Reorder_Graph(S, sort_idx):
    return S[:, sort_idx][sort_idx, :]


def Is_Joint_Distribution(x):
    """
    Joint distribution matrix:
    1. symmetric
    2. non-neg elements
    3. sum of all elem equals one

    From ptsne
    """
    normalized = torch.allclose(torch.sum(x), torch.tensor(1.0, dtype=x.dtype))
    symmetric = torch.allclose(x, x.T)
    positive = torch.all(x >= 0)
    # print(normalized, symmetric, positive)
    return all([normalized, symmetric, positive])


def Is_Double_Random(x):
    """
    Double random matrix:
    1. symmetric
    2. non-neg elements
    3. row-normalized (sum of any row is one)

    From anchor-based/APMC
    """
    normalized = torch.allclose(torch.sum(x, 1), torch.tensor(1.0, dtype=x.dtype))
    symmetric = torch.allclose(x, x.T)
    positive = torch.all(x >= 0)
    # print(normalized, symmetric, positive)
    return all([normalized, symmetric, positive])


def Make_P_joint(allSmp, anchor, k=7):
    return Make_Joint(allSmp, anchor, Gaussian_Kernel, k=k)


def Make_Q_joint(allSmp, anchor):
    return Make_Joint(allSmp, anchor, Student_Kernel, alpha=1)


def Gaussian_AnchorGraph_IMC(X1_ank, X2_ank, X1_all, X2_all, pairedNum, sort_idx, k):
    # sort_idx 是必须的，因为IMC会把sample顺序设置为paired在前，single在后。
    Z1 = Make_AnchorGraph(X1_all, X1_ank, Gaussian_Kernel, k=k)
    Z2 = Make_AnchorGraph(X2_all, X2_ank, Gaussian_Kernel, k=k)
    Z = Fuse_AnchorGraph(Z1, Z2, pairedNum)
    S = Anchor_To_Full(Z)
    S = S[:, sort_idx][sort_idx, :]
    S = DoubleRandom_To_Joint(S)
    return S, Z


def make_joint_anchor(Z):
    "从锚点图Z中得到p_joint"
    S = Anchor_To_Full(Z)  # 得到完整图。
    S = Make_DiagMask_like(S) * S  # 去掉对角线。
    S = S / torch.sum(S)  # 归一化。
    return EPS_max(S)  # 数值稳定。
