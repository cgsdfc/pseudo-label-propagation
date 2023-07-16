from math import log2
from typing import Optional

import torch
from torch import Tensor, device, eye, isnan
from torch import ones, tensor

from .ptsne_utils import distance_functions, entropy
from src.utils.torch_utils import EPS_like, EPS_max


def make_p_joint_TSNE(X_train: Tensor, perplexity, device=None):
    if device is None:
        device = X_train.device
    p_cond = calculate_optimized_p_cond(
        X_train, target_entropy=log2(perplexity), dev=device
    )
    p_joint = make_joint(p_cond)
    return p_joint


def calculate_optimized_p_cond(
    input_points: tensor,
    target_entropy: float,
    dist_func: str = "euc",
    tol: float = 0.0001,
    max_iter: int = 100,
    min_allowed_sig_sq: float = 0,
    max_allowed_sig_sq: float = 10000,
    dev: str = "cpu",
) -> Optional[tensor]:
    """
    Calculates conditional probability matrix optimized by binary search
    :param input_points: A matrix of input data where every row is a data point
    :param target_entropy: The entropy that every distribution (row) in conditional
    probability matrix will be optimized to match
    :param dist_func: A name for desirable distance function (e.g. "euc", "jaccard" etc)
    :param tol: A small number - tolerance threshold for binary search
    :param max_iter: Maximum number of binary search iterations
    :param min_allowed_sig_sq: Minimum allowed value for the spread of any distribution
    in conditional probability matrix
    :param max_allowed_sig_sq: Maximum allowed value for the spread of any distribution
    in conditional probability matrix
    :param dev: device for tensors (e.g. "cpu" or "cuda")
    :return:
    """
    n_points = input_points.size(0)

    # Calculating distance matrix with the given distance function
    dist_f = distance_functions[dist_func]
    distances = dist_f(input_points)
    diag_mask = (1 - eye(n_points)).to(device(dev))

    # Initializing sigmas
    min_sigma_sq = (min_allowed_sig_sq + 1e-20) * ones(n_points).to(device(dev))
    max_sigma_sq = max_allowed_sig_sq * ones(n_points).to(device(dev))
    sq_sigmas = (min_sigma_sq + max_sigma_sq) / 2

    # Computing conditional probability matrix from distance matrix
    p_cond = get_p_cond(distances, sq_sigmas, diag_mask)

    # Making a vector of differences between target entropy and entropies for all rows in p_cond
    ent_diff = entropy(p_cond) - target_entropy

    # Binary search ends when all entropies match the target entropy
    finished = ent_diff.abs() < tol

    curr_iter = 0
    while not finished.all().item():
        if curr_iter >= max_iter:
            print("Warning! Exceeded max iter.", flush=True)
            # print("Discarding batch")
            return p_cond
        pos_diff = (ent_diff > 0).float()
        neg_diff = (ent_diff <= 0).float()

        max_sigma_sq = pos_diff * sq_sigmas + neg_diff * max_sigma_sq
        min_sigma_sq = pos_diff * min_sigma_sq + neg_diff * sq_sigmas

        sq_sigmas = (
            finished.logical_not() * (min_sigma_sq + max_sigma_sq) / 2
            + finished * sq_sigmas
        )
        p_cond = get_p_cond(distances, sq_sigmas, diag_mask)
        ent_diff = entropy(p_cond) - target_entropy
        finished = ent_diff.abs() < tol
        curr_iter += 1
    if isnan(ent_diff.max()):
        print("Warning! Entropy is nan. Discarding batch", flush=True)
        return
    return p_cond


def get_p_cond(distances: tensor, sigmas_sq: tensor, mask: tensor) -> tensor:
    """
    Calculates conditional probability distribution given distances and squared sigmas
    :param distances: Matrix of squared distances ||x_i - x_j||^2
    :param sigmas_sq: Row vector of squared sigma for each row in distances
    :param mask: A mask tensor to set diagonal elements to zero
    :return: Conditional probability matrix
    """
    logits = -distances / (2 * EPS_max(sigmas_sq).view(-1, 1))
    logits.exp_()
    masked_exp_logits = logits * mask
    normalization = EPS_max(masked_exp_logits.sum(1)).unsqueeze(1)
    return masked_exp_logits / normalization + 1e-10


def get_q_joint(
    emb_points: tensor, dist_func: str = "euc", alpha: int = 1, mask_diag=True
) -> tensor:
    """
    Calculates the joint probability matrix in embedding space.
    :param emb_points: Points in embeddings space
    :param alpha: Number of degrees of freedom in t-distribution
    :param dist_func: A kay name for a distance function
    :return: Joint distribution matrix in emb. space
    """
    n_points = emb_points.size(0)
    dist_f = distance_functions[dist_func]
    distances = dist_f(emb_points) / alpha
    q_joint = (1 + distances).pow(-(1 + alpha) / 2)
    if mask_diag:
        mask = (-eye(n_points) + 1).to(emb_points.device)
        q_joint = q_joint * mask
    q_joint /= q_joint.sum()
    return EPS_max(q_joint)


def make_joint(distr_cond: tensor) -> tensor:
    """
    Makes a joint probability distribution out of conditional distribution
    :param distr_cond: Conditional distribution matrix
    :return: Joint distribution matrix. All values in it sum up to 1.
    Too small values are set to fixed epsilon
    """
    n_points = distr_cond.size(0)
    distr_joint = (distr_cond + distr_cond.t()) / (2 * n_points)
    return EPS_max(distr_joint)


def loss_function(p_joint: Tensor, q_joint: Tensor) -> Tensor:
    """
    Calculates KLDiv between joint distributions in original and embedding space
    :param p_joint:
    :param q_joint:
    :return: KLDiv value
    """
    # TODO Add here alpha gradient calculation too
    # TODO Add L2-penalty for early compression?
    EPS = EPS_like(p_joint)
    return (p_joint * torch.log((p_joint + EPS) / (q_joint + EPS))).sum()
