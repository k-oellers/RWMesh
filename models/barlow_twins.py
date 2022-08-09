import argparse

import torch
import torch.nn as nn
import data
from typing import Tuple


class ProjectionHead(nn.Module):
    """
    Simple linear projection head
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, segmentation: bool = True) -> None:
        super().__init__()

        self.proj1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.ReLU()
        self.proj2 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.segmentation = segmentation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj1(x)
        if self.segmentation:
            x = x.permute(0, 2, 1)
        x = self.norm(x)
        if self.segmentation:
            x = x.permute(0, 2, 1)
        x = self.activation(x)
        x = self.proj2(x)
        return x


class BarlowTwins(nn.Module):
    """
    Barlow twins implementation https://arxiv.org/pdf/2103.03230.pdf.
    """

    def __init__(self, encoder: nn.Module, opt: argparse.Namespace) -> None:
        """
        :param encoder: used encoder
        :param opt: input arguments
        """
        super(BarlowTwins, self).__init__()
        self.encoder = encoder
        self.opt = opt
        self.projection_head = ProjectionHead(input_dim=opt.model_size, hidden_dim=opt.model_size,
                                              output_dim=opt.embedding_size,
                                              segmentation=opt.dataset_mode == data.SegmentationData).to(opt.device)

    def cross_correlation_matrix(self, inputs: torch.Tensor, pads: torch.Tensor) -> torch.Tensor:
        """
        Calculates the normalized cross correlation matrix

        :param inputs: inputs with shape (batch_size, walks, vertices, dim)
        :param pads: padding for transformer encoder with shape (batch_size, walks, vertices)
        :return: normalized cross correlation matrix
        """

        with torch.no_grad():
            q, k = self.forward(inputs, pads)
            c = get_cross_correlation(q, k)
            if len(c.shape) > 2:
                c = c[-1]
            # normalize
            c -= torch.min(c)
            c_norm = c / torch.max(c)
            return c_norm

    def forward(self, inputs: torch.Tensor, pads: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # get embeddings and use projection head on embeddings
        return tuple(self.projection_head(self.encoder(inputs[:, i], pads[:, i])) for i in range(2))


def get_cross_correlation(z1: torch.Tensor, z2: torch.Tensor, eps: float = 1e-15) -> torch.Tensor:
    """
    Calculates the cross correlation matrix for two embedding tensor with shape N x D, where N is the batch size and D
    is output dim of projection head.

    :param z1: first embedding tensor
    :param z2: second embedding tensor
    :param eps: small epsilon to avoid division by zero. Defaults to 1e-15
    :return: cross correlation matrix
    """
    #
    z1_norm = (z1 - torch.mean(z1, dim=0)) / (torch.std(z1, dim=0) + eps)
    z2_norm = (z2 - torch.mean(z2, dim=0)) / (torch.std(z2, dim=0) + eps)
    if len(z1.shape) > 2:
        # segmentation
        return torch.matmul(z1_norm.permute(1, 2, 0), z2_norm.permute(1, 0, 2)) / z1_norm.shape[0]
    else:
        # classification
        return torch.matmul(z1_norm.T, z2_norm) / z1_norm.shape[0]


def off_diagonal_ele(src: torch.Tensor) -> torch.Tensor:
    """
    Returns a flattened view of the off-diagonal elements of a square matrix.
    It uses the mean when the dimension of the input is three-dimensional.
    inspired by: https://github.com/facebookresearch/barlowtwins/blob/main/main.py

    :param src: input tensor
    :return: flattened off-diagonal values
    """
    if len(src.shape) > 2:
        # segmentation
        b, n, m = src.shape
        assert n == m
        dia = torch.cat([src.flatten()[i * n * m: (i + 1) * n * m - 1] for i in range(b)])
        return dia.view(b, n - 1, n + 1)[:, :, 1:].flatten()
    else:
        # classification
        n, m = src.shape
        assert n == m
        return src.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwinsLoss(nn.Module):
    """
    Implementation for barlow twin loss
    """

    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Calculates the barlow twin loss

        :param z1: first input embedding
        :param z2: secong input embedding
        :return: barlow twin loss
        """
        cross_corr = get_cross_correlation(z1, z2)
        if len(cross_corr.shape) > 2:
            # segmentation
            dia = torch.diagonal(cross_corr, dim1=1, dim2=2).mean(dim=0)
        else:
            # classification
            dia = torch.diagonal(cross_corr)
        on_diag = dia.add_(-1).pow_(2).sum()
        off_diag = off_diagonal_ele(cross_corr).pow_(2).sum()
        loss = on_diag + self.temperature * off_diag
        return loss
