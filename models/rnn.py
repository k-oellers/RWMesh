import models
import torch.nn as nn
import torch
import argparse
from typing import Optional
import util


class RNNModel(nn.Module):
    """
    Reimplementation of the RNN encoder of the MeshWalker approach.
    """

    def __init__(self, opt: argparse.Namespace, feature_dim: int = 3, classify: bool = True):
        """
        :param opt: Input arguments
        :param feature_dim: size of the feature dimension. Defaults to 3 for the xyz-Position of the vertices. Can be
        increased by using a spectral descriptor.
        :param classify: if True, use classify mode, if False, use segmentation mode. Defaults to True.
        """
        super().__init__()
        self.linear = models.Projection([feature_dim, opt.model_size * 2, opt.model_size], activation=nn.ReLU)
        gru_layers = [opt.model_size, opt.model_size * 4, opt.model_size * 4, opt.model_size]
        self.grus = nn.Sequential(
            *[nn.Sequential(
                nn.GRU(gru_layers[i - 1], gru_layers[i], batch_first=True).to(opt.device),
                SelectItem(0),
            ) for i in range(1, len(gru_layers))]
        )
        self.fc = nn.Linear(gru_layers[-1], opt.n_classes).to(opt.device)
        self.opt = opt
        self.classify = classify
        self.use_fc = not (opt.self_supervised and (not opt.pretrained and opt.phase != 'test'))

    def forward(self, src: torch.Tensor, pad: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Call the transformer model

        :param src: List of vertices with shape (batch_size, walks, vertices, feature_dim) or
        (batch_size, vertices, feature_dim)
        :return: When classifying: a tensor of shape (batch_size, walks, classes)
                 When segmenting: a tensor of shape (batch_size, walks, vertices, classes)
                 When self-supervised and classifying: a tensor of shape (batch_size, walks, model_size)
                 When self-supervised and segmenting: a tensor of shape (batch_size, walks, vertices, model_size)
        """
        input_shape = src.shape
        # (batch_size, walks, vertices, dim) -> (batch_size * walks, vertices, dim)
        if len(input_shape) > 3:
            src = util.union_walks(src)

        x = self.linear(src)
        x = self.grus(x)
        if self.classify:
            x = x[:, -1, :]

        if self.use_fc:
            x = self.fc(x)

        # (batch_size * walks, vertices, dim) -> (batch_size, walks, vertices, dim)
        if len(input_shape) > 3:
            x = util.segregate_walks(x, input_shape[1])
        return x


def cut_segmentation_sequence(outputs, labels, pad):
    max_len = torch.min(torch.argmax(pad.detach().cpu().long(), dim=1))
    if max_len > 0:
        min_len = torch.div(max_len, 2, rounding_mode='floor')
        outputs = outputs[:, :, min_len:max_len]
        labels = labels[:, min_len:max_len]
    return outputs, labels


class SelectItem(nn.Module):
    """
    Simple PyTorch module for selecting a specific element, useful for sequential modules.
    """

    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]
