import argparse
from typing import Optional, List, Tuple

import torch
from torch import nn
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR

import util
import data
import options
import locale


def get_model(opt: argparse.Namespace, imitate_model: bool = False) -> tuple[
    Module, Optional[Optimizer], Optional[StepLR], int]:
    """
    Initializes the selected model and loads the pre-trained checkpoints if necessary. In addition, the model is
    included in a DataParallel if multiple GPUs are used. The optimizer and the learning rate scheduler are initialized
    with the loaded model and returned.

    :param opt: input arguments
    :param imitate_model: if True, load the imitation model.
    :return: Model, optimizer, learning rate scheduler and start epoch.
    """

    if opt.model not in options.model_map.keys():
        raise ValueError(f'unknown model {opt.model}')

    classify = opt.dataset_mode == data.ClassificationData
    # Load model
    model: nn.Module = options.model_map[opt.model](opt, get_feature_dim(opt), classify)

    # init optimizer
    optimizer, lr_scheduler = None, None
    if opt.phase != 'test':
        optimizer, lr_scheduler = init_data_optimizer(model, opt)

    # wrap model inside self-supervised model
    if opt.self_supervised and not opt.pretrained and opt.phase != 'test':
        model = options.supervised_map[opt.self_supervised](model, opt)
    # prepare gpu(s)
    if opt.device_ids:
        model = model.to(opt.device)
        # use data parallel when multi gpu is supported
        if len(opt.device_ids) > 1:
            model = nn.DataParallel(model, device_ids=opt.device_ids)

    # optionally load model checkpoints
    start_epoch = 0
    load_path = util.get_load_path(opt)
    if opt.phase == 'test' or (opt.phase == 'attack' and not imitate_model):
        util.load_state(model, load_path, opt)
    elif opt.resume or opt.pretrained:
        start_epoch = util.load_state(model, load_path, opt, optimizer if opt.resume else None)

    # log number of parameters
    print(f"# parameters: {locale.format_string('%d', sum(p.numel() for p in model.parameters()), grouping=True)}")

    # switch train/eval mode
    if opt.pretrained and opt.self_supervised:
        model.eval()
    else:
        model.train()

    return model, optimizer, lr_scheduler, start_epoch


def init_data_optimizer(model: nn.Module, opt: argparse.Namespace) -> Tuple[
    torch.optim.Optimizer, torch.optim.lr_scheduler.StepLR]:
    """
    Init Adam optimizer and learning rate scheduler

    :param model: model to use for optimizing
    :param opt: input arguments
    :return: Adam optimizer and learning rate scheduler
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1.0)
    return optimizer, lr_scheduler


def get_feature_dim(opt: argparse.Namespace):
    """
    Return the dimension of the feature vector, which is 3 plus the amount of used eigenvalues and eigenvectors in the
    optional spectral descriptor.

    :param opt: input arguments
    :return:
    """
    return opt.eig_basis + 3


class Projection(nn.Module):
    def __init__(self, layers: List[int], activation: nn.Module = nn.PReLU, norm: nn.Module = nn.LayerNorm):
        """
        Simple MLP

        :param layers: Size of layers.
        :param activation: Activation function. Defaults to nn.PReLU.
        :param norm: Normalization function. Defaults to LayerNorm
        """
        super().__init__()

        self.model = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(in_features=layers[i - 1], out_features=layers[i]),
                norm(layers[i]),
                activation())
                for i in range(1, len(layers))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
