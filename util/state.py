import argparse
import torch
from torch import nn as nn
import util
from typing import Optional


def save_state(path: str, model: nn.Module, epoch: int, opt: argparse.Namespace,
               optimizer: torch.optim.Optimizer) -> None:
    """
    Saves model parameters, optimizer parameters, opt and current epoch to disk.

    :param path: save path
    :param model: model
    :param epoch: current epoch
    :param opt: input arguments
    :param optimizer: optimizer
    """
    if isinstance(model, nn.DataParallel):
        model = model.module

    state = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'opt': opt
    }
    torch.save(state, path)


def load_model_opt(target: argparse.Namespace, source: argparse.Namespace) -> None:
    """
    Copies all attributes of one namespace to the other, except for the number of walks and phase attribute.

    :param target: target in which the items from source are placed
    :param source: source namespace
    """
    skipped_keys = ['walks', 'phase']
    for key, value in vars(source).items():
        if hasattr(target, key) and getattr(target, key) != value and key not in skipped_keys:
            setattr(target, key, value)


def load_opt(opt: argparse.Namespace) -> None:
    """
    Restores the opt of the pretrained model

    :param opt: input arguments
    """
    pretrained_dict = torch.load(util.get_load_path(opt))
    if 'opt' in pretrained_dict:
        load_model_opt(opt, pretrained_dict['opt'])


def load_state(model: nn.Module, path: str, opt: argparse.Namespace,
               optimizer: Optional[torch.optim.Optimizer] = None) -> int:
    """
    Loads the save state dict for a given mesh

    :param model: model to be loaded
    :param path: path to the saved state dict
    :param opt: input arguments
    :param optimizer: optional optimizer
    :return: current epoch
    """
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    pretrained_dict = torch.load(path)
    state_dict = pretrained_dict['state_dict']
    if opt.self_supervised and not opt.resume and (opt.pretrained or opt.phase == 'train'):
        state_dict = load_encoder(state_dict, model)

    model.load_state_dict(state_dict)

    if optimizer:
        optimizer.load_state_dict(pretrained_dict['optimizer'])
    print(util.bcolors.OKGREEN + "Loaded", path + "." + util.bcolors.ENDC)
    return pretrained_dict['epoch'] if opt.resume else 0


def load_encoder(state_dict, model, prefix='encoder'):
    """
    Extracts the encoder of a given state dict and resets the head parameters

    :param state_dict: state dict containing the encoder
    :param model: model to be loaded
    :param prefix: prefix of the encoder in the state dict (default: encoder)
    :return: state dict containing the encoder
    """

    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith(prefix) and not k.startswith(prefix + '.fc'):
            # remove prefix
            state_dict[k[len(prefix + "."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    for pname, param in model.named_parameters():
        if pname not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    # init fc layer
    state_dict['fc.weight'] = model.fc.weight.data.normal_(mean=0.0, std=0.01)
    state_dict['fc.bias'] = model.fc.bias.data.zero_()
    return state_dict
