import argparse

import numpy as np
import torch
import os
import random

from typing import List


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def set_global_seed(seed: int) -> None:
    """
    Sets the random seed for the modules torch, numpy and random

    :param seed: seed
    """

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def color_print(color: str, message: str) -> None:
    """
    Prints the message with a certain color

    :param color: color of the message. Must be a color from bcolors
    :param message: printed message
    """
    color = color.upper()
    print(f'{getattr(bcolors, color)}{color}: {message}{bcolors.ENDC}')


def ensure_dir_exists(directory: str) -> None:
    """
    Creates new folder with the given path if the folder does not exist yet

    :param directory: path to the directory
    """

    if os.path.basename(directory).endswith('.pt'):
        directory = os.path.dirname(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_dir_name(opt: argparse.Namespace) -> List[str]:
    """
    Return the base folder with dataset_name/model/supervised or dataset_name/model/pretrain

    :param opt: input arguments
    :return: base folder with dataset_name/model/supervised or dataset_name/model/pretrain
    """
    dataset_name = os.path.basename(opt.dataroot)
    train_approach = 'pretrain' if opt.self_supervised else 'supervised'
    return [dataset_name, opt.model, train_approach]


def get_save_path(opt: argparse.Namespace) -> str:
    """
    Return the save path of the checkpoints for the current input arguments. The filename corresponds to the name
    parameter, with a suffix '_finetuned' if the self-supervised model is finetuned or tested. If the model is imitated,
    the suffix '_imitated' is also added. The default path for the checkpoints are

    output/store/dataset_name/model/supervised|pretrain/name.pt

    :param opt: input arguments
    :return: save path of the checkpoints
    """
    name = opt.name
    if (opt.phase == 'test' or opt.pretrained) and opt.self_supervised:
        name += '_finetuned'
    if opt.phase == 'attack':
        name += '_imitated'

    filepath = os.sep.join([opt.output_dir, opt.checkpoint_dir, *get_dir_name(opt), name])
    filepath_with_extension = filepath + '.pt'
    return filepath_with_extension


def get_load_path(opt: argparse.Namespace) -> str:
    """
    Return the load path of the checkpoints for the current input arguments. The filename corresponds to the name
    parameter, with a suffix '_finetuned' if the self-supervised model is tested or the finetuning is resumed. If the
    model is imitated, the suffix '_imitated' is also added. The default path for the checkpoints are

    output/store/dataset_name/model/supervised|pretrain/name.pt

    :param opt: input arguments
    :return: load path of the checkpoints
    """

    name = opt.name
    if opt.self_supervised and (opt.phase == 'test' or (opt.pretrained and opt.resume)):
        name += '_finetuned'

    if opt.phase == 'attack' and opt.pretrained:
        name += '_imitated'

    filepath = os.sep.join([opt.output_dir, opt.checkpoint_dir, *get_dir_name(opt), name + '.pt'])
    ensure_dir_exists(filepath)
    return filepath


def get_vis_path(opt: argparse.Namespace) -> str:
    """
    Returns the path for the visualization of the meshes, which is by default

    output/visualization/dataset_name/model/supervised|pretrain/name/...

    :param opt: input arguments
    :return: visualization path
    """
    filepath = os.sep.join([opt.output_dir, opt.visualization_dir, *get_dir_name(opt), opt.name])
    ensure_dir_exists(filepath)
    return filepath


def get_log_path(opt: argparse.Namespace) -> str:
    """
    Returns the path for the tensorboard logfile. The filename is the name of the run with the current time as suffix.
    If the training of the model is resumed, the time of the initial training is appended as suffix, so that the log
    file can be continued. The default log path is

    runs/dataset_name/model/supervised|pretrain/name_MONTH-DAY-YEAR_HOURS-MINUTES-SECONDS

    :param opt: input arguments
    :return: log path
    """
    if opt.resume:
        name = opt.name
    else:
        name = f'{opt.name}_{opt.start_time}'
    filepath = os.sep.join([opt.log_dir, *get_dir_name(opt), name])
    ensure_dir_exists(filepath)
    return filepath


def union_walks(inputs: torch.Tensor) -> torch.Tensor:
    """
    Reshapes an input with the shape (batch_size, walks, ...) into the shape (batch_size * walks, ...)

    :param inputs: input tensor
    :return: input tensor reshaped to (batch_size * walks, ...)
    """
    return inputs.reshape(inputs.shape[0] * inputs.shape[1], *inputs.shape[2:])


def segregate_walks(inputs: torch.Tensor, walks: int) -> torch.Tensor:
    """
    Reshapes an input with the shape (batch_size * walks, ...) into the shape (batch_size, walks, ...)

    :param inputs: input tensor
    :param walks: amount of walks
    :return: input tensor reshaped to (batch_size, walks, ...)
    """

    return inputs.reshape(inputs.shape[0] // walks, walks, *inputs.shape[1:])
