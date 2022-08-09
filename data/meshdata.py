import argparse
import re
from torch.utils.data import DataLoader
import data
import numpy as np
from typing import Type, Optional, List, Dict


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Creates mini-batch tensors

    :param batch: batch
    :return: mini-batch tensor
    """
    meta = {}
    keys = batch[0].keys()
    for key in keys:
        meta.update({key: np.array([d[key] for d in batch])})
    return meta


def get_dataloader(opt: argparse.Namespace, phase: str, walks=None, mesh_portion: float = 1):
    """
    Generates the mesh dataset and provides an iteration function.

    :param opt: input args
    :param phase: phase train or test
    :param walks: optional number of walks, otherwise use number of walks of input arguments
    :param mesh_portion: mesh portion for remeshing
    """

    def filter_regex(x):
        return re.match(opt.regex, x)

    filter_func = filter_regex if opt.regex else None
    dataset = opt.dataset_mode(opt, phase, filter_func, walks, mesh_portion)

    return DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=not opt.serial_batches,
        num_workers=int(opt.num_threads),
        collate_fn=collate_fn
    )


def get_train_dataset(opt: argparse.Namespace) -> DataLoader:
    """
    Load train dataset

    :param opt: input arguments
    """
    walks = 2 if opt.self_supervised and not opt.pretrained and opt.walks != 2 else opt.walks
    return get_dataloader(opt, 'train', walks=walks)


def get_finetune_dataset(opt: argparse.Namespace) -> DataLoader:
    """
    Load finetune dataset

    :param opt: input arguments
    """

    return get_dataloader(opt, 'train', mesh_portion=opt.finetune)


def get_test_dataset(opt: argparse.Namespace) -> DataLoader:
    """
    Load test dataset

    :param opt: input arguments
    """

    return get_dataloader(opt, 'test')


def get_dataset_class(path: str) -> Optional[Type[data.BaseDataset]]:
    """
    Return the mesh dataset class that passes the sanity check given the path to the dataset folder

    :param path: path to the dataset folder
    :return: mesh dataset class that passes the sanity check, None when no datasets pass
    """

    for dataset in [data.ClassificationData, data.SegmentationData]:
        if dataset.sanity_check(path):
            return dataset
    return None
