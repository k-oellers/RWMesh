import torch
from torch.utils.data import Dataset
import data
import abc
import models
import os
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Callable, Union, Tuple
import trimesh
import argparse
import random


def locals_to_dict(local_values: Dict) -> Dict:
    """


    :param local_values:
    :return:
    """
    return {key: local_values[key] for key in ['walks', 'pads', 'perm', 'labels', 'mesh']}


def shuffle_paths(paths: List[Dict], mesh_portion: float):
    """
    Randomly shuffles a list and returns a percentage of elements

    :param paths: list of mesh paths
    :param mesh_portion: percentage of elements to keep
    :return: shuffled and reduced list
    """
    assert 0 < mesh_portion <= 1
    random.shuffle(paths)
    return paths[:int(len(paths) * mesh_portion)] if mesh_portion < 0 else paths


class BaseDataset(Dataset):
    """
    A base dataset loader for mesh files
    """

    def __init__(self, opt: argparse.Namespace, phase: str, walks=None):
        """
        :param opt: The input arguments
        :param phase: train or test phase of the dataset
        :param walks: number of walks. If not specified or None, take the number of walks from the opt.
        :param mesh_portion: portion of used meshes
        """

        self.opt = opt
        self.phase = phase
        self.root = opt.dataroot
        self.dataset_name = os.path.normpath(self.root).split(os.sep)[-1]
        self.meshes: List[data.Mesh] = []
        self.size = 0
        self.walks = opt.walks if walks is None else walks
        self.sequence_length = self.opt.sequence_length
        self.test_augmentation = [] if not opt.no_rotation else [data.aug_random_rotate]
        self.train_augmentation = self.get_augmentations()
        super(BaseDataset, self).__init__()

    def __len__(self) -> int:
        """
        :return: number of meshes
        """

        return self.size

    def random_walk(self, mesh, walk_length, walks, perm, pads, walk_idx):
        # generate random walk
        walk, success = data.get_random_walk(mesh.adj.tolil().rows, walk_length)
        walk = np.array(walk, dtype=np.int_)
        # real walk length can be shorter than walk_length
        real_walk_length = walk.shape[0]
        # set padding
        pads[walk_idx, :real_walk_length] = np.zeros((real_walk_length,), dtype=np.bool_)
        # set permutation
        perm[walk_idx, :real_walk_length] = walk
        # permute vertices and optional spectral descriptor according to the random walk and apply position mode
        walks[walk_idx, :real_walk_length] = BaseDataset.permute_data_sequence(mesh, walk, self.opt.position_mode)
        return real_walk_length

    def get_meshes(self, paths: List[Dict]) -> None:
        """
        Loads the meshes for all passed paths.
        If a mesh is not found in the cache, it is first preprocessed and stored as a numpy array.
        With the next call they will be loaded from the cache again.

        :param paths: list of dictionary containing the mesh path and optional mesh label
        """

        with tqdm(paths, desc=f'loading {self.phase} dataset of {self.dataset_name}') as t:
            for i, mesh_info in enumerate(t):
                resolutions = [1.] + self.opt.remesh
                path = mesh_info['path']
                # classification data has a label per mesh
                label = mesh_info['label'] if 'label' in mesh_info else None
                for resolution in resolutions:
                    # calc caching path for file
                    cache_path = data.get_cache_path(self.opt.cache_root, self.dataset_name, path,
                                                     self.opt.eig_basis, resolution, self.phase,
                                                     self.opt.signature, label)
                    # try to load cached mesh
                    mesh = data.load_cache(cache_path)
                    if not mesh:
                        # load mesh from disc
                        mesh = self.get_mesh(i, path, label, resolution)
                        # save loaded mesh in cache
                        data.save_mesh_cache(cache_path, mesh)
                    # append mesh to result
                    self.meshes.append(mesh)
            # save size of dataset
            self.size = len(self.meshes)

    @abc.abstractmethod
    def get_mesh(self, idx: int, path: str, label: str, target_faces_portion: float) -> data.Mesh:
        """
        Abstract function that loads and preprocesses the mesh

        :param idx: index of mesh in paths list
        :param path: load path of the mesh
        :param label: optional label of the mesh
        :param target_faces_portion: percentage of target faces in relation to original number of faces (for remeshing)
        :return: preprocessed mesh
        """

        pass

    @abc.abstractmethod
    def __getitem__(self, index: int) -> Dict:
        """
        Abstract function to apply the random walks on the mesh

        :param index: index of the mesh
        :return: random walks, padding for transformer, list of indices of the visited vertices, labels, and mesh
        """

        pass

    @staticmethod
    @abc.abstractmethod
    def sanity_check(root: str) -> bool:
        """
        Performs a sanity check on the dataset, checking if the folder structure and data are correct.

        :param root: data root path
        :return: True if sanity check is successful, else False
        """
        pass

    @staticmethod
    def is_mesh_file(filename: str) -> bool:
        """
        Checks if mesh file format is supported by trimesh

        :param filename: filename of the mesh
        :return: True if mesh file format is supported by trimesh
        """

        return any(filename.endswith(extension) for extension in BaseDataset.mesh_file_extensions())

    @staticmethod
    def mesh_file_extensions() -> List[str]:
        """
        Returns the supported mesh formats by trimesh

        :return: list of supported mesh formats by trimesh
        """
        return trimesh.exchange.load.mesh_formats()

    def augment(self, mesh: data.Mesh, augmentation: List[Callable]) -> None:
        """
        Applies all augmentations on the mesh

        :param mesh: mesh object
        :param augmentation: list of augmentation functions
        """

        for aug in augmentation:
            aug(mesh=mesh, opt=self.opt)

    def init_walk(self, mesh: data.Mesh) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        """
        Performs the data augmentation of the mesh and initializes the placeholders for the random walks,
        the padding for the transformer and the list of visited vertices of the walks.

        :param mesh: mesh object
        :return: placeholders for the random walks, padding, visited vertices, walk length and (real) sequence length
        """

        # ensure walk is not longer than the amount of vertices
        sequence_length = walk_length = min(mesh.vertices.shape[0], self.sequence_length)
        # relative position mode decreases sequence length by 1
        if self.opt.position_mode == 'relative':
            sequence_length -= 1
        # init placeholder for walks, padding and permutation arrays
        walks = np.zeros((self.walks, sequence_length, models.get_feature_dim(self.opt)), dtype=np.float32)
        pads = np.ones((self.walks, sequence_length), dtype=np.bool_)
        permutations = -1 * np.ones((self.walks, walk_length), dtype=np.int64)
        # apply augmentation to mesh
        self.augment(mesh, self.test_augmentation if self.phase != 'train' else self.train_augmentation)
        return walks, pads, permutations, walk_length, sequence_length

    def get_augmentations(self, prefix: str = 'aug_') -> List[Callable]:
        """
        Check for all data augmentations if a probability is given in the input arguments

        :return: data augmentations with a probability > 0 in the input arguments
        """

        data_augmentations = [] if not self.opt.no_rotation else [data.aug_random_rotate]
        for attribute in [aug_name for aug_name in dir(data) if aug_name.startswith(prefix)]:
            aug_func = getattr(data, attribute)
            if callable(aug_func):
                func_name = attribute[len(prefix):]
                if hasattr(self.opt, func_name):
                    prob = getattr(self.opt, func_name)
                    if prob and prob > 0:
                        data_augmentations.append(aug_func)
        return data_augmentations

    @staticmethod
    def permute(sequence: Union[np.ndarray, torch.Tensor], permutation: np.ndarray, position_mode: str) -> torch.Tensor:
        """
        Permutes sequence and applies the position mode.
        Position mode relative uses the difference between each two consecutive vertices.

        :param sequence: Sequence of data
        :param permutation: Permutation list
        :param position_mode: relative or absolute position mode
        :return: PyTorch Tensor of permuted data
        """
        sequence = sequence[permutation]
        if type(sequence) == np.ndarray:
            sequence = torch.from_numpy(sequence)
        if position_mode == 'relative':
            return torch.diff(sequence, dim=0)
        return sequence

    @staticmethod
    def permute_data_sequence(mesh: data.Mesh, permutation: np.ndarray, position_mode: str) -> torch.Tensor:
        """
        Permutes the vertices and optional spectral descriptor and apply the position mode.
        Position mode relative uses the difference between each two consecutive vertices.

        :param mesh: mesh object
        :param permutation: indices of visited vertices by the random walk
        :param position_mode: relative or absolute position mode
        :return: PyTorch Tensor of permuted data
        """

        # permute vertices positions
        position = BaseDataset.permute(mesh.vertices, permutation, position_mode)
        if mesh.signature is not None:
            laplace = BaseDataset.permute(mesh.signature, permutation, position_mode)
            # concat vertices and spectral descriptor
            return torch.cat((position, laplace), dim=1)
        return position
