import numpy as np
from pathlib import Path
import os
from typing import Optional
import data


def load_cache(cache_path: str) -> Optional[data.Mesh]:
    """
    Loads a save mesh from the cache

    :param cache_path: path of the cached mesh
    :return: mesh object
    """

    if os.path.exists(cache_path):
        mesh = np.load(cache_path, allow_pickle=True).item()
        if isinstance(mesh, data.Mesh):
            return mesh
    return None


def save_mesh_cache(path: str, mesh: data.Mesh) -> None:
    """
    Saves a mesh under the given path

    :param path: cache path
    :param mesh: mesh object
    """

    path, filename = os.path.split(path)
    filename = f'{get_filename(filename)}.npy'
    Path(path).mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(path, filename), mesh)


def get_filename(filename: str) -> str:
    """
    Return a filename without its file extension

    :param filename: filename
    :return: filename without its file extension
    """
    filename = os.path.basename(filename)
    return '.'.join(filename.split('.')[:-1]) if '.' in filename else filename


def get_cache_path(cache_root: str, dataset_name: str, path: str, eig_basis: int, remesh_portion: float, phase: str,
                   signature: str, label: Optional[str]) -> str:
    filename = f'{get_filename(path.split(os.sep)[-1])}_{str(remesh_portion)}.npy'
    eig_label = f'eig_{str(eig_basis)}'
    file_path = [phase, filename]
    if label:
        file_path.insert(0, label)
    return os.path.join(cache_root, dataset_name, signature, eig_label, *file_path)
