import data
import os
import argparse
from typing import List, Dict, Optional, Callable, Tuple


class ClassificationData(data.BaseDataset):
    """
    Classification dataset of meshes. A folder must be created in the root folder for each class in the dataset.
    Each of these folders must have a train folder for training meshes and a test folder for test meshes.
    The labels are automatically generated for each mesh.
    """

    def __init__(self, opt: argparse.Namespace, phase: str, filter_function: Optional[Callable] = None,
                 walks: Optional[int] = None, mesh_portion: float = 1) -> None:
        super().__init__(opt, phase, walks)
        directory = os.path.normpath(os.path.join(opt.dataroot))
        self.classes, self.class_to_idx = self.find_classes(directory)
        paths = data.shuffle_paths(self.make_dataset_by_class(directory, phase, filter_function), mesh_portion)
        self.n_classes = len(self.classes)
        super().get_meshes(paths)

    def get_mesh(self, idx: int, path: str, label: str, target_faces_portion: float) -> data.Mesh:
        # load mesh
        return data.mesh.load_mesh(path, self.opt, self.class_to_idx[label], target_faces_portion)

    def __getitem__(self, index: int) -> Dict:
        mesh: data.Mesh = self.meshes[index]
        walks, pads, perm, walk_length, _ = self.init_walk(mesh)
        # label per mesh
        labels = mesh.labels
        # for every walk...
        for i in range(self.walks):
            self.random_walk(mesh, walk_length, walks, perm, pads, i)
        # dictionary to list
        return data.locals_to_dict(locals())

    @staticmethod
    def sanity_check(root: str) -> bool:
        """
        Checks if the

        :param root:
        :return:
        """
        folders = ['train', 'test']

        for cls in sorted(os.listdir(root)):
            cls_path = os.path.join(root, cls)
            if not os.path.isdir(cls_path):
                continue

            for phase in folders:
                d = os.path.join(root, cls, phase)
                if not os.path.isdir(d):
                    return False

                for dir_root, _, names in sorted(os.walk(d)):
                    if not any([data.BaseDataset.is_mesh_file(name) for name in names]):
                        return False
        return True

    @staticmethod
    def find_classes(folder: str) -> Tuple[List[str], Dict[str, int]]:
        classes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    @staticmethod
    def make_dataset_by_class(folder: str, phase: str, filter_function=None) -> List[Dict]:
        meshes = []
        folder = os.path.expanduser(folder)
        for target in sorted(os.listdir(folder)):
            d = os.path.join(folder, target)
            if not os.path.isdir(d):
                continue
            for root, _, names in sorted(os.walk(d)):
                names = filter(filter_function, names) if filter_function else names
                for name in names:
                    if data.BaseDataset.is_mesh_file(name) and root.count(phase) == 1:
                        meshes.append({'path': os.path.normpath(os.path.join(root, name)), 'label': target})
        return meshes
