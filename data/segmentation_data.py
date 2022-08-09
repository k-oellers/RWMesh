import data
import os
import numpy as np
import argparse
from typing import List, Dict, Optional, Callable, Tuple


class SegmentationData(data.BaseDataset):
    """
    Segmentation dataset for meshes.
    The test sub folder of the root folder contains the test data, the train sub folder contains the training data.
    For each training or test mesh, there must be an .eseg file in the seg sub folder and a .seseg file in the sseg sub
    folder with the same name as the mesh.
    The i-th line in an .eseg file determines the label of the i-th face of the mesh. The i-th line in a .seseg file
    determines the fuzzy label of the i-th face of the mesh, with probabilities specified for each label in the dataset.
    After preprocessing, a classes.txt file is created in the root folder, which specifies per line the existing
    classes in the dataset.
    """

    folders = {'seg': '.eseg', 'sseg': '.seseg'}

    def __init__(self, opt: argparse.Namespace, phase: str, filter_function: Optional[Callable] = None,
                 walks: Optional[int] = None, mesh_portion: float = 1) -> None:
        super().__init__(opt, phase, walks)
        paths = self.make_dataset(os.path.join(opt.dataroot, phase), filter_function)
        paths = data.shuffle_paths(paths, mesh_portion)
        self.seg_paths = self.get_seg_files(paths, os.path.join(self.root, 'seg'),
                                            seg_ext=SegmentationData.folders['seg'])
        self.sseg_paths = self.get_seg_files(paths, os.path.join(self.root, 'sseg'),
                                             seg_ext=SegmentationData.folders['sseg'])
        self.classes, self.offset = self.get_n_segs(os.path.join(self.root, 'classes.txt'), self.seg_paths)
        self.n_classes = len(self.classes)
        super().get_meshes(paths)

    def get_mesh(self, idx: int, path: str, label: str, target_faces_portion: float) -> data.Mesh:
        # load edge labels
        edge_labels = read_seg(self.seg_paths[idx]) - self.offset
        soft_edge_labels = read_sseg(self.sseg_paths[idx])
        # load mesh
        return data.mesh.load_mesh(path, self.opt, edge_labels, target_faces_portion, soft_edge_labels)

    def __getitem__(self, index: int) -> Dict:
        mesh: data.Mesh = self.meshes[index]
        walks, pads, perm, walk_length, sequence_length = self.init_walk(mesh)
        # label per vertex
        labels = -1 * np.ones((self.walks, sequence_length), dtype=np.int64)
        # for every walk...
        for i in range(self.walks):
            real_walk_length = self.random_walk(mesh, walk_length, walks, perm, pads, i)
            # permute labels
            labels[i, :real_walk_length] = mesh.labels[perm[i, 1 if self.opt.position_mode == 'relative' else 0:]]
        # dictionary to list
        return data.locals_to_dict(locals())

    @staticmethod
    def sanity_check(root: str) -> bool:
        folder_files = {**SegmentationData.folders, **{'test': data.BaseDataset.mesh_file_extensions(),
                                                       'train': data.BaseDataset.mesh_file_extensions()}}

        for sub_dir in os.listdir(root):
            path = os.path.join(root, sub_dir)
            if not os.path.isdir(path):
                continue

            if sub_dir not in folder_files.keys():
                return False

            extensions = tuple(folder_files[sub_dir])
            if not any([file.endswith(extensions) for file in os.listdir(path) if
                        os.path.isfile(os.path.join(path, file))]):
                return False
        return True

    @staticmethod
    def get_seg_files(paths: List[Dict], seg_dir: str, seg_ext: str) -> List[str]:
        segs = []
        for path in paths:
            segfile = os.path.join(seg_dir, os.path.splitext(os.path.basename(path['path']))[0] + seg_ext)
            assert (os.path.isfile(segfile))
            segs.append(segfile)
        return segs

    @staticmethod
    def get_n_segs(classes_file: str, seg_files: List[str]) -> Tuple[np.ndarray, int]:
        if not os.path.isfile(classes_file):
            all_segs = np.array([], dtype='float64')
            for seg in seg_files:
                all_segs = np.concatenate((all_segs, read_seg(seg)))
            seg_names = np.unique(all_segs)
            np.savetxt(classes_file, seg_names, fmt='%d')
        classes = np.loadtxt(classes_file)
        offset = classes[0]
        classes = classes - offset
        return classes, offset

    @staticmethod
    def make_dataset(path: str, filter_function: Optional[Callable] = None) -> List[Dict]:
        meshes = []
        for root, _, names in sorted(os.walk(path)):
            names = filter(filter_function, names) if filter_function else names
            for name in names:
                if data.BaseDataset.is_mesh_file(name):
                    meshes.append({'path': os.path.normpath(os.path.join(root, name))})

        return meshes


def read_seg(seg: str) -> np.ndarray:
    with open(seg, 'r') as file:
        seg_labels = np.loadtxt(file, dtype='float64')
        return seg_labels


def read_sseg(sseg_file: str) -> np.ndarray:
    sseg_labels = read_seg(sseg_file)
    sseg_labels = np.array(sseg_labels, dtype=np.float32)
    return sseg_labels
