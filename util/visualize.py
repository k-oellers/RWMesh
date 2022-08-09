from matplotlib import cm
import numpy as np
import copy
import torch
import trimesh
import data
from typing import Union, List, Type
from matplotlib.colors import Normalize


def get_color_map(color_map: str) -> Union[List, np.ndarray]:
    """
    return the matplotlib color map for the given name

    :param color_map: name of the color map
    :return: list or numpy array with the color map
    """

    c_map = cm.get_cmap(color_map)
    if not c_map:
        raise Exception(f'unknown color map {color_map}')
    return c_map.colors


def compute_attention_colors(attn_weights_mat: torch.Tensor, perm: torch.Tensor, n_vertices: int,
                             c_map: Union[List, np.ndarray], normalizer: Type[Normalize]) -> np.ndarray:
    """
    Calculates the attention colors for the attention weight matrices.

    :param attn_weights_mat: the Attention weight matrix per walk and per layer, which are specified in
    shape (walk, layer, walk length + 1, walk length + 1).
    :param perm: indices of the selected vertices in the sequence
    :param n_vertices: amount of mesh vertices
    :param c_map: matplotlib color map
    :param normalizer: matplotlib normalizer function
    :return: colors for every vertex
    """

    # logarithmic normalization needs values > 0
    attention = 1e-15 * torch.ones((n_vertices,), device=attn_weights_mat.device)
    for i in range(attn_weights_mat.shape[0]):
        # compute rollout attention
        att = compute_joint_attention(attn_weights_mat[i])
        # get attention weights for CLS token in the last layer
        attention[perm[i]] = att[-1, 0, 1:]
    # torch to numpy
    attention = attention.detach().cpu().numpy()
    # init normalizer
    normalizer = normalizer(np.min(attention), np.max(attention))
    # normalize from 0 to 255
    a = (normalizer(attention) * 255).astype(np.int_)
    # normalize color map from 0 to 255
    color_map = (np.array(copy.copy(c_map)) * 255).astype(np.int_)
    return color_map[a]


def compute_joint_attention(att_mat: torch.Tensor, add_residual: bool = True) -> torch.Tensor:
    """
    Computes the attention rollout.

    Original implementation by samiraabnar (https://github.com/samiraabnar/attention_flow), slightly modified to work
    with torch tensors.

    :param att_mat: attention tensor
    :param add_residual: use residual connection
    :return: attention rollout
    """

    if add_residual:
        residual_att = torch.eye(att_mat.shape[1], device=att_mat.device)[None, ...]
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1)[..., None]
    else:
        aug_att_mat = att_mat

    joint_attentions = torch.zeros(aug_att_mat.shape, device=att_mat.device)

    layers = joint_attentions.shape[0]
    joint_attentions[0] = aug_att_mat[0]
    for i in range(1, layers):
        joint_attentions[i] = aug_att_mat[i].matmul(joint_attentions[i - 1])

    return joint_attentions


def save_obj(mesh: data.Mesh, save_path: str) -> None:
    """
    Saves the mesh object as .obj file

    :param mesh: mesh object
    :param save_path: path to save the model (without .obj extension)
    """

    save_mesh = trimesh.Trimesh(faces=mesh.faces, vertices=mesh.vertices)
    trimesh.exchange.export.export_mesh(save_mesh, f'{save_path}.obj', file_type='obj')


def write_mesh_colors(mesh: data.Mesh, path: str, colors: np.ndarray) -> None:
    """
    Exports the mesh object with given vertex colors as .ply file under the given path

    :param mesh: mesh object
    :param path: save path of the .ply file
    :param colors: numpy array for vertex colors with shape (# vertices, 3) for the rgb values from 0 to 255.
    """
    has_colors = colors is not None
    start_line_elements = ["ply", "format ascii 1.0", f"element vertex {len(mesh.vertices)}",
                           *[f"property float {x}" for x in ['x', 'y', 'z']],
                           f"element edge {len(mesh.edges) - 1}", *[f"property int vertex{x}" for x in [1, 2]],
                           "end_header", ""]
    if has_colors:
        # insert color properties
        start_line_elements[6:6] = [f"property uchar {clr}" for clr in ["red", "green", "blue"]]
    start_line = "\n".join(start_line_elements)
    with open(f"{path}.ply", "w") as f:
        f.write(start_line)
        for i in range(len(mesh.vertices)):
            vertex_entry = [str(v.item()) for v in mesh.vertices[i]]
            if has_colors:
                vertex_entry += [str(x) for x in colors[i]]

            f.write(f"{' '.join(vertex_entry)}\n")

        for i in range(len(mesh.edges)):
            f.write(f"{mesh.edges[i, 0]} {mesh.edges[i, 1]}\n")
