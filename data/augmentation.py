import numpy as np
from scipy.spatial.transform import Rotation as R
import random
import torch
import data
import argparse
from typing import List, Tuple, Optional, Union


def deg_mask(prob: float, deg: np.ndarray, threshold=0.7) -> torch.Tensor:
    """
    Creates a mask according to the centrality of a given degree matrix of a mesh

    :param prob: probability for belonging to the mask
    :param deg: degree matrix of mesh
    :param threshold:
    :return: centrality mask
    """

    s_col = np.log(deg)
    weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())
    edge_weights = weights / weights.mean() * prob
    edge_weights = np.where(edge_weights < threshold, edge_weights, np.ones_like(edge_weights) * threshold)
    mask = torch.bernoulli(1. - torch.from_numpy(edge_weights)).to(torch.bool).numpy()
    return mask


def rand_mask(prob: float, seq: np.ndarray) -> torch.Tensor:
    """
    Creates a random mask

    :param prob: probability for belonging to the mask
    :param seq: data used for mask
    :return: random mask
    """
    # get random distribution for all edges
    rand = torch.rand(seq.shape[0])
    # mask edges with probability
    mask = torch.where(rand > prob, True, False)
    return mask


def apply_edge_mask(mesh: data.Mesh, mask: torch.Tensor) -> None:
    """
    Masks the edges and optional soft edge labels of a given mesh and recalculated the adjacency matrix

    :param mesh: mesh object
    :param mask: PyTorch tensor mask
    """
    mesh.update_edges(mesh.edges[mask])


def aug_random_rotate(mesh: data.Mesh, opt: argparse.Namespace) -> None:
    """
    Randomly rotates the vertices of a given mesh

    :param mesh: mesh object
    :param opt: input arguments
    """

    rotation_pipeline = R.from_euler('zyx', [random.randint(0, 360) for _ in range(3)], degrees=True)
    for i in range(len(mesh.vertices)):
        mesh.vertices[i] = rotation_pipeline.apply(mesh.vertices[i])


def noise_weight(mesh: data.Mesh, std: float) -> np.ndarray:
    """
    Creates a normal distribution for every vertex of a given mesh

    :param mesh: mesh object
    :param std: standard deviation
    :return: normal distribution for every vertex of a given mesh
    """

    return np.array(torch.empty(mesh.vertices.shape).normal_(0, std))


def aug_add_noise(mesh: data.Mesh, opt: argparse.Namespace) -> None:
    """
    Adds random noise to the vertices of a given mesh

    :param mesh: mesh object
    :param opt: input arguments
    """

    # create random mask
    mask = rand_mask(opt.noise, mesh.vertices)
    # create normal distribution
    gaussian_noise = noise_weight(mesh, opt.noise_weight)
    # apply noise to vertices
    mesh.vertices[mask] += gaussian_noise[np.where(mask)]


def aug_add_noise_deg(mesh: data.Mesh, opt: argparse.Namespace) -> None:
    """
    Adds noise according to centrality to the vertices of a given mesh

    :param mesh: mesh object
    :param opt: input arguments
    """

    # create centrality mask
    mask = deg_mask(opt.noise_deg, mesh.degree)
    # create normal distribution
    gaussian_noise = noise_weight(mesh, opt.noise_weight)
    # apply noise to vertices
    mesh.vertices[np.where(mask == False)] += gaussian_noise[np.where(mask == False)]


def aug_edge_drop(mesh: data.Mesh, opt: argparse.Namespace) -> None:
    """
    Randomly drops edges of a given mesh

    :param mesh: mesh object
    :param opt: input arguments
    """

    apply_edge_mask(mesh, rand_mask(opt.edge_drop, mesh.edges))


def aug_edge_drop_deg(mesh: data.Mesh, opt: argparse.Namespace) -> None:
    """
    Drops edges according to centrality to the vertices of a given mesh

    :param mesh: mesh object
    :param opt: input arguments
    """

    apply_edge_mask(mesh, deg_mask(opt.edge_drop_deg, mesh.degree[mesh.edges[:, 1]]))


def aug_edge_add(mesh: data.Mesh, opt: argparse.Namespace) -> None:
    """
    Randomly adds edges of a given mesh

    :param mesh: mesh object
    :param opt: input arguments
    """

    # get all possible new edge connections
    possible_edges = [[i, j] for i in range(mesh.vertices.shape[0]) for j in range(i)]
    # randomly sample n edges (number of edges * edge_add)
    added_edges = np.array(random.sample(possible_edges, int(mesh.edges.shape[0] * opt.edge_add)))
    # update mesh
    mesh.update_edges(np.concatenate((mesh.edges, added_edges), axis=0))


def get_random_walk(adj: np.ndarray, length: int, start: Optional[int] = None,
                    path: Optional[Union[List[int], np.ndarray]] = None) -> Tuple[Union[List[int], np.ndarray], bool]:
    if path is None:
        path = []

    if length - len(path) == 0:
        return path, True

    if start is None:
        start = np.random.choice([idx for idx, x in enumerate(adj) if len(x) > 0])

    if len(adj[start]) == 0:
        return path, False

    else:
        success = False
        while not success:
            if len(adj[start]) == 0:
                not_visited_vertices = [idx for idx, x in enumerate(adj) if len(x) > 0 and idx not in path]
                if len(not_visited_vertices) == 0:
                    return path, False

                path, success = get_random_walk(adj, length, np.random.choice(not_visited_vertices), path)
            else:
                new_elm_id = np.random.choice(np.arange(len(adj[start])))
                new_elm = adj[start][new_elm_id]
                del adj[start][new_elm_id]
                if new_elm not in path:
                    path.append(new_elm)
                    path, success = get_random_walk(adj, length, new_elm, path)

        return path, success
