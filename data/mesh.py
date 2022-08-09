from scipy import sparse as sp
import ntpath
import open3d as o3d
import numpy as np
import torch
import trimesh
import data
import argparse
from typing import Optional, Union, Tuple


class Mesh:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray, path: str, opt: argparse.Namespace,
                 labels: Optional[Union[np.ndarray, int]], soft_edge_labels=Optional[np.ndarray]) -> None:
        """
        Initializes a mesh object, calculating the adjacency matrix, filename, degree matrix and edges

        :param vertices: vertices of the mesh
        :param faces: faces of the mesh
        :param path: path to the mesh file
        :param opt: input arguments
        :param labels: labels
        :param soft_edge_labels: optional soft labels
        """
        self.opt = opt
        # normalize vertices
        self.vertices = self.norm_vertices(vertices)
        self.faces = faces
        # create adjacency matrix from vertices and faces
        self.adj = self.get_vert_connectivity(self.vertices, faces)
        self.filename = self.path_leaf(path).split('.')[0]
        # calculate degree matrix
        self.degree = self.calc_degree(self.adj)
        # calculate signature
        self.signature = self.calc_signature()
        if soft_edge_labels is not None:
            # calc edges, vertex labels and soft vertex labels
            self.edges, self.labels, self.soft_labels = self.calc_edge_properties(labels, soft_edge_labels)
        else:
            # calc edges from adjacency matrix
            self.edges, self.labels, self.soft_labels = self.adj_to_edges(self.adj), labels, None

    def calc_signature(self) -> Optional[np.ndarray]:
        """
        Calculate the signature for a mesh with specific eig_basis

        :return: signature
        """

        if not self.opt.eig_basis:
            return None

        eig_values, eig_vectors = data.calc_laplacian_eig(self.vertices, self.faces, self.opt.eig_basis)
        return data.calc_signature(eig_values, eig_vectors, self.opt.signature)

    @staticmethod
    def remesh(mesh_orig, target_n_faces: int):
        mesh = mesh_orig.simplify_quadric_decimation(target_n_faces)
        mesh = mesh.remove_unreferenced_vertices()
        return mesh

    @staticmethod
    def adj_to_edges(adj: sp.csc_matrix) -> np.ndarray:
        """
        Converts adjacency matrix to edge array

        :param adj:
        :return: edge array
        """

        return np.array(list(set(map(lambda pair: (min(pair), max(pair)), adj.todok().keys()))))

    def update_edges(self, edges) -> None:
        self.edges = edges
        # convert edges to adj matrix
        self.adj = self.edges_to_adj(edges, tuple([self.vertices.shape[0]] * 2))

    @staticmethod
    def edges_to_adj(edges, shape) -> sp.csc_matrix:
        """
        Converts the edges of the mesh to the adjacency matrix

        :return:
        """

        rows, cols = [edges[:, i] for i in range(2)]
        return sp.csc_matrix((np.ones((edges.shape[0],)), (rows, cols)), shape=shape)

    @staticmethod
    def calc_degree(adj: sp.csc_matrix) -> np.ndarray:
        """
        Calculates the degree matrix of a adjacency matrix

        :param adj: adjacency matrix
        :return: degree matrix
        """
        return np.sum(np.asarray(adj.todense()), axis=1)

    @staticmethod
    def norm_vertices(vertices: np.ndarray) -> np.ndarray:
        """
        Normalizes vertices to the bounding box [-1, 1]^3

        :param vertices: vertices of a mesh
        :return: normalizd vertices
        """

        vertices -= np.mean((np.min(vertices, axis=0), np.max(vertices, axis=0)), axis=0)
        vertices /= np.max(vertices)
        return vertices

    @staticmethod
    def get_vert_connectivity(vertices: np.ndarray, faces: np.ndarray) -> sp.csc_matrix:
        """
        Returns a sparse matrix (of size #verts x #verts) adjacency matrix, where each nonzero
        element indicates a neighborhood relation.

        :param vertices: vertices of a mesh
        :param faces: faces of a mesh
        :return: adjacency matrix of the mesh
        """

        def row(a):
            return a.reshape((1, -1))

        vpv = sp.csc_matrix((len(vertices), len(vertices)))

        # for each column in the faces...
        for i in range(3):
            IS = faces[:, i]
            JS = faces[:, (i + 1) % 3]
            data = np.ones(len(IS))
            ij = np.vstack((row(IS.flatten()), row(JS.flatten())))
            mtx = sp.csc_matrix((data, ij), shape=vpv.shape)
            vpv = vpv + mtx + mtx.T
        vpv.data = np.array([1 for _ in vpv.data], dtype=np.float64)
        return vpv

    def calc_edge_properties(self, edge_label: np.ndarray, soft_edge_label: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates edges, vertex labels and soft vertex labels.

        :param edge_label: numpy array of edge labels (from .eseg file)
        :param soft_edge_label: numpy array of soft edge labels (from .seseg file)
        :return: edges, face area of the edges, vertex labels and soft vertex labels
        """

        # list for every vertex
        v_labels = [[] for _ in range(self.vertices.shape[0])]
        # soft labels for vertices with shape (number of vertices, number of classes)
        vertex_soft_labels = np.zeros((self.vertices.shape[0], soft_edge_label.shape[1]))
        edges_count = 0
        edges = []
        rng = range(3)
        # for every face...
        for face_id, face in enumerate(self.faces):
            # get sorted list of edges of face
            faces_edges = [tuple(sorted([face[i], face[(i + 1) % 3]])) for i in rng]
            for edge in faces_edges:
                # edge not yet used
                if edge not in edges:
                    # start and end vertex of edge
                    for vertex in edge:
                        # add soft label for edge to vertex
                        vertex_soft_labels[vertex] += soft_edge_label[edges_count]
                        # add new entry for vertex labels
                        v_labels[vertex].append(edge_label[edges_count])

                    edges_count += 1
                    edges.append(edge)

        # normalize soft labels
        vertex_soft_labels = (vertex_soft_labels.T / np.array([len(v) for v in v_labels])).T
        assert np.max(np.sum(vertex_soft_labels, axis=1)) == 1, 'soft vertex label must sum to 1!'
        vertex_labels = np.zeros((len(v_labels),), dtype=np.int_)
        for index, adj_labels in enumerate(v_labels):
            vertex_labels[index] = np.argmax(np.bincount(np.array(adj_labels, dtype=np.int_)))
        return np.array(edges), vertex_labels, vertex_soft_labels

    @staticmethod
    def path_leaf(path):
        head, tail = ntpath.split(path)
        return tail or ntpath.basename(head)


def load_mesh(path: str, opt: argparse.Namespace, labels: Union[np.ndarray, int], target_faces_portion: float = 1,
              soft_edge_labels: Optional[np.ndarray] = None) -> Mesh:
    """
    Loads and optionally remeshes a mesh given a file

    :param path: path of the mesh file
    :param opt: input arguments
    :param target_faces_portion: percentage of faces to be used (remeshing)
    :param labels: labels of the mesh
    :param soft_edge_labels: optional softlabels of the mesh
    :return:
    """

    # don't process mesh in order to preserve face and vertex order
    mesh_ = trimesh.load_mesh(path, **{'process': False})
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(mesh_.vertices)
    mesh.triangles = o3d.utility.Vector3iVector(mesh_.faces)
    # remesh
    if target_faces_portion != 1:
        mesh = Mesh.remesh(mesh, int(len(mesh.triangles) * target_faces_portion))

    # extract vertices and faces
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    # create mesh obj
    return Mesh(vertices, faces, path, opt, labels, soft_edge_labels)
