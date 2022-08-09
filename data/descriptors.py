import scipy.sparse.linalg as sla
import scipy
import potpourri3d as pp3d
import numpy as np
from typing import Tuple, Optional


def calc_laplacian_eig(vertices: np.ndarray, faces: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes k eigenvalues and eigenvectors of discrete Laplace-Beltrami operator for given vertices and faces

    :param vertices: vertices of a mesh
    :param faces: faces of a mesh
    :param k: number of eigenvalues and eigenvectors, must be 0 < k < |number of vertices| - 2
    :return: k eigenvalues and eigenvectors
    """

    assert k > 0, f'{k} must be > 0'
    # use cotan laplacian
    L = pp3d.cotan_laplacian(vertices, faces, denom_eps=1e-10)
    k = min(k, L.shape[0] - 2)
    eps = 1e-8
    # calculate mass matrix
    massvec_np = pp3d.vertex_areas(vertices, faces)
    massvec_np += eps * np.mean(massvec_np)

    # prepare matrices
    L_eigsh = (L + scipy.sparse.identity(L.shape[0]) * eps).tocsc()
    massvec_eigsh = massvec_np
    Mmat = scipy.sparse.diags(massvec_eigsh)
    eigs_sigma = eps

    # calculate eigenvalues and eigenvectors
    eig_values, eig_vectors = sla.eigsh(L_eigsh, k=k + 1, M=Mmat, sigma=eigs_sigma)
    eig_values = np.clip(eig_values, a_min=0., a_max=float('inf'))
    # skip first eigenvector because first eigenvalue of laplacian is always 0
    return eig_values[1:k + 1], eig_vectors[:, 1:k + 1]


def compute_hks(eig_values: np.ndarray, eig_vectors: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """
    Compute the heat kernel signature for all vertices

    :param eig_values: eigenvalues of discrete Laplace-Beltrami operator
    :param eig_vectors: eigenvectors of discrete Laplace-Beltrami operator
    :param scales: times
    :return: heat kernel signature
    """

    # expand batch
    if len(eig_values.shape) == 1:
        expand_batch = True
        eig_values, eig_vectors, scales = [np.expand_dims(x, axis=0) for x in [eig_values, eig_vectors, scales]]
    else:
        expand_batch = False

    power_coefs = np.expand_dims(np.exp(-np.expand_dims(eig_values, axis=1) * np.expand_dims(scales, axis=-1)), 1)
    terms = power_coefs * np.expand_dims((eig_vectors * eig_vectors), 2)  # (B,V,S,K)

    out = np.sum(terms, axis=-1)  # (B,V,S)

    if expand_batch:
        return out.squeeze(0)
    else:
        return out


def compute_hks_autoscale(eig_values: np.ndarray, eig_vectors: np.ndarray,
                          count: Optional[int] = None) -> np.ndarray:
    """
    Compute the heat kernel signature for all vertices with logarithmically distributed times

    :param eig_values: eigenvalues of discrete Laplace-Beltrami operator
    :param eig_vectors: eigenvectors of discrete Laplace-Beltrami operator
    :param count: number of logarithmically distributed times
    :return: heat kernel signature
    """

    if count is None:
        count = eig_values.shape[0]

    scales = np.logspace(-2, 0., num=count, dtype=eig_values.dtype)
    return compute_hks(eig_values, eig_vectors, scales)


def compute_wks(eig_values: np.ndarray, eig_vectors: np.ndarray, dim: Optional[int] = None,
                variance: float = 6.0) -> np.ndarray:
    """
    Compute the wave signature for all vertices

    This signature is based on 'The Wave Kernel Signature: A Quantum Mechanical Approach to Shape Analysis'
    by Mathieu Aubry et al (https://vision.informatik.tu-muenchen.de/_media/spezial/bib/aubry-et-al-4dmod11.pdf)

    :param eig_values: eigenvalues of discrete Laplace-Beltrami operator
    :param eig_vectors: eigenvectors of discrete Laplace-Beltrami operator
    :param dim: dimensionality (energy spectra) of the signature.
    :param variance: variance of the WKS gaussian (wih respect to the difference of the two first eigenvalues).
                   For easy or precision tasks (eg. matching with only isometric deformations) you can take it smaller
    :return: an array of shape (#vertices, dim) containing the heat signatures of every vertex.
         If return_times is True this function returns a tuple (Signature, timesteps).
    """

    if dim is None:
        dim = eig_values.shape[0]

    log_e = np.log(np.maximum(np.abs(eig_values), 1e-6))
    energies = np.linspace(log_e[1], log_e[-1] / 1.02, dim)

    sigma = variance * (energies[1] - energies[0])
    phi2 = np.square(eig_vectors)
    exp = np.exp(-np.square(energies[None] - log_e[:, None])) / (2.0 * sigma * sigma)
    s = np.sum((phi2[:, :, None] * exp[None]), axis=1)
    energy_trace = np.sum(exp, axis=0)
    return s / energy_trace[None]


def calc_signature(eig_values: np.ndarray, eig_vectors: np.ndarray, signature: str) -> np.ndarray:
    """
    Computes the signature according to input parameter

    :param eig_values: eigenvalues of discrete Laplace-Beltrami operator
    :param eig_vectors: eigenvectors of discrete Laplace-Beltrami operator
    :param signature: signature name
    :return: signature
    """
    import options
    return options.signatures[signature](eig_values, eig_vectors)
