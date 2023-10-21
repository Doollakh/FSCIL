"""Algorithms for spectral clustering"""

from numbers import Integral, Real
import warnings

import numpy as np

from scipy.linalg import LinAlgError, qr, svd
from scipy.sparse import csc_matrix
from scipy.spatial import cKDTree

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils import check_random_state, as_float_array
from sklearn.metrics.pairwise import pairwise_kernels, KERNEL_PARAMS
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn.manifold import spectral_embedding
from sklearn.cluster._kmeans import k_means



def cluster_qr(vectors):

    k = vectors.shape[1]
    _, _, piv = qr(vectors.T, pivoting=True)
    ut, _, v = svd(vectors[piv[:k], :].T)
    vectors = abs(np.dot(vectors, np.dot(ut, v.conj())))
    return vectors.argmax(axis=1)


def discretize(
    vectors, *, copy=True, max_svd_restarts=30, n_iter_max=20, random_state=None
):

    random_state = check_random_state(random_state)

    vectors = as_float_array(vectors, copy=copy)

    eps = np.finfo(float).eps
    n_samples, n_components = vectors.shape
    norm_ones = np.sqrt(n_samples)
    for i in range(vectors.shape[1]):
        vectors[:, i] = (vectors[:, i] / np.linalg.norm(vectors[:, i])) * norm_ones
        if vectors[0, i] != 0:
            vectors[:, i] = -1 * vectors[:, i] * np.sign(vectors[0, i])

    vectors = vectors / np.sqrt((vectors**2).sum(axis=1))[:, np.newaxis]

    svd_restarts = 0
    has_converged = False

    while (svd_restarts < max_svd_restarts) and not has_converged:

        rotation = np.zeros((n_components, n_components))
        rotation[:, 0] = vectors[random_state.randint(n_samples), :].T

        c = np.zeros(n_samples)
        for j in range(1, n_components):
            c += np.abs(np.dot(vectors, rotation[:, j - 1]))
            rotation[:, j] = vectors[c.argmin(), :].T

        last_objective_value = 0.0
        n_iter = 0

        while not has_converged:
            n_iter += 1

            t_discrete = np.dot(vectors, rotation)

            labels = t_discrete.argmax(axis=1)
            vectors_discrete = csc_matrix(
                (np.ones(len(labels)), (np.arange(0, n_samples), labels)),
                shape=(n_samples, n_components),
            )

            t_svd = vectors_discrete.T * vectors

            try:
                U, S, Vh = np.linalg.svd(t_svd)
            except LinAlgError:
                svd_restarts += 1
                print("SVD did not converge, randomizing and trying again")
                break

            ncut_value = 2.0 * (n_samples - S.sum())
            if (abs(ncut_value - last_objective_value) < eps) or (n_iter > n_iter_max):
                has_converged = True
            else:
                # otherwise calculate rotation and continue
                last_objective_value = ncut_value
                rotation = np.dot(Vh.T, U.T)

    if not has_converged:
        raise LinAlgError("SVD did not converge")
    return labels


def spectral_clustering(
    affinity,
    *,
    n_clusters=8,
    n_components=None,
    eigen_solver=None,
    random_state=None,
    n_init=10,
    eigen_tol="auto",
    assign_labels="kmeans",
    verbose=False,
):

    if assign_labels not in ("kmeans", "discretize", "cluster_qr"):
        raise ValueError(
            "The 'assign_labels' parameter should be "
            "'kmeans' or 'discretize', or 'cluster_qr', "
            f"but {assign_labels!r} was given"
        )
    if isinstance(affinity, np.matrix):
        raise TypeError(
            "spectral_clustering does not support passing in affinity as an "
            "np.matrix. Please convert to a numpy array with np.asarray. For "
            "more information see: "
            "https://numpy.org/doc/stable/reference/generated/numpy.matrix.html",  # noqa
        )

    random_state = check_random_state(random_state)
    n_components = n_clusters if n_components is None else n_components

    # We now obtain the real valued solution matrix to the
    # relaxed Ncut problem, solving the eigenvalue problem
    # L_sym x = lambda x  and recovering u = D^-1/2 x.
    # The first eigenvector is constant only for fully connected graphs
    # and should be kept for spectral clustering (drop_first = False)
    # See spectral_embedding documentation.
    maps = spectral_embedding(
        affinity,
        n_components=n_components,
        eigen_solver=eigen_solver,
        random_state=random_state,
        eigen_tol=eigen_tol,
        drop_first=False,
    )
    if verbose:
        print(f"Computing label assignment using {assign_labels}")

    if assign_labels == "kmeans":

        import pdb
        # pdb.set_trace()

        centroids, labels, _ = k_means(
            maps, n_clusters, random_state=random_state, n_init=n_init, verbose=verbose
        )

        nearest_points = [];
        for centroid in centroids:
            distances = np.linalg.norm(maps - centroid, axis=1);
            nearest_index = np.argmin(distances);
            # nearest_point = maps[nearest_index];
            nearest_points.append(nearest_index);

        print(nearest_points)



    elif assign_labels == "cluster_qr":
        labels = cluster_qr(maps)
    else:
        labels = discretize(maps, random_state=random_state)

    return [labels,nearest_points]


class SpectralClusteringMod(ClusterMixin, BaseEstimator):
    """ np.exp(-gamma * d(X,X) ** 2) """

    _parameter_constraints: dict = {
        "n_clusters": [Interval(Integral, 1, None, closed="left")],
        "eigen_solver": [StrOptions({"arpack", "lobpcg", "amg"}), None],
        "n_components": [Interval(Integral, 1, None, closed="left"), None],
        "random_state": ["random_state"],
        "n_init": [Interval(Integral, 1, None, closed="left")],
        "gamma": [Interval(Real, 0, None, closed="left")],
        "affinity": [
            callable,
            StrOptions(
                set(KERNEL_PARAMS)
                | {"nearest_neighbors", "precomputed", "precomputed_nearest_neighbors"}
            ),
        ],
        "n_neighbors": [Interval(Integral, 1, None, closed="left")],
        "eigen_tol": [
            Interval(Real, 0.0, None, closed="left"),
            StrOptions({"auto"}),
        ],
        "assign_labels": [StrOptions({"kmeans", "discretize", "cluster_qr"})],
        "degree": [Interval(Integral, 0, None, closed="left")],
        "coef0": [Interval(Real, None, None, closed="neither")],
        "kernel_params": [dict, None],
        "n_jobs": [Integral, None],
        "verbose": ["verbose"],
    }

    def __init__(
        self,
        n_clusters=8,
        *,
        eigen_solver=None,
        n_components=None,
        random_state=None,
        n_init=10,
        gamma=1.0,
        affinity="rbf",
        n_neighbors=10,
        eigen_tol="auto",
        assign_labels="kmeans",
        degree=3,
        coef0=1,
        kernel_params=None,
        n_jobs=None,
        verbose=False,
    ):
        self.n_clusters = n_clusters
        self.eigen_solver = eigen_solver
        self.n_components = n_components
        self.random_state = random_state
        self.n_init = n_init
        self.gamma = gamma
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.eigen_tol = eigen_tol
        self.assign_labels = assign_labels
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y=None):
        self._validate_params()

        X = self._validate_data(
            X,
            accept_sparse=["csr", "csc", "coo"],
            dtype=np.float64,
            ensure_min_samples=2,
        )
        allow_squared = self.affinity in [
            "precomputed",
            "precomputed_nearest_neighbors",
        ]
        if X.shape[0] == X.shape[1] and not allow_squared:
            warnings.warn(
                "The spectral clustering API has changed. ``fit``"
                "now constructs an affinity matrix from data. To use"
                " a custom affinity matrix, "
                "set ``affinity=precomputed``."
            )

        if self.affinity == "nearest_neighbors":
            connectivity = kneighbors_graph(
                X, n_neighbors=self.n_neighbors, include_self=True, n_jobs=self.n_jobs
            )
            self.affinity_matrix_ = 0.5 * (connectivity + connectivity.T)
        elif self.affinity == "precomputed_nearest_neighbors":
            estimator = NearestNeighbors(
                n_neighbors=self.n_neighbors, n_jobs=self.n_jobs, metric="precomputed"
            ).fit(X)
            connectivity = estimator.kneighbors_graph(X=X, mode="connectivity")
            self.affinity_matrix_ = 0.5 * (connectivity + connectivity.T)
        elif self.affinity == "precomputed":
            self.affinity_matrix_ = X
        else:
            params = self.kernel_params
            if params is None:
                params = {}
            if not callable(self.affinity):
                params["gamma"] = self.gamma
                params["degree"] = self.degree
                params["coef0"] = self.coef0
            self.affinity_matrix_ = pairwise_kernels(
                X, metric=self.affinity, filter_params=True, **params
            )

        random_state = check_random_state(self.random_state)
        self.labels_ = spectral_clustering(
            self.affinity_matrix_,
            n_clusters=self.n_clusters,
            n_components=self.n_components,
            eigen_solver=self.eigen_solver,
            random_state=random_state,
            n_init=self.n_init,
            eigen_tol=self.eigen_tol,
            assign_labels=self.assign_labels,
            verbose=self.verbose,
        )
        return self

    def fit_predict(self, X, y=None):
                return super().fit_predict(X, y)

    def _more_tags(self):
        return {
            "pairwise": self.affinity
            in ["precomputed", "precomputed_nearest_neighbors"]
        }

def chamfer_distance(point_cloud_A, point_cloud_B):
    # Build k-d trees for fast nearest neighbor search
    tree_A = cKDTree(point_cloud_A)
    tree_B = cKDTree(point_cloud_B)

    # Compute the distances from A to B and B to A
    dist_A_to_B, _ = tree_A.query(point_cloud_B, k=1)
    dist_B_to_A, _ = tree_B.query(point_cloud_A, k=1)

    # Sum the squared distances
    term1 = np.sum(dist_A_to_B ** 2)
    term2 = np.sum(dist_B_to_A ** 2)

    # Return the sum of the two terms
    return term1 + term2


def save_ply(point_cloud, address):
  point_cloud = point_cloud.T
  vertex = np.zeros(point_cloud.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

  for i in range(point_cloud.shape[0]):
      vertex[i] = (point_cloud[i][0], point_cloud[i][1], point_cloud[i][2])

  vertex_element = PlyElement.describe(vertex, 'vertex')

  PlyData([vertex_element], text=True).write(address + '.ply')
  print('saved_ply', address)

# --------------------------- Usage example ----------------------------------

# class_name = 'bowl'
# path = f'/content/{class_name}/train/'

# data = read_data(path) # Read, sample and normalize data

# affinity_matrix = np.zeros((len(data), len(data)))
# dis_matrix = np.zeros((len(data), len(data)))

# for i in range(len(data)):  # Compute the distance matrix to specify sigma
#       for j in range(i,len(data)):
#             dis_matrix[i,j] = chamfer_distance(data[i], data[j])
#             dis_matrix[j,i] = dis_matrix[i,j]

# sigma = int(np.median(dis_matrix[dis_matrix != 0]))
# # print(f'sigma (Median) : {sigma}')
# # print(f'sigma (Mean) : {int(np.mean(dis_matrix[dis_matrix != 0]))}')

# for i in range(len(data)):
#       for j in range(i,len(data)):  # Compute the affinity matrix to feed the spectral clustering algorithm
#             distance = chamfer_distance(data[i], data[j])
#             affinity_matrix[i, j] = np.exp(-distance**2 / (2.0 * sigma**2))
#             affinity_matrix[j,i] = affinity_matrix[i,j]

# clustering1 = SpectralClusteringMod(n_clusters=5, random_state=0,affinity='precomputed').fit(affinity_matrix)

# actual_labels = np.array(clustering1.labels_[1])  # 0 : labels of each sampel, 1 : index of the nearst sample to the centroid
# air_exem = np.array(data)[actual_labels]
# for i,j in enumerate(air_exem):
#   save_ply(j,f'/content/Best/{class_name}_{actual_labels[i]+1}' )

# print(clustering1.labels_) 