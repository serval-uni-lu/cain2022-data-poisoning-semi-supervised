from sklearn.semi_supervised import LabelPropagation
import numpy as np
from os import path
from sklearn.metrics import pairwise
import time


class LabelPropagationCached(LabelPropagation):
    def __init__(
        self,
        kernel="rbf",
        gamma=20,
        n_neighbors=7,
        max_iter=1000,
        tol=1e-3,
        n_jobs=None,
    ):
        self.affinity_mtx = None
        self.cache_vars = dict()
        super().__init__(
            kernel=self.custom_graph,
            gamma=gamma,
            n_neighbors=n_neighbors,
            max_iter=max_iter,
            tol=tol,
            n_jobs=n_jobs,
        )
        

    def _build_graph(self):
        file_name = (
            f"transition_matrix_{self.gamma}_{self.X_.shape[0]}_{self.X_.shape[0]}"
        )
        if file_name in self.cache_vars:
            return self.cache_vars.get(file_name)
        else:
            affinity_matrix = self._get_kernel(self.X_)
            normalizer = affinity_matrix.sum(axis=0)
            affinity_matrix /= normalizer[:, np.newaxis]
            self.cache_vars[file_name] = affinity_matrix
            self.affinity_mtx = affinity_matrix
            return affinity_matrix

    def custom_graph(self, X, y):
        if y is None:
            file_name_weights = f"weights_{self.gamma}_{X.shape[0]}_{X.shape[0]}"
            if file_name_weights in self.cache_vars:
                weights = self.cache_vars[file_name_weights]
            else:
                print(f"computing rbf kernel {X.shape[0]} , {self.gamma} ")
                weights = pairwise.rbf_kernel(X, X, gamma=self.gamma)
                print("saving kernel")
                self.cache_vars[file_name_weights] = weights
        else:
            file_name_weights = f"weights_{self.gamma}_{X.shape[0]}_{y.shape[0]}"
            if file_name_weights in self.cache_vars:
                weights = self.cache_vars[file_name_weights]
            else:
                weights = pairwise.rbf_kernel(X, y, gamma=self.gamma)
                self.cache_vars[file_name_weights] = weights
        return weights

    def clear_cache(self):
        del self.cache_vars
