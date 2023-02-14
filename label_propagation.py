from sklearn.metrics import pairwise
import numpy as np
import os.path
from os import path


class LabelPropagation:
    def __init__(self, x, y, n_unlabeled, gamma=2, iteration=10):
        self.result_labels_distribution = None
        self.x_train = x
        self.y_train = y
        self.n_unlabeled = n_unlabeled
        self.gamma = gamma
        self.n_iter = iteration
        self.file_name_w = f"weights_{gamma}_{self.x_train.shape[0]}_{n_unlabeled}.npy"
        self.file_name_t = (
            f"transition_matrix_{gamma}_{self.x_train.shape[0]}_{n_unlabeled}.npy"
        )
        self.result_labels = y
        self.fitted = False

    def get_file_name_w(self):
        return self.file_name_w

    def get_model(self):
        """
        Will return the label distribution probabilities
        @return: np.array shape (n_train_samples,n_classes)
        """
        if not self.fitted:
            raise Exception(f"Model not fitted yet !")
        return self.x_train, self.result_labels_distribution

    def compute_graph_matrix(self, x, gamma=2):
        if path.exists(self.file_name_w):
            print(f"Found weights {self.file_name_w}")
            weights = np.load(self.file_name_w)
        else:
            print(f"Not found weights {self.file_name_w}")
            weights = pairwise.rbf_kernel(X=x, gamma=gamma)
            np.save(self.file_name_w, weights)

        normalizer = weights.sum(axis=0)
        T = weights / normalizer[:, np.newaxis]
        normalizer_T = np.sum(T, axis=1)
        T_norm = T / normalizer[:, np.newaxis]
        np.save(self.file_name_t, T_norm)
        T_uu = np.copy(T_norm[self.n_unlabeled :, self.n_unlabeled :])
        T_ul = np.copy(T_norm[self.n_unlabeled :, : self.n_unlabeled])
        return weights, T_norm, T_uu, T_ul

    def fit(self, y_train_poisoned=None):
        weights, T_norm, T_uu, T_ul = self.compute_graph_matrix(
            self.x_train, self.gamma
        )
        Y_all = np.zeros((self.x_train.shape[0], 2))
        for label in [0, 1]:
            Y_all[
                np.copy(self.y_train if y_train_poisoned is None else y_train_poisoned)
                == label,
                np.array([0, 1]) == label,
            ] = 1

        Y_L = np.copy(Y_all[: self.n_unlabeled])
        Y_U = np.copy(Y_all[self.n_unlabeled :])

        n = self.n_iter
        for i in range(n):
            Y_U = np.dot(T_uu, Y_U) + np.dot(T_ul, Y_L)
            normalizer_yu = np.sum(Y_U, axis=1)[:, np.newaxis]
            Y_U /= normalizer_yu
        labels_u = Y_U.argmax(axis=1)
        self.result_labels[self.n_unlabeled :] = labels_u
        self.result_labels_distribution = np.concatenate((Y_L, Y_U), axis=0)
        self.fitted = True
        # print(
        #     f"Model fitted over {self.x_train.shape[0]} samples with {self.n_unlabeled} labels over {n_iter} iterations"
        # )
        return self

    def accuracy_rbf(
        self, x_test, y_test, x_train=None, y_train_label_distribution=None,
    ):
        if not self.fitted:
            raise Exception(f"Model not fitted yet !")

        if x_train is None:
            x_train = self.x_train

        if y_train_label_distribution is None:
            y_train_label_distribution = self.result_labels_distribution

        x_test = np.copy(x_test)
        weights_test = pairwise.rbf_kernel(X=x_test, Y=x_train, gamma=self.gamma)
        y_test_distribution = np.dot(weights_test, y_train_label_distribution)
        y_test_labels = y_test_distribution.argmax(axis=1)
        y_test_thruth = y_test
        n_correct_pred = np.array(np.where(y_test_thruth == y_test_labels)).shape[1]
        score = n_correct_pred / y_test_thruth.shape[0]
        return score
