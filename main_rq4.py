from sklearn.semi_supervised import LabelSpreading, LabelPropagation
import datasets as ds
import numpy as np
import adversarial
from sklearn.metrics import pairwise
from label_propagation_cached import LabelPropagationCached
from sklearn.exceptions import ConvergenceWarning
import warnings
import time
from config import Config
from greedy_adv_attack import (
    multiple_flips_greedy,
    perturb_y_classification,
    multiple_flips_random,
)


warnings.filterwarnings("ignore", category=ConvergenceWarning)
random_seed = 2020


config = Config(path="config.yaml")


def random_selection_defence(
    x_train_shuffled, y_train_shuffled, labels_unlabelled, n_labelled
):
    lp = LabelPropagationCached(gamma=5, max_iter=10)

    n_flips = int(0.2 * n_labelled)
    max_def = n_flips

    print(f"[n_d][n_flip][acc]")
    acc_u_rbf = np.empty((n_flips,))
    for n_d in range(1, max_def + 1):
        idx_to_add = adversarial.reduce_influence_random(
            x_train_shuffled,
            labels_unlabelled,
            2020,
            n_d,
        )
        labels_unlabelled[idx_to_add.astype(int)] = y_train_shuffled[
            idx_to_add.astype(int)
        ]
        for i in range(1, n_flips + 1):
            acc_u_rbf[i - 1] = adversarial.multiple_flips_accuracy_sklearn(
                x_train_shuffled,
                labels_unlabelled,
                lp,
                x_train_shuffled[n_labelled:],
                y_train_shuffled[n_labelled:],
                i,
            )
            print(f"[{n_d}][{i}][{acc_u_rbf[i - 1]}]")

        np.savetxt(f"defense_{n_d}_{n_flips}_trans_rand", acc_u_rbf)


def ssim_based_def(x_train_shuffled, y_train_shuffled, labels_unlabelled, n_labelled):
    lp = LabelPropagationCached(gamma=5, max_iter=10)

    n_flips = int(0.2 * n_labelled)
    max_def = n_flips

    print(f"[n_d][n_flip][acc]")
    acc_u_rbf = np.empty((n_flips, n_flips))
    idx_to_add = adversarial.reduce_influence_ssim(
        x_train_shuffled,
        labels_unlabelled,
        lp,
        n_flips,
    )
    for i in range(1, n_flips + 1):
        for n_def in range(1, n_flips + 1):
            labels_unlabelled[idx_to_add[:n_def].astype(int)] = y_train_shuffled[
                idx_to_add[:n_def].astype(int)
            ]
            acc_u_rbf[i - 1][n_def - 1] = adversarial.multiple_flips_accuracy_sklearn(
                x_train_shuffled,
                labels_unlabelled,
                lp,
                x_train_shuffled[n_labelled:],
                y_train_shuffled[n_labelled:],
                i,
            )
            print(f"[{n_def}][{i}][{acc_u_rbf[i - 1][n_def - 1]}]")

    np.savetxt(f"defense_26_26_trans_ssim_mnist17_full_matrix", acc_u_rbf)


def main():
    x_train, x_test, y_train, y_test = ds.mnist17_prep()
    dataset = "mnist17"
    p_labelled = 0.01
    n_labelled = int(y_train.shape[0] * p_labelled)
    randomState = np.random.RandomState(random_seed)
    (
        y_train_shuffled,
        x_train_shuffled,
        labels_unlabelled,
    ) = ds.unlabel_shuffle_training_set(y_train, x_train, n_labelled, randomState)

    ssim_based_def(x_train_shuffled, y_train_shuffled, labels_unlabelled, n_labelled)


if __name__ == "__main__":
    main()
