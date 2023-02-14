import numpy as np
from sklearn import svm
from skimage.metrics import structural_similarity as ssim
import scipy as sp
from sklearn.ensemble import RandomForestClassifier
from greedy_adv_attack import (
    multiple_flips_greedy,
    perturb_y_classification,
    multiple_flips_random,
)


def poison_sample(original_label, classes):
    """
    Utility function to change (poison) a sample

    Parameters
    ----------
        original_label : int
            The original class number
        classes : array_like
            All possible classes
    """
    label = original_label
    classes = np.delete(classes, np.where(classes == label))
    return np.random.choice(classes, 1)


def multiple_flips_training(
    x_train, y_train_unlabeled, lp_model, x_test, y_test, idxs_to_flip
):
    classes = np.unique(y_train_unlabeled)
    classes = classes[classes != -1]

    y_train_flip = np.copy(y_train_unlabeled)
    n_to_flip = idxs_to_flip.shape[0]
    for index in range(n_to_flip):
        to_flip = idxs_to_flip[index]
        y_train_flip[to_flip] = poison_sample(y_train_flip[to_flip], classes)

    lp_model.fit(x_train, y_train_flip)
    acc = lp_model.score(x_test, y_test)
    return acc, x_train, lp_model.transduction_


def single_flip_training(x_train, y_train_unlabeled, lp_model, x_test, y_test, i_flip):
    y_train_flip = np.copy(y_train_unlabeled)
    classes = np.unique(y_train_flip)
    classes = classes[classes != -1]
    y_train_flip[i_flip] = poison_sample(y_train_unlabeled[i_flip], classes)
    lp_model.fit(x_train, y_train_flip)
    acc = lp_model.score(x_test, y_test)

    return acc, x_train, lp_model.transduction_


def max_influence_labeled(weights, n_labelled=130):
    weights_ul = weights[n_labelled:, :n_labelled]
    s = np.array([sum(i) for i in weights_ul])
    m = np.array([max(i) for i in weights_ul])
    an = (s - m) / s
    im = np.array(np.where((an < 0.5))).reshape(
        -1,
    )
    idx_max = np.argmax(weights_ul[im], axis=1)
    res = np.array([np.count_nonzero(idx_max == i) for i in range(n_labelled)])
    return res


def min_influence_uu(weights, n_labelled=130):
    weights_uu = weights[n_labelled:, n_labelled:]
    s = np.array([sum(i) for i in weights_uu])
    m = np.array([max(i) for i in weights_uu])
    an = (s - m) / s
    im = np.array(np.where((an < 0.5))).reshape(
        -1,
    )
    idx_max = np.argmin(weights_uu[im], axis=1)
    res = np.array([np.count_nonzero(idx_max == i) for i in range(weights_uu.shape[1])])
    return res


def particularity_max_influence_single_flip(load_weights, n_labelled=130):
    weights = load_weights
    weights_ul = weights[n_labelled:, :n_labelled]
    results_weight = np.zeros(
        n_labelled,
    )
    for i in range(n_labelled):
        results_weight[i] = np.array(np.where(weights_ul.argmax(axis=1) == i)).shape[1]

    return results_weight


def best_to_flip(n_flips, weights, n_labelled=130):
    max_influence_per_labelled = max_influence_labeled(weights, n_labelled)

    best_idx = max_influence_per_labelled.argmax()
    search_array = np.copy(max_influence_per_labelled)
    idxs_to_flip = np.empty((n_flips,))
    for i in range(n_flips):
        idxs_to_flip[i] = best_idx
        search_array[best_idx] = 0
        best_idx = search_array.argmax()

    idxs_to_flip = idxs_to_flip.astype(int)
    return idxs_to_flip


def exhaustive_accuracy_sklearn(
    x_train, y_train_unlabeled, lp_model, x_test, y_test, sample_to_test
):
    result = np.empty((sample_to_test.shape[0],))
    for i, sample in enumerate(sample_to_test):
        acc, _, _ = single_flip_training(
            x_train, y_train_unlabeled, lp_model, x_test, y_test, sample
        )
        result[i] = acc
        print(f"{i}:{result[i]}")
    return result


def exhaustive_accuracy_svm(x_train, y_train_unlabeled, lp_model, x_test, y_test):
    result = np.empty((130,))
    for i, label in enumerate(y_train_unlabeled[:130]):
        y_train_flip = np.copy(y_train_unlabeled)
        y_train_flip[i] = 1 if y_train_flip[i] == 0 else 0
        clf = svm.SVC()
        acc_lp, x_train, y_train = single_flip_training(
            x_train, y_train_unlabeled, lp_model, x_test, y_test, i
        )
        clf.fit(x_train, y_train)
        acc_svm = clf.score(x_test, y_test)
        result[i] = acc_svm
        print(f"{i}:svm {acc_svm} , lp {acc_lp}")

    return result


def multiple_flips_accuracy_rfc(
    x_train, y_train_unlabeled, lp_model, x_test, y_test, n_flips, weights=None
):
    if weights is None:
        weights = lp_model.custom_graph(x_train, y=None)
    if sp.sparse.issparse(y_train_unlabeled):
        n_labelled = (y_train_unlabeled != -1).data.shape[0]
    else:
        n_labelled = np.array(np.where(y_train_unlabeled != -1)).shape[1]

    idx_to_flip = best_to_flip(n_flips, weights, n_labelled)
    acc, x_train, y_train = multiple_flips_training(
        x_train, y_train_unlabeled, lp_model, x_test, y_test, idx_to_flip
    )
    clf = RandomForestClassifier(random_state=2020)
    # clf = svm.SVC(verbose=True,max_iter=1000,gamma='auto')
    clf.fit(x_train, y_train)
    acc_svm = clf.score(x_test, y_test)
    return acc_svm


def multiple_flips_accuracy_svm(
    x_train, y_train_unlabeled, lp_model, x_test, y_test, n_flips, weights=None
):
    if weights is None:
        weights = lp_model.custom_graph(x_train, y=None)
    if sp.sparse.issparse(y_train_unlabeled):
        n_labelled = (y_train_unlabeled != -1).data.shape[0]
    else:
        n_labelled = np.array(np.where(y_train_unlabeled != -1)).shape[1]

    idx_to_flip = best_to_flip(n_flips, weights, n_labelled)
    acc, x_train, y_train = multiple_flips_training(
        x_train, y_train_unlabeled, lp_model, x_test, y_test, idx_to_flip
    )
    clf = svm.SVC(verbose=True, max_iter=1000, gamma="auto")
    clf.fit(x_train, y_train)
    acc_svm = clf.score(x_test, y_test)
    return acc_svm


def multiple_flips_accuracy_sklearn_defence_rfc(
    x_train,
    y_train_unlabeled,
    y_train,
    lp_model,
    x_test,
    y_test,
    n_flips,
    gamma,
    weights=None,
):
    if weights is None:
        weights = lp_model.custom_graph(x_train, y=None)

    n_labelled = np.array(np.where(y_train_unlabeled != -1)).shape[1]
    idx_to_flip_i = best_to_flip(n_flips, weights, n_labelled)
    perturbation = perturb_y_classification(
        n_labelled,
        x_train,
        y_train,
        n_flips,
        gamma=gamma,
    )

    n_perturb = np.array(np.where(perturbation == -1)).shape[1]
    idxs_to_flip_p = np.array(np.where(perturbation == -1)).reshape((n_perturb,))

    defence = idx_to_flip_i[int(1 / 3 * n_flips) :]
    acc_t_I, _, y_transduction = multiple_flips_training(
        x_train,
        y_train_unlabeled,
        lp_model,
        x_train[n_labelled:],
        y_train[n_labelled:],
        defence,
    )

    clf = RandomForestClassifier(random_state=2020)
    # clf = svm.SVC(verbose=True,max_iter=1000,gamma='auto')
    clf.fit(x_train, y_transduction)
    acc_i_I = clf.score(x_test, y_test)

    return acc_t_I, acc_i_I


def multiple_flips_accuracy_rfc_defense(
    x_train, y_train_unlabeled, lp_model, x_test, y_test, n_flips, weights=None
):
    if weights is None:
        weights = lp_model.custom_graph(x_train, y=None)
    if sp.sparse.issparse(y_train_unlabeled):
        n_labelled = (y_train_unlabeled != -1).data.shape[0]
    else:
        n_labelled = np.array(np.where(y_train_unlabeled != -1)).shape[1]

    idx_to_flip = best_to_flip(n_flips, weights, n_labelled)
    acc, x_train, y_train = multiple_flips_training(
        x_train,
        y_train_unlabeled,
        lp_model,
        x_test,
        y_test,
        idx_to_flip[int(1 / 3 * n_flips) :],
    )
    clf = RandomForestClassifier(random_state=2020)
    # clf = svm.SVC(verbose=True,max_iter=1000,gamma='auto')
    clf.fit(x_train, y_train)
    acc_svm = clf.score(x_test, y_test)
    return acc_svm


def multiple_flips_accuracy_sklearn_rfc(
    x_train, y_train_unlabeled, y_train, lp_model, x_test, y_test, n_flips, weights=None
):
    if weights is None:
        weights = lp_model.custom_graph(x_train, y=None)

    n_labelled = np.array(np.where(y_train_unlabeled != -1)).shape[1]
    idx_to_flip = best_to_flip(n_flips, weights, n_labelled)
    acc, _, y_transduction = multiple_flips_training(
        x_train,
        y_train_unlabeled,
        lp_model,
        x_train[n_labelled:],
        y_train[n_labelled:],
        idx_to_flip,
    )

    clf = RandomForestClassifier(random_state=2020)
    # clf = svm.SVC(verbose=True,max_iter=1000,gamma='auto')
    clf.fit(x_train, y_transduction)
    acc_rfc = clf.score(x_test, y_test)

    return acc, acc_rfc


def find_redundancy(weights, n_labelled):
    # find some sample in Unlabelled set that have similar influence qty than sensible data
    weights_uu = np.copy(weights[n_labelled:, n_labelled:])
    np.fill_diagonal(weights_uu, 0.0)

    s = np.array([sum(i) for i in weights_uu])
    m = np.array([max(i) for i in weights_uu])
    an = (s - m) / s
    im = np.array(np.where((an < 0.5))).reshape(
        -1,
    )
    idx_max = np.argmax(weights_uu[im], axis=1)
    res = np.array([np.count_nonzero(idx_max == i) for i in range(weights_uu.shape[1])])
    return res


def reduce_influence_random(y_train_unlabeled, random_seed, n_flips):
    n_labelled = np.array(np.where(y_train_unlabeled != -1)).shape[1]
    n_unlabelled = y_train_unlabeled.shape[0] - n_labelled
    randomState = np.random.RandomState(random_seed)
    idx_to_flip = np.random.choice(n_unlabelled, n_flips)
    res = [
        int(randomState.choice(n_unlabelled, 1) + n_labelled)
        for i in range(1, n_flips + 1)
    ]
    return np.array(res)


def reduce_influence_ssim(x_train, y_train_unlabeled, lp_model, n_flips):
    weights = lp_model.custom_graph(x_train, y=None)
    n_labelled = np.array(np.where(y_train_unlabeled != -1)).shape[1]
    idx_to_flip = best_to_flip(n_flips, weights, n_labelled)

    res = np.empty((n_flips,))
    for i, idx in enumerate(idx_to_flip):
        maxssim_idx = -1
        maxssim_val = 0
        for c, u_sample in enumerate(x_train[n_labelled:]):
            similarity = ssim(x_train[idx], u_sample)
            # print(f"{c}:{similarity}")
            maxssim_val, maxssim_idx = (
                (similarity, c)
                if similarity > maxssim_val
                else (maxssim_val, maxssim_idx)
            )
        res[i] = int(maxssim_idx + n_labelled)

    return res
