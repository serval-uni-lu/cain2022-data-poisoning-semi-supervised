import numpy as np
import os
import scipy as sp
import sklearn as scikit
from itertools import combinations
from sklearn import svm
import sklearn
from adversarial import poison_sample
from sklearn.ensemble import RandomForestClassifier
from tqdm import trange

# Following methods and algorithm inspired from Liu, Xuanqing
# Si, Si
# Zhu, Xiaojin
# Li, Yang
# Hsieh, Cho-Jui
# 2019 : A Unified Framework for Data Poisoning Attack to Graph-based Semi-supervised Learning


def probablistic_method(K, y_l, y_u, c, random_seed=2020):
    alpha = 0.5 * np.ones_like(y_l)
    tau = 0.5  # temperature
    lam = 0.1
    lr = 1.0e-5
    random = np.random.RandomState(random_seed)
    iterations = 100
    for i in trange(iterations):
        epsilon = random.gumbel(len(y_l)) - random.gumbel(len(y_l))
        tmp = np.exp((np.log(alpha / (1.0 - alpha)) + epsilon) / tau)
        z = 2.0 / (1.0 + tmp) - 1.0  # normalize z from [0, 1] to [-1, 1]
        v = y_l * z
        grad_v = -K.T @ (K @ v - y_u)
        grad_z = grad_v * y_l
        grad_alpha = (
            grad_z
            * (-2 * tmp / (1.0 + tmp) / (1.0 + tmp))
            * (1.0 / alpha + 1.0 / (1.0 - alpha))
            / tau
        )
        grad_alpha += lam * alpha  # add a regularization term
        alpha -= lr * grad_alpha
        # project alpha to [0, 1]
        alpha = np.clip(alpha, 1.0e-3, 1 - 1.0e-3)
    # evaluate function value
    idx = np.argsort(alpha)[::-1]
    d_y = np.ones_like(y_l)
    count = 0
    for i in idx:
        if alpha[i] > 0.5:
            d_y[i] = -1
            count += 1
            if count == c:
                break
    return d_y


def exhaustive_search(K, y_l, y_u, c):
    """Performs a brute-force search of all choices"""
    d_y = np.ones_like(y_l)
    if c == 0:  # fix a corner case
        return d_y
    original = func_val(K, y_l, d_y, y_u)
    progress = 0
    combi = combinations(range(len(y_l)), c)
    i = 0
    # total = len(list(combi))
    # generate combinations
    for selection in combi:
        print(f"{i}")
        i += 1
        selection = list(selection)
        flip = np.ones_like(y_l)
        flip[selection] = -1
        val = func_val(K, y_l, flip, y_u)
        if val - original < progress:
            progress = val - original
            d_y = flip
    return d_y


def greedy_method(K, y_l, y_u, c):
    """
    make sure c < len(y_l)
    """
    nu, nl = K.shape
    # see paper for algorithm details
    d_y = np.ones(nl, dtype=int)
    for i in trange(c):
        # print(f"greedy iteration :{i}/{c}")
        # calculate the original function value
        func_original = func_val(K, y_l, d_y, y_u)
        progress = 0
        best_idx = -1
        # find the i-th best element to flip
        for j, is_flipped in enumerate(d_y):

            if is_flipped == -1:
                # if this element is flipped, skip it
                continue
            # try flip this element and check the function decrement
            d_y[j] = -1
            func_try = func_val(K, y_l, d_y, y_u)
            if func_try - func_original < progress:
                # made a better progress
                progress = func_try - func_original
                best_idx = j
                # print(f"{i} progress : {progress}")
            # reset this element
            d_y[j] = 1
        # greedy
        if best_idx >= 0:
            d_y[best_idx] = -1
        else:
            # not improvable
            break
    return d_y


def func_val(K, y_l, d_y, y_u):
    # calculate the objective function value
    tmp = np.sign(K @ (y_l * d_y)) - y_u
    return -0.5 * np.sum(tmp * tmp)


def similarity_matrix(X, gamma, usecache=False):
    data_name = "sim_matrix_mnist"
    cache_file = f"./data/{data_name}.npy"
    if os.path.exists(cache_file) and usecache:
        tmp = np.load(cache_file)
    else:
        if sp.sparse.issparse(X):
            X = X.tocoo()
            tmp = X @ X.T
            tmp = np.asarray(tmp.todense())
        else:
            tmp = X @ X.T
        if usecache:
            np.save(cache_file, tmp)
    n_data = X.shape[0]
    diag = np.diag(tmp)
    S = gamma * (2 * tmp - diag.reshape(1, n_data) - diag.reshape(n_data, 1))
    return np.exp(S)


def perturb_y_classification(n_labeled, x_train, y_train, c_max, lp=None, gamma=6):
    if c_max > 0:
        X = x_train
        # scaler = StandardScaler()
        # scaler.fit(X)
        # features = scaler.transform(X)
        y_train_mod = np.copy(y_train)
        y_train_mod[y_train_mod == 0] = -1
        y_l = y_train_mod[:n_labeled]
        n_l = n_labeled
        y_u = y_train_mod[n_labeled:]
        S = lp.affinity_mtx if lp is not None else similarity_matrix(X, gamma)
        # S = similarity_matrix(X, gamma)
        D = np.diag(np.sum(S, axis=1, keepdims=False))
        Suu = S[n_l:, n_l:]
        Duu = D[n_l:, n_l:]
        Sul = S[n_l:, :n_l]
        K = np.linalg.inv(Duu - Suu) @ Sul
        # greedy / threshold / probablistic / exhaustive_search
        perturbation = greedy_method(K, y_l, y_u, c_max)
        # perturbation = exhaustive_search(K, y_l, y_u, c_max)
        # perturbation = probablistic_method(K, y_l, y_u, c_max)
        return perturbation
    return []


def multiple_flips_greedy(
    x_train,
    y_train,
    lp_model,
    x_test,
    y_test,
    n_flips,
    n_labelled,
    gamma=6,
    x_test_t=None,
    y_test_t=None,
    clf=None,
):

    perturbation = perturb_y_classification(
        n_labelled,
        x_train,
        y_train,
        n_flips,
        gamma=gamma,
    )
    classes = np.unique(y_train)
    classes = classes[classes != -1]
    n_perturb = np.array(np.where(perturbation == -1)).shape[1]
    idxs_to_flip = np.array(np.where(perturbation == -1)).reshape((n_perturb,))
    y_train_unlabeled = np.copy(y_train)
    y_train_unlabeled[n_labelled:] = -1
    y_train_flip = np.copy(y_train_unlabeled)
    n_to_flip = idxs_to_flip.shape[0]

    for index in range(n_to_flip):
        to_flip = idxs_to_flip[index]
        y_train_flip[to_flip] = poison_sample(y_train_flip[to_flip], classes)

    lp_model.fit(x_train, y_train_flip)
    acc_rbf = lp_model.score(x_test, y_test)
    # clf = svm.SVC()
    # clf.fit(x_train, lp_model.transduction_)
    # acc_svm = clf.score(x_test, y_test)
    # acc_rbf = 0
    # clf = sl_algo(random_state=2020)
    clf.fit(x_train, lp_model.transduction_, args={"epochs": 5, "verbose": 2})
    acc_rfc = clf.score(x_test_t, y_test_t)
    return acc_rbf, acc_rfc


def multiple_flips_random(
    x_train, y_train, lp_model, x_test, y_test, n_flips, n_labelled, seed
):
    randomState = np.random.RandomState(seed)
    idxs_to_flip = randomState.choice(n_labelled, n_flips)
    y_train_unlabeled = np.copy(y_train)
    y_train_unlabeled[n_labelled:] = -1
    y_train_flip = np.copy(y_train_unlabeled)
    n_to_flip = idxs_to_flip.shape[0]
    classes = np.unique(y_train_unlabeled)
    classes = classes[classes != -1]
    for index in range(n_to_flip):
        to_flip = idxs_to_flip[index]
        y_train_flip[to_flip] = poison_sample(y_train_flip[to_flip], classes)

    lp_model.fit(x_train, y_train_flip)
    # acc_rbf = lp_model.score(x_test, y_test)
    clf = RandomForestClassifier(random_state=2020)
    clf.fit(x_train, lp_model.transduction_)
    acc_svm = clf.score(x_test, y_test)
    return acc_svm
    # return acc_rbf
