from sklearn.semi_supervised import LabelSpreading, LabelPropagation
import datasets as ds
import numpy as np
import adversarial
from sklearn.metrics import pairwise
from label_propagation_cached import LabelPropagationCached
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

random_seed = 2020
# gamma : rcv1 = 20 ; cifar10 = 0.10 ; mnist = 5 ; heloc = 20,iter:5

gamma = 0.12
max_iter = 10

def single_lp_training(x_train, y_train, y_train_groundThruth, n_labelled, sample_to_test):
    lp = LabelPropagation(gamma=gamma, max_iter=max_iter)
    lp_model = lp.fit(x_train, y_train)
    transductive_lp_initial = lp_model.score(
        X=x_train[n_labelled:], y=y_train_groundThruth[n_labelled:])
    print(f"initial accuracy {transductive_lp_initial}" )

def nmis_value_computation(x_train, y_train, y_train_groundThruth, n_labelled):
    lp = LabelPropagationCached(gamma=gamma, max_iter=max_iter)
    print("Computing nmis value")
    weights = lp.custom_graph(x_train, y=None)
    print("/Computing nmis value")
    return adversarial.particularity_max_influence_single_flip(weights, n_labelled)


def exhaustive_lp_accuracy(
    x_train, y_train, y_train_groundThruth, n_labelled, sample_to_test
):
    lp = LabelPropagation(gamma=gamma, max_iter=max_iter)
    lp_model = lp.fit(x_train, y_train)
    transductive_lp_initial = lp_model.score(
        X=x_train[n_labelled:], y=y_train_groundThruth[n_labelled:])

    print(f"initial accuracy {transductive_lp_initial}" )

    transductive_lp = adversarial.exhaustive_accuracy_sklearn(
        x_train,
        y_train,
        LabelPropagation(gamma=gamma, max_iter=max_iter),
        x_test=x_train[n_labelled:],
        y_test=y_train_groundThruth[n_labelled:],
        sample_to_test=sample_to_test,
    )

    return transductive_lp_initial - transductive_lp


def exhaustive_ls_accuracy(x_train, y_train, y_train_groundThruth, n_labelled,sample_to_test):
    ls = LabelSpreading(gamma=gamma, max_iter=max_iter)
    ls_model = ls.fit(x_train, y_train)

    transductive_ls_initial = ls_model.score(
        X=x_train[n_labelled:], y=y_train_groundThruth[n_labelled:],
    )

    transductive_ls = adversarial.exhaustive_accuracy_sklearn(
        x_train,
        y_train,
        LabelSpreading(gamma=gamma, max_iter=max_iter),
        x_test=x_train[n_labelled:],
        y_test=y_train_groundThruth[n_labelled:],
        sample_to_test=sample_to_test,
    )

    return transductive_ls_initial - transductive_ls


def main():
    # x_train, x_test, y_train, y_test = ds.mnist15_prep()
    # x_train, x_test, y_train, y_test = ds.mnist17_prep()
    # x_train, x_test, y_train, y_test = ds.mnist178_prep()
    # x_train, x_test, y_train, y_test = ds.rcv_prep_f()
    # x_train, x_test, y_train, y_test = ds.cifarBinary_prep()
    # x_train, x_test, y_train, y_test = ds.heloc_prep()
    # x_train, x_test, y_train, y_test = ds.mnist_prep_tf()
    x_train, x_test, y_train, y_test = ds.cifar10_prep_tf()
    dataset = "cifar10"

    for p_labelled in [0.15]:
        n_labelled = int(y_train.shape[0] * p_labelled)
        randomState = np.random.RandomState(random_seed)
        (
            y_train_shuffled,
            x_train_shuffled,
            labels_unlabelled,
        ) = ds.unlabel_shuffle_training_set(y_train, x_train, n_labelled, randomState)

        print(f"Class repartition : {dict(zip(np.unique(labels_unlabelled,return_counts=True)[0][1:],np.unique(labels_unlabelled,return_counts=True)[1][1:]/n_labelled))}")
        # n_mis_value = nmis_value_computation(
        #     x_train_shuffled, labels_unlabelled, y_train_shuffled, n_labelled
        # )
        # np.savetxt(f"n_mis_value_{dataset}_{p_labelled}", n_mis_value)
        sample_size_p = 0.10
        sample_to_test = np.random.choice(
            range(0, n_labelled), size=int(n_labelled * sample_size_p)
        )
        print(f"Sample to test : {len(sample_to_test)}")
        accuracy = np.zeros((sample_to_test.shape[0], 3))
        # accuracy[:, 2] = n_mis_value[sample_to_test]
        single_lp_training(x_train_shuffled,
            labels_unlabelled,
            y_train_shuffled,
            n_labelled,
            sample_to_test,)
        
        np.savetxt(
            f"transductive_accuracy_exhaustive_lp_ls_{dataset}_{p_labelled}", accuracy,
        )


        accuracy[:, 0] = exhaustive_lp_accuracy(
            x_train_shuffled,
            labels_unlabelled,
            y_train_shuffled,
            n_labelled,
            sample_to_test,
        )

        np.savetxt(
            f"transductive_accuracy_exhaustive_lp_ls_{dataset}_{p_labelled}", accuracy,
        )

        accuracy[:, 1] = exhaustive_ls_accuracy(
            x_train_shuffled,
            labels_unlabelled,
            y_train_shuffled,
            n_labelled,
            sample_to_test,
        )

        accuracy[:, 2] = n_mis_value[sample_to_test]

        np.savetxt(
            f"transductive_accuracy_exhaustive_lp_ls_{dataset}_{p_labelled}", accuracy,
        )

if __name__ == "__main__":
    main()
