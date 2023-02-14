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
from mlp import SimpleMlp

warnings.filterwarnings("ignore", category=ConvergenceWarning)
config = Config(path="config.yaml")


def efficiency():
    for ds_name, attr in config.datasets.items():
        x_train, x_test, y_train, y_test = attr["loader_function"]()
        random_seed = config.seed
        gamma = attr["gamma"]
        max_iter = attr["iter"]
        dataset = ds_name
        randomState = np.random.RandomState(random_seed)
        # x_train, x_test, y_train, y_test = ds.mnist17_prep()
        # x_train, x_test, y_train, y_test = ds.mnist15_prep()
        # x_train, x_test, y_train, y_test = ds.mnist178_prep()
        # x_train, x_test, y_train, y_test = ds.rcv_prep_f()
        # x_train, x_test, y_train, y_test = ds.cifarBinary_prep()

        p_labelled = 0.05
        n_labelled = int(y_train.shape[0] * p_labelled)

        (
            y_train_shuffled,
            x_train_shuffled,
            labels_unlabelled,
        ) = ds.unlabel_shuffle_training_set(y_train, x_train, n_labelled, randomState)

        flips_p = 0.2
        n_flips = int(flips_p * n_labelled)
        repetition = 50
        results_gm = np.empty((repetition,))
        results_im = np.empty((repetition,))
        print(f"{dataset} loaded")
        print(f"[{ds_name}][{p_labelled}][fb:{flips_p}]")
        for i in range(repetition):
            # print(f"[{i}]")
            start_time = time.time()
            perturb_y_classification(
                n_labelled, x_train_shuffled, y_train_shuffled, n_flips, gamma=gamma
            )
            t_gm = time.time() - start_time
            print(
                f"[{i}/{repetition}][{ds_name}][{p_labelled}][fb:{flips_p}]\t PM {t_gm:.2f}s"
            )
            # print("---Probabilistic %s seconds ---" % t_gm)
            results_gm[i] = t_gm

            start_time = time.time()
            weights = pairwise.rbf_kernel(
                x_train_shuffled, x_train_shuffled, gamma=gamma
            )
            adversarial.best_to_flip(n_flips, weights, n_labelled)
            t_im = time.time() - start_time
            print(
                f"[{i}/{repetition}][{ds_name}][{p_labelled}][fb:{flips_p}]\t IM {t_im:2f}s"
            )
            # print("---Influence %s seconds ---" % t_im)
            results_im[i] = t_im
            np.savetxt(f"time_pm_{dataset}_{p_labelled}", results_gm)
            np.savetxt(f"time_im_{dataset}_{p_labelled}", results_im)


def transductive_accuracy():

    for ds_name, attr in config.datasets.items():
        x_train, x_test, y_train, y_test = attr["loader_function"]()
        random_seed = config.seed
        gamma = attr["gamma"]
        max_iter = attr["iter"]
        dataset = ds_name
        randomState = np.random.RandomState(random_seed)

        p_labelled = config.p_labelled
        flip_budgets = config.flip_budgets
        for i, p_l in enumerate(p_labelled):
            n_labelled = int(y_train.shape[0] * p_l)
            (
                y_train_shuffled,
                x_train_shuffled,
                labels_unlabelled,
            ) = ds.unlabel_shuffle_training_set(
                y_train, x_train, n_labelled, randomState
            )

            lp = LabelPropagationCached(gamma=gamma, max_iter=max_iter)
            # ls = LabelSpreading(gamma=gamma, max_iter=10)
            gssl = "lp"
            weights = pairwise.rbf_kernel(
                x_train_shuffled, x_train_shuffled, gamma=gamma
            )
            flip_budget = int(0.2 * n_labelled)
            result_gm_t = np.empty((len(flip_budgets),))
            result_im_t = np.empty((len(flip_budgets),))
            result_gm_i = np.empty((len(flip_budgets),))
            result_im_i = np.empty((len(flip_budgets),))

            print(
                f"Transductive & inductive study on {dataset} [{p_l} labelled] for max_flip {flip_budget} with {gssl}"
            )
            for i, flips_p in enumerate(flip_budgets):
                n_flips = int(flips_p * n_labelled)
                print(f"{flips_p}:{n_flips}/{n_labelled}")
                acc_t, acc_i = multiple_flips_greedy(
                    x_train_shuffled,
                    labels_unlabelled,
                    lp,
                    x_train_shuffled[n_labelled:],
                    y_train_shuffled[n_labelled:],
                    n_flips,
                    n_labelled,
                    gamma=gamma,
                    x_test_t=x_test,
                    y_test_t=y_test,
                    clf=SimpleMlp(
                        random_state=2020, impl="tf", input_shape=(x_train.shape[1],)
                    ),
                )
                print(
                    f"Greedy method : [{i}]=>[transduction {acc_t}][induction {acc_i}]"
                )
                result_gm_t[i] = acc_t
                result_gm_i[i] = acc_i

                # acc_t, acc_i = adversarial.multiple_flips_accuracy_sklearn(
                #     x_train_shuffled,
                #     labels_unlabelled,
                #     lp,
                #     x_train_shuffled[n_labelled:],
                #     y_train_shuffled[n_labelled:],
                #     n_flips,
                #     weights=weights,
                #     x_test_t=x_test,
                #     y_test_t=y_test,
                #     clf=SimpleMlp(
                #         random_state=2020, impl="tf", input_shape=(x_train.shape[1],)
                #     ),
                # )
                # result_im_t[i] = acc_t
                # result_im_i[i] = acc_i
                # print(
                #     f"Influence method : [{i}]=>[transduction {acc_t}][induction {acc_i}]"
                # )
                np.savetxt(f"{gssl}_{dataset}_{p_l}_gm_t", result_gm_t)
                # np.savetxt(f"{gssl}_{dataset}_{p_l}_im_t", result_im_t)
                np.savetxt(f"{gssl}_{dataset}_{p_l}_gm_i", result_gm_i)
                # np.savetxt(f"{gssl}_{dataset}_{p_l}_im_i", result_im_i)


def inductive_accuracy():
    randomState = np.random.RandomState(random_seed)
    # x_train, x_test, y_train, y_test = ds.mnist17_prep()
    # x_train, x_test, y_train, y_test = ds.mnist15_prep()
    # x_train, x_test, y_train, y_test = ds.mnist178_prep()
    x_train, x_test, y_train, y_test = ds.rcv_prep_f()
    # x_train, x_test, y_train, y_test = ds.cifarBinary_prep()
    dataset = "rcv"

    p_labelled = 0.25
    flip_budgets = [0, 0.05, 0.10, 0.15, 0.2]
    n_labelled = int(y_train.shape[0] * p_labelled)

    (
        y_train_shuffled,
        x_train_shuffled,
        labels_unlabelled,
    ) = ds.unlabel_shuffle_training_set(y_train, x_train, n_labelled, randomState)

    lp = LabelPropagationCached(gamma=gamma, max_iter=max_iter)
    # ls = LabelSpreading(gamma=gamma, max_iter=10)
    # weights = pairwise.rbf_kernel(x_train_shuffled, x_train_shuffled, gamma=gamma)
    gssl = "lp"
    result_gm = np.empty((len(flip_budgets),))
    result_im = np.empty((len(flip_budgets),))
    result_rm = np.empty((len(flip_budgets),))
    flip_budget = int(0.2 * n_labelled)
    print(f"Inductive study on {dataset} for flip budget [{flip_budgets}] with {gssl}")
    for i, flips_p in enumerate(flip_budgets):
        n_flips = int(flips_p * n_labelled)
        print(f"{flips_p}:{n_flips}/{n_labelled}")
        acc = multiple_flips_random(
            x_train_shuffled,
            labels_unlabelled,
            lp,
            x_test,
            y_test,
            n_flips,
            n_labelled,
            seed=2020,
        )
        print(f"random [{i}] [{acc}]")
        result_rm[i] = acc
        _, acc = multiple_flips_greedy(
            x_train_shuffled,
            labels_unlabelled,
            lp,
            x_test,
            y_test,
            n_flips,
            n_labelled,
            gamma=gamma,
        )
        print(f"greedy [{i}] [{acc}]")
        result_gm[i] = acc

        # acc = adversarial.multiple_flips_accuracy_rfc(
        #     x_train_shuffled,
        #     labels_unlabelled,
        #     ls,
        #     x_test,
        #     y_test,
        #     i,
        #     weights=weights,
        # )
        # result_im[i] = acc
        # print(f"influence [{i}] [{acc}]")

        np.savetxt(f"rfc_{gssl}_{dataset}_{p_labelled}_rm", result_rm)
        np.savetxt(f"rfc_{gssl}_{dataset}_{p_labelled}_gm", result_gm)
        # np.savetxt("induc_im_15_ls", result_im)


def main():
    # efficiency()
    transductive_accuracy()
    # inductive_accuracy()


if __name__ == "__main__":
    main()
