from sklearn.semi_supervised import LabelSpreading, LabelPropagation
import datasets as ds
import numpy as np
import adversarial
from sklearn.metrics import pairwise
from label_propagation_cached import LabelPropagationCached
from sklearn.exceptions import ConvergenceWarning
import warnings
from config import Config
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)
config = Config(path="config.yaml")

def main_rf():
    for ds_name, attr in config.datasets.items():
        # print(f'************{ds_name}')
        x_train, x_test, y_train, y_test = attr["loader_function"]()
        random_seed = config.seed
        gamma = attr["gamma"]
        max_iter = attr["iter"]
        dataset = ds_name
        inductive_algo = "rfc"
        lp = LabelPropagationCached(gamma=gamma, max_iter=max_iter)
        # print(f"{dataset} loaded")
        for ssl_name, ssl in config.ssl_algo.items():
            # print(f'[{ssl_name}')
            ssl_instance = ssl(gamma=gamma, max_iter=max_iter)
            for p_labelled in [0.05, 0.15, 0.25]:
                # print(f"[{ds_name}][{ssl_name}][{p_labelled}]")
                n_labelled = int(y_train.shape[0] * p_labelled)
                random_state = np.random.RandomState(random_seed)
                (
                    y_train_shuffled,
                    x_train_shuffled,
                    labels_unlabelled,
                ) = ds.unlabel_shuffle_training_set(
                    y_train, x_train, n_labelled, random_state
                )

                weights = lp.custom_graph(x_train_shuffled, y=None)

                flip_budgets = [0, 0.05, 0.10, 0.15, 0.2]
                result = np.empty((len(flip_budgets),))
                for i, flips_p in enumerate(flip_budgets):
                    n_flips = int(flips_p * n_labelled)
                    print(f"[{ds_name}][{ssl_name}][{p_labelled}][fb:{flips_p}]")
                    result[i] = adversarial.multiple_flips_accuracy_rfc(
                        x_train_shuffled,
                        labels_unlabelled,
                        ssl_instance,
                        x_test,
                        y_test,
                        n_flips=n_flips,
                        weights=weights,
                    )
                    print(
                        f"[{ds_name}][{ssl_name}][{p_labelled}][fb:{flips_p}]\t -- \t{inductive_algo} [{result[i] * 100:.2f}%]"
                    )
                    # print(f"\t\t\t MLP [{flips_p}] [{result[i]}]")

                    np.savetxt(
                        f"{inductive_algo}_{ssl_name}_{dataset}_{p_labelled}", result
                    )

def main_mlp():
    for ds_name, attr in config.datasets.items():
        # print(f'************{ds_name}')
        x_train, x_test, y_train, y_test = attr["loader_function"]()
        random_seed = config.seed
        gamma = attr["gamma"]
        max_iter = attr["iter"]
        dataset = ds_name
        inductive_algo = "mlp"
        lp = LabelPropagationCached(gamma=gamma, max_iter=max_iter)
        # print(f"{dataset} loaded")
        for ssl_name, ssl in config.ssl_algo.items():
            # print(f'[{ssl_name}')
            ssl_instance = ssl(gamma=gamma, max_iter=max_iter)
            for p_labelled in [0.05, 0.15, 0.25]:
                # print(f"[{ds_name}][{ssl_name}][{p_labelled}]")
                n_labelled = int(y_train.shape[0] * p_labelled)
                random_state = np.random.RandomState(random_seed)
                (
                    y_train_shuffled,
                    x_train_shuffled,
                    labels_unlabelled,
                ) = ds.unlabel_shuffle_training_set(
                    y_train, x_train, n_labelled, random_state
                )

                weights = lp.custom_graph(x_train_shuffled, y=None)

                flip_budgets = [0, 0.05, 0.10, 0.15, 0.2]
                result = np.empty((len(flip_budgets),))
                for i, flips_p in enumerate(flip_budgets):
                    n_flips = int(flips_p * n_labelled)
                    print(f"[{ds_name}][{ssl_name}][{p_labelled}][fb:{flips_p}]")
                    result[i] = adversarial.multiple_flips_accuracy_mlp(
                        x_train_shuffled,
                        labels_unlabelled,
                        ssl_instance,
                        x_test,
                        y_test,
                        n_flips=n_flips,
                        weights=weights,
                    )
                    print(
                        f"[{ds_name}][{ssl_name}][{p_labelled}][fb:{flips_p}]\t -- \t{inductive_algo} [{result[i] * 100:.2f}%]"
                    )
                    # print(f"\t\t\t MLP [{flips_p}] [{result[i]}]")

                    np.savetxt(
                        f"{inductive_algo}_{ssl_name}_{dataset}_{p_labelled}", result
                    )

if __name__ == "__main__":
    main_rf()
    main_mlp()
