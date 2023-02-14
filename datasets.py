from sklearn.datasets import load_svmlight_file, fetch_rcv1
import numpy as np
import tensorflow as tf
from heloc_dataset import HELOCDataset, nan_preprocessing
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import SimpleImputer


def heloc_prep():
    heloc = HELOCDataset(custom_preprocessing=nan_preprocessing, dirpath="data").data()
    print("Size of HELOC dataset:", heloc.shape)
    print('Number of "Good" applicants:', np.sum(heloc["RiskPerformance"] == 1))
    print('Number of "Bad" applicants:', np.sum(heloc["RiskPerformance"] == 0))
    y = heloc.pop("RiskPerformance")
    X_train, X_test, y_train, y_test = train_test_split(
        heloc, y, random_state=2020, stratify=y
    )
    imp = SimpleImputer(missing_values=np.nan, strategy="mean")

    x = X_train.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    X_train = np.array(x_scaled)
    imp = imp.fit(X_train)
    X_train = imp.transform(X_train)

    x = X_test.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    X_test = np.array(x_scaled)
    imp = imp.fit(X_test)
    X_test = imp.transform(X_test)

    return X_train, X_test, np.array(y_train.tolist()), np.array(y_test.tolist())


def cifar10_prep():
    x_train, y_train = load_svmlight_file("./data/cifar10")
    x_test, y_test = load_svmlight_file("./data/cifar10.t")
    x_train = np.asarray(x_train.todense())
    x_test = np.asarray(x_test.todense())
    return x_train, x_test, y_train, y_test


def cifar10_prep_tf():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = np.reshape(x_train, (x_train.shape[0], 32 * 32 * 3))
    x_test = np.reshape(x_test, (x_test.shape[0], 32 * 32 * 3))

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255
    return (
        x_train,
        x_test,
        y_train.flatten().astype("int32"),
        y_test.flatten().astype("int32"),
    )


def mnist_prep_tf():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = np.reshape(x_train, (x_train.shape[0], 28 * 28))
    x_test = np.reshape(x_test, (x_test.shape[0], 28 * 28))

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255
    return (
        x_train,
        x_test,
        y_train.flatten().astype("int32"),
        y_test.flatten().astype("int32"),
    )


def cifarBinary_prep():
    c0, c1 = 3, 8
    x_train, x_test, y_train, y_test = cifar10_prep_tf()
    x_train_c0 = x_train[np.where(y_train == c0)]
    x_train_c1 = x_train[np.where(y_train == c1)]
    y_train_c0 = y_train[np.where(y_train == c0)]
    y_train_c1 = y_train[np.where(y_train == c1)]

    x_test_c0 = x_test[np.where(y_test == c0)]
    x_test_c1 = x_test[np.where(y_test == c1)]
    y_test_c0 = y_test[np.where(y_test == c0)]
    y_test_c1 = y_test[np.where(y_test == c1)]

    x_train = np.append(x_train_c0, x_train_c1, axis=0)
    y_train = np.append(y_train_c0, y_train_c1)

    x_test = np.append(x_test_c0, x_test_c1, axis=0)
    y_test = np.append(y_test_c0, y_test_c1)
    n_data = x_train.shape[0]
    y_train[y_train == c0] = 0
    y_test[y_test == c0] = 0
    y_train[y_train == c1] = 1
    y_test[y_test == c1] = 1
    return x_train, x_test, y_train, y_test


def mnist_prep():
    """
    Function that load the full scaled mnist dataset from svmlight file
    """
    x_train, y_train = load_svmlight_file("./data/mnist.scale")
    x_test, y_test = load_svmlight_file("./data/mnist.t.scale")
    x_train = np.asarray(x_train.todense())
    x_train = np.c_[x_train, np.zeros((x_train.shape[0], 4))]
    x_test = np.asarray(x_test.todense())
    x_test = np.c_[x_test, np.zeros((x_test.shape[0], 6))]
    print(f"Dataset loaded {y_train.shape[0]} samples")
    return x_train, x_test, y_train, y_test


def mnist178_prep():
    """
    Function that return a subset of mnist containing only 1s 7s 8s.

    The corresponding classes number are c_0 = 7, c_1 = 1, c_2 = 8
    """
    x_train, x_test, y_train, y_test = mnist_prep()

    x_train_1 = x_train[np.where(y_train == 1)]
    x_train_7 = x_train[np.where(y_train == 7)]
    x_train_8 = x_train[np.where(y_train == 8)]
    y_train_1 = y_train[np.where(y_train == 1)]
    y_train_7 = y_train[np.where(y_train == 7)]
    y_train_8 = y_train[np.where(y_train == 8)]

    x_test_1 = x_test[np.where(y_test == 1)]
    x_test_7 = x_test[np.where(y_test == 7)]
    x_test_8 = x_test[np.where(y_test == 8)]
    y_test_1 = y_test[np.where(y_test == 1)]
    y_test_7 = y_test[np.where(y_test == 7)]
    y_test_8 = y_test[np.where(y_test == 8)]

    x_train = np.append(x_train_1, x_train_7, axis=0)
    x_train = np.append(x_train, x_train_8, axis=0)
    y_train = np.append(y_train_1, y_train_7)
    y_train = np.append(y_train, y_train_8)

    x_test = np.append(x_test_1, x_test_7, axis=0)
    x_test = np.append(x_test, x_test_8, axis=0)
    y_test = np.append(y_test_1, y_test_7)
    y_test = np.append(y_test, y_test_8)

    y_train[y_train == 7] = 0
    y_train[y_train == 8] = 2
    y_test[y_test == 7] = 0
    y_test[y_test == 8] = 2

    return x_train, x_test, y_train, y_test


def mnist15_prep():
    """
    Function that return a subset of mnist containing only 1s and 5s.

    The corresponding classes number are c_0 = 5, c_1 = 1
    """
    x_train, x_test, y_train, y_test = mnist_prep()
    x_train_1 = x_train[np.where(y_train == 1)]
    x_train_5 = x_train[np.where(y_train == 5)]
    y_train_1 = y_train[np.where(y_train == 1)]
    y_train_5 = y_train[np.where(y_train == 5)]

    x_test_1 = x_test[np.where(y_test == 1)]
    x_test_5 = x_test[np.where(y_test == 5)]
    y_test_1 = y_test[np.where(y_test == 1)]
    y_test_5 = y_test[np.where(y_test == 5)]

    x_train = np.append(x_train_1, x_train_5, axis=0)
    y_train = np.append(y_train_1, y_train_5)

    x_test = np.append(x_test_1, x_test_5, axis=0)
    y_test = np.append(y_test_1, y_test_5)
    n_data = x_train.shape[0]
    y_train[y_train == 5] = 0
    y_test[y_test == 5] = 0
    return x_train, x_test, y_train, y_test


def mnist17_prep():
    """
    Function that return a subset of mnist containing only 1s and 7s.

    The corresponding classes number are c_0 = 7, c_1 = 1
    """
    x_train, x_test, y_train, y_test = mnist_prep()
    x_train_1 = x_train[np.where(y_train == 1)]
    x_train_7 = x_train[np.where(y_train == 7)]
    y_train_1 = y_train[np.where(y_train == 1)]
    y_train_7 = y_train[np.where(y_train == 7)]

    x_test_1 = x_test[np.where(y_test == 1)]
    x_test_7 = x_test[np.where(y_test == 7)]
    y_test_1 = y_test[np.where(y_test == 1)]
    y_test_7 = y_test[np.where(y_test == 7)]

    x_train = np.append(x_train_1, x_train_7, axis=0)
    y_train = np.append(y_train_1, y_train_7)

    x_test = np.append(x_test_1, x_test_7, axis=0)
    y_test = np.append(y_test_1, y_test_7)
    n_data = x_train.shape[0]
    y_train[y_train == 7] = 0
    y_test[y_test == 7] = 0
    return x_train, x_test, y_train, y_test


def rcv_prep_f():
    x_train, y_train = load_svmlight_file("./data/rcv1_train.binary")
    x_test, y_test = load_svmlight_file("./data/rcv1_test.binary")

    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0

    return (
        x_train.toarray(),
        x_test[:2500].toarray(),
        y_train,
        y_test[:2500],
    )


def rcv_prep():
    print("parsing RCV")
    rcv1 = fetch_rcv1(subset="train")
    x_train = rcv1.data
    y_train = rcv1.target
    rcv1 = fetch_rcv1(subset="test")
    x_test = rcv1.data
    y_test = rcv1.target
    ccat = rcv1.target_names.tolist().index("CCAT")
    ecat = rcv1.target_names.tolist().index("ECAT")
    gcat = rcv1.target_names.tolist().index("GCAT")
    mcat = rcv1.target_names.tolist().index("MCAT")

    train_indices = filter(
        lambda x: (
            bool(y_train[x, ccat] or y_train[x, ecat])
            != bool(y_train[x, gcat] or y_train[x, mcat])
        ),
        range(0, x_train.shape[0]),
    )
    test_indices = filter(
        lambda x: (
            bool(y_test[x, ccat] or y_test[x, ecat])
            != bool(y_test[x, gcat] or y_test[x, mcat])
        ),
        range(0, x_test.shape[0]),
    )

    train_indices = list(train_indices)
    test_indices = list(test_indices)[:2500]

    x_train = x_train[train_indices, :]
    y_train = y_train[train_indices, :]
    y_train = (y_train[:, ccat]) + (y_train[:, ecat]) >= 1

    x_test = x_test[test_indices, :]

    y_test = y_test[test_indices, :]
    y_test = (y_test[:, ccat]) + (y_test[:, ecat]) >= 1

    y_test = y_test.astype(int)
    y_train = y_train.astype(int)
    print("\parsing RCV")
    return (
        x_train.toarray(),
        x_test.toarray(),
        y_train.toarray().ravel(),
        y_test.toarray().ravel(),
    )


def shuffle(n, randomState):
    """
    Utility function to shuffle indexes.

    Parameters
    ----------
    n : int
        Total number of samples
    randomState : np.random.RandomState
        Random generator
    """
    n_data = n
    shuffle_idx = randomState.permutation(n_data)
    # do two inplace shuffle operations
    return shuffle_idx


def unlabel_shuffle_training_set(y_training, x_training, n_labeled, randomState):
    """Utility function to unlabel subset of the dataset.

    Parameters
    ----------
    y_training : darray
        training labels
    x_training : 2-D array
        training features
    n_labeled : int
        number of labeled samples
    randomState : np.random.RandomState
        Random generator
    """

    y_train = y_training.copy()
    x_train = x_training.copy()
    shuffle_i = shuffle(y_train.shape[0], randomState)
    x_train = x_train[shuffle_i]
    y_train = y_train[shuffle_i]
    labels_unlabelled = y_train.copy()
    labels_unlabelled[n_labeled:] = -1
    return y_train, x_train, labels_unlabelled
