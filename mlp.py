from sklearn.neural_network import MLPClassifier
import tensorflow as tf


class SimpleMlp:
    def __init__(self, random_state, input_shape=None, impl="sk", **fit_args):
        self.score = None
        self.mlp = None
        self.random_state = random_state
        self.switcher = {"sk": self.sk_mlp(), "tf": self.tf_mlp(input_shape)}

        self.switcher.get(impl)

    def fit(self, X, y, args=None):
        self.mlp.fit(X, y, **args)

    def sk_mlp(self):
        mlp_params = {
            "solver": "adam",
            "activation": "tanh",
            "alpha": 1e-5,
            "hidden_layer_sizes": 128,
            "random_state": self.random_state,
            "verbose": True,
            "early_stopping": True,
        }
        self.mlp = MLPClassifier(**mlp_params)
        self.score = self.mlp.score

    def tf_mlp(self, input_shape):
        if input_shape is None:
            raise ValueError("Please provide input shape for tf mlp")

        tf.random.set_seed(self.random_state)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=input_shape),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        self.mlp = model
        self.score = lambda x_test, y_test: self.mlp.evaluate(x_test, y_test)[1]
