"""LongShort Term Memory Fully Convolutional Network (LSTM-FCN)."""

__author__ = ["jnrusson1", "solen0id"]

__all__ = ["LSTMFCNClassifier"]

from sktime.classification.deep_learning._tensorflow import BaseDeepClassifier
from sktime.networks.lstmfcn import LSTMFCNNetwork


class LSTMFCNClassifier(BaseDeepClassifier):
    """Implementation of LSTMFCNClassifier from Karim et al (2019) [1].

    Combines an LSTM arm with a CNN arm. Optionally uses an attention mechanism in the
    LSTM which the author indicates provides improved performance.

    Parameters
    ----------
    n_epochs: int, default=2000
     the number of epochs to train the model
    batch_size: int, default=128
        the number of samples per gradient update.
    dropout: float, default=0.8
        controls dropout rate of LSTM layer
    kernel_sizes: list of ints, default=[8, 5, 3]
        specifying the length of the 1D convolution windows
    filter_sizes: int, list of ints, default=[128, 256, 128]
        size of filter for each conv layer
    lstm_size: int, default=8
        output dimension for LSTM layer
    attention: boolean, default=False
        If True, uses custom attention LSTM layer
    callbacks: keras callbacks, default=ReduceLRonPlateau
        Keras callbacks to use such as learning rate reduction or saving best model
        based on validation error
    verbose: 'auto', 0, 1, or 2. Verbosity mode.
        0 = silent, 1 = progress bar, 2 = one line per epoch.
        'auto' defaults to 1 for most cases, but 2 when used with
        ``ParameterServerStrategy``. Note that the progress bar is not
        particularly useful when logged to a file, so verbose=2 is
        recommended when not running interactively (eg, in a production
        environment).
    random_state : int or None, default=None
        Seed for random, integer.

    References
    ----------
    .. [1] Karim et al. Multivariate LSTM-FCNs for Time Series Classification, 2019
    https://arxiv.org/pdf/1801.04503.pdf

    Examples
    --------
    >>> import sktime.classification.deep_learning as dl_clf  # doctest: +SKIP
    >>> from dl_clf.lstmfcn import LSTMFCNClassifier  # doctest: +SKIP
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True)
    >>> lstmfcn = FCNClassifier(n_epochs=20,batch_size=4)  # doctest: +SKIP
    >>> lstmfcn.fit(X_train, y_train)  # doctest: +SKIP
    FCNClassifier(...)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["jnrusson1", "solen0id"],
        "maintainers": ["jnrusson1", "solen0id"],
        "python_dependencies": "tensorflow",
        # estimator type handled by parent class
    }

    def __init__(
        self,
        n_epochs=2000,
        batch_size=128,
        dropout=0.8,
        kernel_sizes=(8, 5, 3),
        filter_sizes=(128, 256, 128),
        lstm_size=8,
        attention=False,
        callbacks=None,
        random_state=None,
        verbose=0,
    ):
        # predefined
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.kernel_sizes = kernel_sizes
        self.filter_sizes = filter_sizes
        self.lstm_size = lstm_size
        self.dropout = dropout
        self.attention = attention

        self.callbacks = callbacks
        self.random_state = random_state
        self.verbose = verbose

        super().__init__()

        self._network = LSTMFCNNetwork(
            kernel_sizes=self.kernel_sizes,
            filter_sizes=self.filter_sizes,
            random_state=self.random_state,
            lstm_size=self.lstm_size,
            dropout=self.dropout,
            attention=self.attention,
        )

    def build_model(self, input_shape, n_classes, **kwargs):
        """
        Construct a compiled, un-trained, keras model that is ready for training.

        ----------
        input_shape : tuple
            The shape of the data fed into the input layer
        n_classes: int
            The number of classes, which shall become the size of the output
             layer

        Returns
        -------
        output : a compiled Keras Model
        """
        import tensorflow as tf
        from tensorflow import keras

        tf.random.set_seed(self.random_state)

        input_layers, output_layer = self._network.build_network(input_shape, **kwargs)

        output_layer = keras.layers.Dense(units=n_classes, activation="softmax")(
            output_layer
        )

        model = keras.models.Model(inputs=input_layers, outputs=output_layer)

        model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )
        # .get_params() returns an empty list for callback.
        # inconsistent with function initial run where callbacks was set to None
        self._callbacks = self.callbacks or None

        return model

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``.
        """
        from sktime.utils.dependencies import _check_soft_dependencies

        param1 = {
            "n_epochs": 25,
            "batch_size": 4,
            "kernel_sizes": (3, 2, 1),
            "filter_sizes": (2, 4, 2),
        }

        param2 = {
            "n_epochs": 5,
            "batch_size": 2,
            "kernel_sizes": (3, 2, 1),
            "filter_sizes": (2, 4, 2),
            "lstm_size": 2,
            "attention": True,
        }
        test_params = [param1, param2]

        if _check_soft_dependencies("keras", severity="none"):
            from keras.callbacks import LambdaCallback

            test_params.append(
                {
                    "n_epochs": 2,
                    "callbacks": [LambdaCallback()],
                }
            )

        return test_params
