"""Abstract base class for the Keras neural network classifiers.

The reason for this class between BaseClassifier and deep_learning classifiers is
because we can generalise tags, _predict and _predict_proba
"""

__author__ = ["James-Large", "ABostrom", "TonyBagnall", "aurunmpegasus", "achieveordie"]
__all__ = ["BaseDeepClassifier"]

import os
import threading
from abc import abstractmethod
from copy import deepcopy

import keras
import numpy as np
import tensorflow as tf
import re
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_random_state

from sktime.base._base import SERIALIZATION_FORMATS
from sktime.base._base_panel import _is_lazy_panel
from sktime.classification.base import BaseClassifier
from sktime.utils.dependencies import _check_soft_dependencies

 
# keras 3: PyDataset; keras 2: Sequence (same protocol, no worker kwargs)
_PyDataset = getattr(keras.utils, "PyDataset", keras.utils.Sequence)

class _LazyPanelDataset(_PyDataset):
    """Keras PyDataset reading a lazily loaded sktime panel batch by batch.

    Parameters
    ----------
    X : lazy panel, see ``sktime.base._base_panel._is_lazy_panel``
        yields np.ndarray of shape (batch, n_dimensions, series_length)
    prepare : callable
        maps a numpy3D batch to the keras input, i.e., the estimator's
        ``_prepare_data``, applied per batch instead of to the full array
    y_idx : 1D np.ndarray of int or None
        column index of the one-hot target per instance; None for prediction
    n_classes : int
        number of one-hot columns
    batch_size : int
    shuffle : bool
        whether to reshuffle instances (across the whole X) after every epoch
    seed : int or None
        seed for the shuffling
    **kwargs : passed to PyDataset, i.e., workers, use_multiprocessing,
        max_queue_size (keras 3 only),  ``use_multiprocessing=True`` is not
        supported with ``shuffle=True``
    """

    def __init__(
        self,
        X,
        prepare,
        y_idx=None,
        n_classes=None,
        batch_size=32,
        shuffle=False,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if shuffle and kwargs.get("use_multiprocessing", False):
            raise ValueError(
                "use_multiprocessing=True is not supported for training on lazily "
                "loaded X, use threads instead, e.g., loader_kwargs={'workers': 4}"
            )
        self.X = X
        self.prepare = prepare
        self.y_idx = None if y_idx is None else np.asarray(y_idx)
        self._eye = None if y_idx is None else np.eye(n_classes, dtype="float32")
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self._rng = np.random.default_rng(seed)
        self._order = np.arange(len(X))
        self._cursor = len(X)
        self._lock = threading.Lock()

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def _next_shuffled_indices(self):
        """Next batch of the running pass over a random permutation (thread-safe)."""
        with self._lock:
            if self._cursor >= len(self._order):
                self._rng.shuffle(self._order)
                self._cursor = 0
            lo = self._cursor
            self._cursor += self.batch_size
            return self._order[lo : lo + self.batch_size].copy()

    def __getitem__(self, i):
        if i < 0 or i >= len(self):
            raise IndexError(f"batch index {i} out of range [0, {len(self)})")
        if self.shuffle:
            idx = self._next_shuffled_indices()
        else:
            idx = np.arange(i * self.batch_size, min((i + 1) * self.batch_size, len(self.X)))
        # sorted indices: faster reads, order within a batch is irrelevant
        idx = np.sort(idx)
        Xb = self.prepare(np.asarray(self.X[idx]))
        if isinstance(Xb, list):
            # multi-input models (e.g., MCDCNN): tf.data requires tuples, not lists
            Xb = tuple(Xb)
        if self.y_idx is None:
            # 1-tuple, so that list-valued inputs (e.g., MCDCNN) are not
            # misinterpreted by keras as (x, y)
            return (Xb,)
        return Xb, self._eye[self.y_idx[idx]]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()


class BaseDeepClassifier(BaseClassifier):
    """Abstract base class for deep learning time series classifiers.

    The base classifier provides a deep learning default method for
    _predict and _predict_proba, and provides a new abstract method for building a
    model.

    Parameters
    ----------
    batch_size : int, default = 40
        training batch size for the model

    Attributes
    ----------
    self.model_ - the fitted DL model
    """

    _tags = {
        "X_inner_mtype": "numpy3D",
        "capability:multivariate": True,
        "capability:lazy_panel": True,
        "python_dependencies": "tensorflow",
    }

    @abstractmethod
    def build_model(self, input_shape, n_classes, **kwargs):
        """Construct a compiled, un-trained, keras model that is ready for training.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer
        n_classes: int
            The number of classes, which shall become the size of the output
            layer

        Returns
        -------
        A compiled Keras Model
        """
        ...

    def summary(self):
        """Summary function to return the losses/metrics for model fit.

        Returns
        -------
        history: dict or None,
            Dictionary containing model's train/validation losses and metrics
        """
        return self.history.history if self.history is not None else None

    def _fit(self, X, y, X_val=None, y_val=None, skip_setup=False, **kwargs):
        """Fit the classifier on the training set (X, y).

        Parameters
        ----------
        X : np.ndarray of shape = (n_instances (n), n_dimensions (d), series_length (m))
            The training input samples.
        y : np.ndarray of shape n
            The training data class labels.
        X_val : np.ndarray of shape = (n_instances (n), n_dimensions (d), series_length (m))
            The validation input samples.
        y_val : np.ndarray of shape n
            The validation data class labels.
        skip_setup : bool, default = False
        **kwargs : additional fitting parameter
            ``loader_kwargs`` (dict) is popped and passed to the keras PyDataset
            used for lazily loaded X/X_val, e.g., ``{"workers": 4}``

        Returns
        -------
        self : object
        """
        # X and/or X_val may be lazily loaded panels -> fed to keras as PyDataset
        self._loader_kwargs = dict(kwargs.pop("loader_kwargs", None) or {})
        lazy, lazy_val = _is_lazy_panel(X), _is_lazy_panel(X_val)
        fit_kwargs = {}

        # prepare input & target data
        if lazy:
            self._prepare_data(np.asarray(X[np.arange(1)]))
            train_data = self._make_lazy_dataset(
                X, y, self.batch_size, shuffle=True
            )
            X, y_onehot = train_data, None
        else:
            X = self._prepare_data(X)
            y_onehot = self._convert_y_to_keras(y)
            fit_kwargs["batch_size"] = self.batch_size

        # compose validation data if both given
        if X_val is not None and y_val is not None:
            if lazy_val:
                # keras semantics: validation_batch_size defaults to batch_size
                validation_data = self._make_lazy_dataset(
                    X_val, y_val, self.pred_batch_size or self.batch_size
                )
            else:
                validation_data = (
                    self._prepare_data(X_val),
                    self._convert_y_to_keras(y_val),
                )
                fit_kwargs["validation_batch_size"] = self.pred_batch_size
        else:
            validation_data = None

        # initialise model and callbacks
        if not skip_setup:
            check_random_state(self.random_state)
            self.model_ = self.build_model(self.input_shape, self.n_classes_)
        self._configure_callbacks()

        if self.verbose:
            self.model_.summary()

        # fit model
        if self.n_epochs:
            self.history = self.model_.fit(
                X,
                y_onehot,
                epochs=self.n_epochs,
                verbose=self.verbose,
                validation_data=validation_data,
                callbacks=self.callbacks,
                **fit_kwargs,
                **kwargs,
            )

            # check callbacks for checkpoints
            ckpt_callback = [
                isinstance(cbk, keras.callbacks.ModelCheckpoint)
                for cbk in self.callbacks
            ]
            if any(ckpt_callback):
                cbk = self.callbacks[ckpt_callback.index(True)]
                self.model_ = self._load_best_model_from_checkpoints(
                    os.path.dirname(cbk.filepath)
                )

        return self

    def _load_best_model_from_checkpoints(self, checkpoint_dir):
        # if only one checkpoint, return it
        ckpts = [f for f in os.listdir(checkpoint_dir) if f.endswith(".keras")]
        if not ckpts:
            raise FileNotFoundError(f"No .keras checkpoints in {checkpoint_dir}")
        # if only one checkpoint, return it
        if len(ckpts) == 1:
            return keras.models.load_model(os.path.join(checkpoint_dir, ckpts[0]))
        # else: find best one with minimum validation loss
        pattern = re.compile(r"checkpoint-epoch-(\d+)-val_loss-([0-9.]+)\.(tf|keras)$")
        best_loss = float("inf")
        best_model_path = None
        for fname in os.listdir(checkpoint_dir):
            match = pattern.match(fname)
            if match:
                val_loss = float(match.group(2))
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model_path = os.path.join(checkpoint_dir, fname)
        if best_model_path is None:
            raise FileNotFoundError("No checkpoint files found in directory.")
        print(f"Loading best model from: {best_model_path}")
        return keras.models.load_model(best_model_path)

    def _predict(self, X, **kwargs):
        probs = self._predict_proba(X, **kwargs)
        max_indices = tf.argmax(probs, axis=1)
        return_obj = tf.gather(self.classes_, max_indices)
        return return_obj

    def _predict_proba(self, X, **kwargs):
        """Find probability estimates for each class for all cases in X.

        Parameters
        ----------
        X : an np.ndarray of shape = (n_instances, n_dimensions, series_length)
            The training input samples.

        Returns
        -------
        output : array of shape = [n_instances, n_classes] of probabilities
        """
        if _is_lazy_panel(X):
            # keras semantics: predict batch_size defaults to 32
            X = self._make_lazy_dataset(X, None, self.pred_batch_size or 32)
            probs = self.model_.predict(X, **kwargs)
        else:
            # Transpose to work correctly with keras
            X = self._prepare_data(X)
            # The following is the slow part of sktime
            # Takes approx. 95% of the time
            # Convert_to_tensor if not explictly called, internally called by .predict
            X = tf.convert_to_tensor(X)
            probs = self.model_.predict(X, self.pred_batch_size, **kwargs)

        # check if binary classification
        if probs.shape[1] == 1:
            # first column is probability of class 0 and second is of class 1
            probs = np.hstack([1 - probs, probs])
        probs = probs / probs.sum(axis=1, keepdims=1)
        return probs

    def _make_lazy_dataset(self, X, y, batch_size, shuffle=False):
        """Wrap a lazily loaded panel (and labels) into a keras PyDataset.

        Labels are kept as integer column indices and one-hot encoded per batch,
        with the same column order as ``_convert_y_to_keras``.
        """
        y_idx = None if y is None else self._encode_y_to_index(y)
        seed = self.random_state if isinstance(self.random_state, int) else None
        return _LazyPanelDataset(
            X,
            prepare=self._prepare_data,
            y_idx=y_idx,
            n_classes=len(self._class_dictionary),
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            **getattr(self, "_loader_kwargs", {}),
        )

    def _encode_y_to_index(self, y):
        """Map labels to one-hot column indices (order of self._class_dictionary)."""
        y = np.asarray(y).reshape(-1)
        keys = np.asarray(list(self._class_dictionary.keys()))
        order = np.argsort(keys, kind="stable")
        pos = np.clip(np.searchsorted(keys[order], y), 0, len(keys) - 1)
        y_idx = order[pos]
        unknown = keys[y_idx] != y
        if unknown.any():
            raise ValueError(
                f"y contains labels not in the class dictionary: {np.unique(y[unknown])}"
            )
        return y_idx.astype(np.int64)

    def _convert_y_to_keras(self, y):
        """Convert y to required Keras format."""
        # in sklearn 1.2, sparse was renamed to sparse_output
        if _check_soft_dependencies("scikit-learn>=1.2", severity="none"):
            sparse_kw = {"sparse_output": False}
        else:
            sparse_kw = {"sparse": False}

        # encode target values as integers
        y = np.vectorize(self._class_dictionary.get)(y)

        # encode target values in onehot fashion
        self.onehot_encoder = OneHotEncoder(
            categories=[np.fromiter(self._class_dictionary.values(), dtype="int")],
            **sparse_kw,
        )
        y = y.reshape(len(y), 1)
        y = self.onehot_encoder.fit_transform(y)
        return y

    def _configure_callbacks(self):
        """Add callbacks to the model."""
        self.callbacks = deepcopy(self.callbacks) if self.callbacks else []

    def _prepare_data(self, X):
        """Prepare input data for fitting/prediction mode.

        Parameters
        ----------
        X : np.ndarray of shape = (n_instances (n), n_dimensions (d), series_length (m))
            The training input samples.

        Returns
        -------
        X : np.ndarray suitable for input to a Keras model
        """
        X = X.transpose(0, 2, 1)
        self.input_shape = X.shape[1:]
        return X

    def __getstate__(self):
        """Get Dict config that will be used when a serialization method is called.

        Returns
        -------
        copy : dict, the config to be serialized
        """
        from tensorflow.keras.optimizers import Optimizer, serialize

        copy = self.__dict__.copy()

        # Either optimizer might not exist at all(-1),
        # or it does and takes a value(including None)
        optimizer_attr = copy.get("optimizer", -1)
        if not isinstance(optimizer_attr, str):
            if optimizer_attr is None:
                # if it is None, then save it as 0, so it can be
                # later correctly restored as None
                copy["optimizer"] = 0
            elif optimizer_attr == -1:
                # if an `optimizer` parameter doesn't exist at all
                # save it as -1
                copy["optimizer"] = -1
            elif isinstance(optimizer_attr, Optimizer):
                copy["optimizer"] = serialize(optimizer_attr)
            else:
                raise ValueError(
                    f"`optimizer` of type {type(optimizer_attr)} cannot be "
                    "serialized, it should either be absent/None/str/"
                    "tf.keras.optimizers.Optimizer object"
                )
        else:
            # if it was a string, don't touch since already serializable
            pass

        check_before_deletion = ["model_", "history", "optimizer_"]
        for attribute in check_before_deletion:
            if copy.get(attribute) is not None:
                del copy[attribute]
        return copy

    def __setstate__(self, state):
        """Magic method called during deserialization.

        Parameters
        ----------
        state : dict, as returned from __getstate__(), used for correct deserialization

        Returns
        -------
        -
        """
        from tensorflow.keras.optimizers import deserialize

        self.__dict__ = state

        if hasattr(self, "model_"):
            self.__dict__["model_"] = self.model_
            if hasattr(self, "model_.optimizer"):
                self.__dict__["optimizer_"] = self.model_.optimizer

        # if optimizer_ exists, set optimizer as optimizer_
        if self.__dict__.get("optimizer_") is not None:
            self.__dict__["optimizer"] = self.__dict__["optimizer_"]
        # else model may not have been built, but an optimizer might be passed
        else:
            # Having 0 as value implies "optimizer" attribute was None
            # as per __getstate__()
            if self.__dict__.get("optimizer") == 0:
                self.__dict__["optimizer"] = None
            elif self.__dict__.get("optimizer") == -1:
                # `optimizer` doesn't exist as a parameter alone, so delete it.
                del self.__dict__["optimizer"]
            else:
                if isinstance(self.optimizer, dict):
                    self.__dict__["optimizer"] = deserialize(self.optimizer)
                else:
                    # must have been a string already, no need to set
                    pass

        if hasattr(self, "history"):
            self.__dict__["history"] = self.history

    def save(self, path=None, serialization_format="pickle"):
        """Save serialized self to bytes-like object or to (.zip) file.

        Behaviour:
        if ``path`` is None, returns an in-memory serialized self
        if ``path`` is a file, stores the zip with that name at the location.
        The contents of the zip file are:
        _metadata - contains class of self, i.e., type(self).
        _obj - serialized self. This class uses the default serialization (pickle).
        keras/ - model, optimizer and state stored inside this directory.
        history - serialized history object.


        Parameters
        ----------
        path : None or file location (str or Path)
            if None, self is saved to an in-memory object
            if file location, self is saved to that file location. For eg:
                path="estimator" then a zip file ``estimator.zip`` will be made at cwd.
                path="/home/stored/estimator" then a zip file ``estimator.zip`` will be
                stored in ``/home/stored/``.

        serialization_format : str, default = "pickle"
            Module to use for serialization.
            The available options are present under
            ``sktime.base._base.SERIALIZATION_FORMATS``. Note that non-default formats
            might require installation of other soft dependencies.

        Returns
        -------
        if ``path`` is None - in-memory serialized self
        if ``path`` is file location - ZipFile with reference to the file
        """
        import pickle
        from pathlib import Path

        if serialization_format not in SERIALIZATION_FORMATS:
            raise ValueError(
                f"The provided `serialization_format`='{serialization_format}' "
                "is not yet supported. The possible formats are: "
                f"{SERIALIZATION_FORMATS}."
            )

        if path is not None and not isinstance(path, (str, Path)):
            raise TypeError(
                "`path` is expected to either be a string or a Path object "
                f"but found of type:{type(path)}."
            )

        if path is not None:
            path = Path(path) if isinstance(path, str) else path
            path.mkdir()

        if serialization_format == "cloudpickle":
            _check_soft_dependencies("cloudpickle", severity="error")
            import cloudpickle

            serializer = cloudpickle
        elif serialization_format == "pickle":
            serializer = pickle

        return self._serialize_using_dump_func(
            path=path,
            dump=serializer.dump,
            dumps=serializer.dumps,
        )

    def _serialize_using_dump_func(self, path, dump, dumps):
        """Serialize & return DL Estimator using ``dump`` and ``dumps`` functions."""
        import shutil
        from zipfile import ZipFile

        history = self.history.history if self.history is not None else None
        if path is None:
            _check_soft_dependencies("h5py", severity="error")
            import h5py

            in_memory_model = None
            if self.model_ is not None:
                self.model_.save("disk_less.h5")
                with h5py.File("disk_less.h5", "r") as h5file:
                    in_memory_model = h5file.id.get_file_image()

            in_memory_history = dumps(history)
            return (
                type(self),
                (
                    dumps(self),
                    in_memory_model,
                    in_memory_history,
                ),
            )

        if self.model_ is not None:
            keras_path = path / "keras" / "model.keras"
            os.makedirs(keras_path.parent, exist_ok=True)
            self.model_.save(keras_path)

        with open(path / "history", "wb") as history_writer:
            dump(history, history_writer)
        with open(path / "_metadata", "wb") as file:
            dump(type(self), file)
        with open(path / "_obj", "wb") as file:
            dump(self, file)

        shutil.make_archive(base_name=path, format="zip", root_dir=path)
        shutil.rmtree(path)
        return ZipFile(path.with_name(f"{path.stem}.zip"))

    @classmethod
    def load_from_serial(cls, serial):
        """Load object from serialized memory container.

        Parameters
        ----------
        serial: 1st element of output of ``cls.save(None)``
                This is a tuple of size 3.
                The first element represents pickle-serialized instance.
                The second element represents h5py-serialized ``keras`` model.
                The third element represent pickle-serialized history of ``.fit()``.

        Returns
        -------
        Deserialized self resulting in output ``serial``, of ``cls.save(None)``
        """
        _check_soft_dependencies("h5py")
        import pickle

        from tensorflow.keras.models import load_model

        if not isinstance(serial, tuple):
            raise TypeError(
                "`serial` is expected to be a tuple, "
                f"instead found of type: {type(serial)}"
            )
        if len(serial) != 3:
            raise ValueError(
                "`serial` should have 3 elements. "
                "All 3 elements represent in-memory serialization "
                "of the estimator. "
                f"Found a tuple of length: {len(serial)} instead."
            )

        serial, in_memory_model, in_memory_history = serial
        if in_memory_model is None:
            cls.model_ = None
        else:
            with open("diskless.h5", "wb") as store_:
                store_.write(in_memory_model)
                cls.model_ = load_model("diskless.h5")

        cls.history = pickle.loads(in_memory_history)
        return pickle.loads(serial)

    @classmethod
    def load_from_path(cls, serial):
        """Load object from file location.

        Parameters
        ----------
        serial : Name of the zip file.

        Returns
        -------
        deserialized self resulting in output at ``path``, of ``cls.save(path)``
        """
        import pickle
        from shutil import rmtree
        from zipfile import ZipFile

        from tensorflow import keras

        temp_unzip_loc = serial.parent / "temp_unzip/"
        temp_unzip_loc.mkdir()

        with ZipFile(serial, mode="r") as zip_file:
            for file in zip_file.namelist():
                if not file.startswith("keras/"):
                    continue
                zip_file.extract(file, temp_unzip_loc)

        keras_location_legacy = temp_unzip_loc / "keras"
        keras_location = temp_unzip_loc / "keras" / "model.keras"
        if keras_location.exists():
            cls.model_ = keras.models.load_model(keras_location)
        elif keras_location_legacy.exists():
            cls.model_ = keras.models.load_model(keras_location_legacy)
        else:
            cls.model_ = None

        rmtree(temp_unzip_loc)
        cls.history = keras.callbacks.History()
        with ZipFile(serial, mode="r") as file:
            cls.history.set_params(pickle.loads(file.open("history").read()))
            return pickle.loads(file.open("_obj").read())
