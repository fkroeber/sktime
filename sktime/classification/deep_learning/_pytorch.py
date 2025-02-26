"""Abstract base class for the Pytorch neural network classifiers."""

__author__ = ["geetu040"]


__all__ = ["BaseDeepClassifierPytorch"]

import abc

import numpy as np
import pytorch_lightning as pl

from sklearn.preprocessing import LabelEncoder
from sktime.classification.base import BaseClassifier
from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    import torch
    from torch.utils.data import DataLoader, Dataset

    OPTIMIZERS = {
        "Adadelta": torch.optim.Adadelta,
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
        "SGD": torch.optim.SGD,
        "Adagrad": torch.optim.Adagrad,
    }
else:

    class Dataset:
        """Dummy class if torch is unavailable."""


class BaseDeepClassifierPytorch(BaseClassifier):
    """Abstract base class for the Pytorch neural network classifiers."""

    _tags = {
        "authors": ["geetu040"],
        "maintainers": ["geetu040"],
        "python_dependencies": ["torch"],
        "X_inner_mtype": "numpy3D",
        "y_inner_mtype": "numpy1D",
        "capability:multivariate": True,
        "capability:multioutput": False,
    }

    def __init__(
        self,
        num_epochs=16,
        batch_size=8,
        criterion=None,
        criterion_kwargs=None,
        optimizer=None,
        optimizer_kwargs=None,
        lr=0.001,
        lr_scheduler=None,
        lr_scheduler_kwargs=None,
        callbacks=None,
        verbose=True,
        random_state=None,
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.criterion = criterion
        self.criterion_kwargs = criterion_kwargs
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.callbacks = callbacks if callbacks else []
        self.verbose = verbose
        self.random_state = random_state

        if self.random_state is not None:
            if _check_soft_dependencies("torch", severity="none"):
                torch.manual_seed(self.random_state)

        # use this when y has str
        self.label_encoder = None
        super().__init__()

        # instantiate optimizers
        self.optimizers = OPTIMIZERS

    def _build_lightning_module(self):

        class LitNetwork(pl.LightningModule):
            def __init__(
                self, network, criterion, optimizer, lr_scheduler, lr_scheduler_kwargs
            ):
                super().__init__()
                self.network = network
                self.criterion = criterion
                self.optimizer = optimizer
                self.lr_scheduler = lr_scheduler
                self.lr_scheduler_kwargs = lr_scheduler_kwargs

            def forward(self, x):
                return self.network(x)

            def training_step(self, batch, batch_idx):
                inputs, targets = batch
                outputs = self.network(**inputs)
                loss = self.criterion(outputs, targets)
                self.log(
                    "train_loss", loss, prog_bar=True, on_step=False, on_epoch=True
                )
                return loss

            def validation_step(self, batch, batch_idx):
                inputs, targets = batch
                outputs = self.network(**inputs)
                loss = self.criterion(outputs, targets)
                self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

            def on_train_end(self):
                # Look for ModelCheckpoint in callbacks
                for callback in self.trainer.callbacks:
                    if isinstance(callback, pl.callbacks.ModelCheckpoint):
                        # Load the best model weights
                        if callback.best_model_path:
                            print(
                                f"Restoring best model from: {callback.best_model_path}"
                            )
                            self.load_state_dict(
                                torch.load(callback.best_model_path)["state_dict"]
                            )
                        break

            def configure_optimizers(self):
                optim_dict = {}
                optim_dict["optimizer"] = self.optimizer
                if self.lr_scheduler:
                    interval = self.lr_scheduler_kwargs.pop("interval", "epoch")
                    monitor = self.lr_scheduler_kwargs.pop("monitor", "val_loss")
                    frequency = self.lr_scheduler_kwargs.pop("frequency", 1)
                    optim_dict["lr_scheduler"] = {
                        "scheduler": self.lr_scheduler(
                            self.optimizer, **self.lr_scheduler_kwargs
                        ),
                        "interval": interval,
                        "monitor": monitor,
                        "frequency": frequency,
                    }
                return optim_dict

        return LitNetwork(
            self.network,
            self._instantiate_criterion(),
            self._instantiate_optimizer(),
            self.lr_scheduler,
            self.lr_scheduler_kwargs,
        )

    def _fit(self, X, y, X_val=None, y_val=None):
        # encode y
        y = self._encode_y(y)
        if y_val is not None:
            y_val = self._encode_y(y_val)

        # build dataloaders
        train_dataloader = self._build_dataloader(X, y)
        val_dataloader = (
            self._build_dataloader(X_val, y_val) if X_val is not None else None
        )

        # instantiate the torch network & lightning module
        self.network = self._build_network(X, y)
        self.model = self._build_lightning_module()

        # initialize the trainer
        trainer = pl.Trainer(
            max_epochs=self.num_epochs,
            callbacks=self.callbacks,
            enable_progress_bar=self.verbose,
        )
        trainer.fit(
            self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

    def _instantiate_optimizer(self):
        if self.optimizer:
            if self.optimizer in self.optimizers.keys():
                if self.optimizer_kwargs:
                    return self.optimizers[self.optimizer](
                        self.network.parameters(), lr=self.lr, **self.optimizer_kwargs
                    )
                else:
                    return self.optimizers[self.optimizer](
                        self.network.parameters(), lr=self.lr
                    )
            else:
                raise TypeError(
                    f"Please pass one of {self.optimizers.keys()} for `optimizer`."
                )
        else:
            # default optimizer
            return torch.optim.Adam(self.network.parameters(), lr=self.lr)

    def _instantiate_criterion(self):
        if self.criterion:
            if self.criterion in self.criterions.keys():
                if self.criterion_kwargs:
                    return self.criterions[self.criterion](**self.criterion_kwargs)
                else:
                    return self.criterions[self.criterion]()
            else:
                raise TypeError(
                    f"Please pass one of {self.criterions.keys()} for `criterion`."
                )
        else:
            # default criterion
            return torch.nn.CrossEntropyLoss()

    @abc.abstractmethod
    def _build_network(self):
        pass

    def _build_dataloader(self, X, y=None):
        # default behaviour if estimator doesnot implement
        # dataloader of its own
        dataset = PytorchDataset(X, y)
        return DataLoader(dataset, self.batch_size)

    def _predict(self, X):
        """Predict labels for sequences in X.

        private _predict containing the core logic, called from predict

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            if self.get_tag("X_inner_mtype") = "numpy3D":
            3D np.ndarray of shape = [n_instances, n_dimensions, series_length]
            if self.get_tag("X_inner_mtype") = "pd-multiindex:":
            pd.DataFrame with columns = variables,
            index = pd.MultiIndex with first level = instance indices,
            second level = time indices
            for list of other mtypes, see datatypes.SCITYPE_REGISTER
            for specifications, see examples/AA_datatypes_and_datasets.ipynb

        Returns
        -------
        y : should be of mtype in self.get_tag("y_inner_mtype")
            1D iterable, of shape [n_instances]
            or 2D iterable, of shape [n_instances, n_dimensions]
            predicted class labels
            indices correspond to instance indices in X
            if self.get_tag("capaility:multioutput") = False, should be 1D
            if self.get_tag("capaility:multioutput") = True, should be 2D
        """
        y_pred_prob = self._predict_proba(X)
        y_pred = np.argmax(y_pred_prob, axis=-1)
        y_decoded = self._decode_y(y_pred)
        return y_decoded

    def _predict_proba(self, X):
        """Predicts labels probabilities for sequences in X.

        private _predict_proba containing the core logic, called from predict_proba

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            if self.get_tag("X_inner_mtype") = "numpy3D":
            3D np.ndarray of shape = [n_instances, n_dimensions, series_length]
            if self.get_tag("X_inner_mtype") = "pd-multiindex:":
            pd.DataFrame with columns = variables,
            index = pd.MultiIndex with first level = instance indices,
            second level = time indices
            for list of other mtypes, see datatypes.SCITYPE_REGISTER
            for specifications, see examples/AA_datatypes_and_datasets.ipynb

        Returns
        -------
        y : 2D array of shape [n_instances, n_classes] - predicted class probabilities
            1st dimension indices correspond to instance indices in X
            2nd dimension indices correspond to possible labels (integers)
            (i, j)-th entry is predictive probability that i-th instance is of class j
        """
        import torch.nn.functional as F
        from torch import cat

        self.network.eval()
        dataloader = self._build_dataloader(X)
        y_pred = []
        for inputs in dataloader:
            y_pred.append(self.network(**inputs).detach())
        y_pred = cat(y_pred, dim=0)
        # (batch_size, num_outputs)
        y_pred = F.softmax(y_pred, dim=-1)
        y_pred = y_pred.numpy()
        return y_pred

    def _encode_y(self, y):
        unique = np.unique(y)
        if np.array_equal(unique, np.arange(len(unique))):
            return y

        self.label_encoder = LabelEncoder()
        return self.label_encoder.fit_transform(y)

    def _decode_y(self, y):
        if self.label_encoder is None:
            return y

        return self.label_encoder.inverse_transform(y)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            Reserved values for classifiers:
                "results_comparison" - used for identity testing in some classifiers
                    should contain parameter settings comparable to "TSC bakeoff"

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        return []


class PytorchDataset(Dataset):
    """Dataset for use in sktime deep learning classifier based on pytorch."""

    def __init__(self, X, y=None):
        # X.shape = (batch_size, n_dims, n_timestamps)
        X = np.transpose(X, (0, 2, 1))
        # X.shape = (batch_size, n_timestamps, n_dims)

        self.X = X
        self.y = y

    def __len__(self):
        """Get length of dataset."""
        return len(self.X)

    def __getitem__(self, i):
        """Get item at index."""
        x = self.X[i]
        x = torch.tensor(x, dtype=torch.float)
        inputs = {"X": x}
        # to make it reusable for predict
        if self.y is None:
            return inputs

        # return y during fit
        y = self.y[i]
        y = torch.tensor(y, dtype=torch.long)
        return inputs, y
