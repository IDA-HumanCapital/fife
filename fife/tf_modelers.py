"""FIFE modelers based on TensorFlow, which trains neural networks."""

from inspect import getfullargspec
from typing import List, Union
from warnings import warn

from fife.base_modelers import default_subset_to_all, Modeler, SurvivalModeler
from fife.nnet_survival import make_surv_array, surv_likelihood, PropHazards
import numpy as np
from random import randrange
import optuna
import pandas as pd
from matplotlib import pyplot as plt
import shap
from scipy.stats import norm
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    Concatenate,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    Input,
    Lambda,
    Layer,
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2


def binary_encode_feature(col: pd.core.series.Series) -> pd.core.frame.DataFrame:
    """Map whole numbers to bits.

    Args:
        col: a pandas Series of whole numbers.

    Returns:
        A pandas DataFrame of Boolean values, each combination of values unique
        to each unique value in the given Series.
    """
    bits = int(np.log2(max(np.max(col), 1)) + 1)
    binary_columns = pd.DataFrame()
    for i in reversed(range(bits)):
        digit_value = 2 ** i
        binary_columns[f"{col.name}_bit{i}"] = col >= digit_value
        col = col % digit_value
    return binary_columns


def split_categorical_features(
    data: pd.core.frame.DataFrame,
    categorical_features: List[str],
    numeric_features: List[str],
) -> List[Union[pd.core.series.Series, pd.core.frame.DataFrame]]:
    """Split each categorical column in a DataFrame into its own list item.

    Necessary to specify inputs to a neural network with embeddings.

    Args:
        df: A DataFrame.
        categorical_features: A list of column names to split out.
        numeric_features: A list of column names to keep in a single DataFrame.

    Returns:
        A list where each element but the last is a Series and the last element
        is a DataFrame of the given numeric features.
    """
    return [data[col].cat.codes for col in categorical_features] + [
        data[numeric_features]
    ]


def freeze_embedding_layers(model: keras.Model) -> keras.Model:
    """Prevent embedding layers of the given neural network from training."""
    new_model = Model(model.inputs, model.outputs)
    for layer in new_model.layers:
        if type(layer).__name__ == "Embedding":
            layer.trainable = False
    return new_model


def make_predictions_cumulative(model: keras.Model) -> keras.Model:
    """Append a cumulative product layer to a neural network."""
    if isinstance(model.layers[-1], CumulativeProduct):
        return model
    cumprod_layer = CumulativeProduct()(model.output)
    return Model(model.inputs, cumprod_layer)


def make_predictions_marginal(model: keras.Model) -> keras.Model:
    """Remove final layer of a neural network if a cumulative product layer."""
    if isinstance(model.layers[-1], CumulativeProduct):
        return Model(model.inputs, model.layers[-2].output)
    return model


class CumulativeProduct(Layer):
    """Transform an array into its row-wise cumulative products."""

    def call(self, inputs, **kwargs):
        """Multiply each value with all previous values in the same row."""
        return K.cumprod(inputs, axis=1)


class TFModeler(Modeler):
    """Train a neural network model using Keras with TensorFlow backend.

    Attributes:
        config (dict): User-provided configuration parameters.
        data (pd.core.frame.DataFrame): User-provided panel data.
        categorical_features (list): Column names of categorical features.
        reserved_cols (list): Column names of non-features.
        numeric_features (list): Column names of numeric features.
        n_intervals (int): The largest number of one-period intervals any
            individual is observed to survive.
        model (keras.Model): A trained neural network.
    """

    def hyperoptimize(
        self,
        n_trials: int = 64,
        subset: Union[None, pd.core.series.Series] = None,
        max_epochs: int = 128,
        hidden_activation: str = "sigmoid",
        output_activation: str = "sigmoid"
    ) -> dict:
        """Search for hyperparameters with greater out-of-sample performance.

        Args:
            n_trials: The number of hyperparameter sets to evaluate for each
                time horizon. Return None if non-positive.
            subset:  A Boolean Series that is True for observations on which
                to train and validate. If None, default to all observations not
                flagged by self.test_col or self.predict_col.

        Returns:
            A dictionary containing the best-performing parameters.
        """

        self.config["HIDDEN_ACTIVATION"] = hidden_activation
        self.config["OUTPUT_ACTIVATION"] = output_activation

        def evaluate_params(
            trial: optuna.trial.Trial,
            x_train: List[np.array],
            y_train: np.array,
            x_valid: List[np.array],
            y_valid: np.array,
            train_weights: np.array,
            valid_weights: np.array,
            max_epochs: int,
            hidden_activation: str,
            output_activation: str
        ) -> Union[None, dict]:
            """Compute out-of-sample performance for a parameter set."""
            params = {}
            params["HIDDEN_ACTIVATION"] = hidden_activation
            params["OUTPUT_ACTIVATION"] = output_activation
            params["BATCH_SIZE"] = trial.suggest_int(
                "BATCH_SIZE", min(32, x_train[0].shape[0]), x_train[0].shape[0]
            )
            params["DENSE_LAYERS"] = trial.suggest_int("DENSE_LAYERS", 0, 3)
            params["DROPOUT_SHARE"] = trial.suggest_uniform("DROPOUT_SHARE", 0, 0.5)
            params["EMBED_EXPONENT"] = trial.suggest_uniform("EMBED_EXPONENT", 0, 0.25)
            # params["EMBED_L2_REG"] = trial.suggest_loguniform("EMBED_L2_REG", 0.0001, 0.1)
            params["EMBED_L2_REG"] = trial.suggest_uniform("EMBED_L2_REG", 0, 0.2)
            # params["EMBED_L2_REG"] = trial.suggest_uniform("EMBED_L2_REG", 0, 16.0)
            if self.categorical_features:
                params["POST_FREEZE_EPOCHS"] = trial.suggest_int(
                    "POST_FREEZE_EPOCHS", 4, max_epochs
                )
            else:
                params["POST_FREEZE_EPOCHS"] = 0
            max_pre_freeze_epochs = max_epochs
            params["PRE_FREEZE_EPOCHS"] = trial.suggest_int(
                "PRE_FREEZE_EPOCHS", 4, max_epochs
            )
            params["NODES_PER_DENSE_LAYER"] = trial.suggest_int(
                "NODES_PER_DENSE_LAYER", 16, 1024
            )
            construction_args = getfullargspec(self.construct_embedding_network).args
            self.model = self.construct_embedding_network(
                **{k.lower(): v for k, v in params.items() if k.lower() in construction_args}
            )
            self.data[self.numeric_features] = self.data[self.numeric_features].fillna(
                self.config.get("NON_CAT_MISSING_VALUE", -1)
            )
            model = self.construct_embedding_network(
                **{
                    k.lower(): v
                    for k, v in self.config.items()
                    if k.lower() in construction_args
                }
            )
            model.compile(
                loss=surv_likelihood(self.n_intervals), optimizer=Adam(amsgrad=True)
            )
            for step in range(params["PRE_FREEZE_EPOCHS"]):
                model.fit(
                    x_train,
                    y_train,
                    batch_size=params["BATCH_SIZE"],
                    epochs=1,
                    sample_weight=train_weights,
                )
                validation_loss = model.evaluate(
                    x_valid, y_valid, sample_weight=valid_weights
                )
                trial.report(validation_loss, step)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            model = freeze_embedding_layers(model)
            model.compile(
                loss=surv_likelihood(self.n_intervals), optimizer=Adam(amsgrad=True)
            )
            for step in range(params["POST_FREEZE_EPOCHS"]):
                model.fit(
                    x_train,
                    y_train,
                    batch_size=params["BATCH_SIZE"],
                    epochs=1,
                    sample_weight=train_weights,
                )
                validation_loss = model.evaluate(
                    x_valid, y_valid, sample_weight=valid_weights
                )
                trial.report(validation_loss, step + max_pre_freeze_epochs)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            return validation_loss

        default_params = {"BATCH_SIZE": 512, "PRE_FREEZE_EPOCHS": 16}
        if n_trials <= 0:
            return {
                time_horizon: default_params for time_horizon in range(self.n_intervals)
            }
        params = {}
        if subset is None:
            subset = ~self.data[self.test_col] & ~self.data[self.predict_col]
        train_subset = subset & ~self.data[self.validation_col]
        valid_subset = subset & self.data[self.validation_col]
        x_train = self.format_input_data(subset=train_subset)
        x_valid = self.format_input_data(subset=valid_subset)
        y_surv = make_surv_array(
            self.data[[self.duration_col, self.max_lead_col]].min(axis=1),
            self.data[self.event_col],
            np.arange(self.n_intervals + 1),
        )
        y_train = y_surv[train_subset]
        y_valid = y_surv[valid_subset]
        train_weights = data[train_subset][self.weight_col] if self.weight_col else None
        valid_weights = data[valid_subset][self.weight_col] if self.weight_col else None
        study = optuna.create_study(
            pruner=optuna.pruners.MedianPruner(),
            sampler=optuna.samplers.TPESampler(seed=self.config.get("SEED", 9999)),
        )
        study.optimize(
            lambda trial: evaluate_params(
                trial,
                x_train,
                y_train,
                x_valid,
                y_valid,
                train_weights,
                valid_weights,
                max_epochs,
                hidden_activation,
                output_activation
            ),
            n_trials=n_trials,
        )
        params = study.best_params
        if self.categorical_features:
            default_params["POST_FREEZE_EPOCHS"] = 16
        else:
            default_params["POST_FREEZE_EPOCHS"] = 0
        construction_args = getfullargspec(self.construct_embedding_network).args
        default_model = self.construct_embedding_network(
            **{
                k.lower(): v
                for k, v in default_params.items()
                if k.lower() in construction_args
            }
        )
        default_model.compile(
            loss=surv_likelihood(self.n_intervals), optimizer=Adam(amsgrad=True)
        )
        default_model.fit(
            x_train,
            y_train,
            batch_size=default_params["BATCH_SIZE"],
            epochs=default_params["PRE_FREEZE_EPOCHS"],
            sample_weight=train_weights,
        )
        default_model = freeze_embedding_layers(default_model)
        default_model.compile(
            loss=surv_likelihood(self.n_intervals), optimizer=Adam(amsgrad=True)
        )
        default_model.fit(
            x_train,
            y_train,
            batch_size=default_params["BATCH_SIZE"],
            epochs=default_params["POST_FREEZE_EPOCHS"],
            sample_weight=train_weights,
        )
        default_validation_loss = default_model.evaluate(
            x_valid, y_valid, sample_weight=valid_weights
        )
        if default_validation_loss <= study.best_value:
            params = default_params
        return params

    def build_model(
        self, n_intervals: Union[None, int] = None, params: dict = None
    ) -> None:
        """Train and store a neural network, freezing embeddings midway."""

        tf.random.set_seed(self.config.get("SEED", 9999))


        if params is None:
            params = self.config
        if n_intervals:
            self.n_intervals = n_intervals
        else:
            self.n_intervals = self.set_n_intervals()
        self.data[self.numeric_features] = self.data[self.numeric_features].fillna(
            self.config.get("NON_CAT_MISSING_VALUE", -1)
        )
        construction_args = getfullargspec(self.construct_embedding_network).args
        self.model = self.construct_embedding_network(
            **{k.lower(): v for k, v in params.items() if k.lower() in construction_args}
        )
        pre_freeze_early_stopping = "PRE_FREEZE_EPOCHS" not in params.keys()
        post_freeze_early_stopping = "POST_FREEZE_EPOCHS" not in params.keys()
        if post_freeze_early_stopping:
            params["POST_FREEZE_EPOCHS"] = params.get("MAX_EPOCHS", 256)
        if not pre_freeze_early_stopping:
            params["MAX_EPOCHS"] = params["PRE_FREEZE_EPOCHS"]
        self.model = self.train(
            params, validation_early_stopping=pre_freeze_early_stopping
        )
        if self.categorical_features:
            self.model = freeze_embedding_layers(self.model)
            params["MAX_EPOCHS"] = params["POST_FREEZE_EPOCHS"]
            self.model = self.train(
                params, validation_early_stopping=post_freeze_early_stopping
            )
        self.model = make_predictions_cumulative(self.model)


    def construct_embedding_network(
        self,
        dense_layers: int = 2,
        nodes_per_dense_layer: int = 512,
        dropout_share: float = 0.25,
        embed_exponent: float = 0,
        embed_L2_reg: float = 2.0,
        hidden_activation: str = "sigmoid",
        output_activation: str = "sigmoid"
    ) -> keras.Model:
        """Set embedding layers followed by alternating dropout/dense layers.

        Each categorical feature passes through its own embedding layer, which
        maps whole numbers to the real line.

        Each dense layer has a sigmoid activation function. The output layer
        has one node for each lead length.

        Args:
            dense_layers: The number of dense layers in the neural network.
            nodes_per_dense_layer: The number of nodes per dense layer in the neural network.
            dropout_share: The probability of a densely connected node of the neural network being set to zero weight during training.
            embed_exponent: The ratio of the natural logarithm of the number of embedded values to the natural logarithm of the number of unique categories for each categorical feature.
            embed_L2_reg: The L2 regularization coefficient for each embedding layer.

        Returns:
            An untrained Keras model.
        """


        if self.categorical_features:
            categorical_input_layers = [
                Input(shape=(1,), dtype="int32") for _ in self.categorical_features
            ]
            embed_layers = [
                Embedding(
                    input_dim=self.data[col].nunique() + 2,
                    output_dim=int((self.data[col].nunique() + 2) ** embed_exponent),
                    embeddings_regularizer=l2(embed_L2_reg),
                )(lyr)
                for (col, lyr) in zip(
                    self.categorical_features, categorical_input_layers
                )
            ]
            flatten_layers = [Flatten()(lyr) for lyr in embed_layers]
            numeric_input_layer = Input(shape=(len(self.numeric_features),))
            concat_layer = Concatenate(axis=1)(flatten_layers + [numeric_input_layer])
            dropout_layer = Dropout(dropout_share)(concat_layer)
        else:
            categorical_input_layers = []
            numeric_input_layer = Input(shape=(len(self.numeric_features),))
            dropout_layer = Dropout(dropout_share)(numeric_input_layer)
        for _ in range(dense_layers):
            dense_layer = Dense(nodes_per_dense_layer, activation=hidden_activation)(
                dropout_layer
            )
            dropout_layer = Dropout(dropout_share)(dense_layer)
        output_layer = Dense(self.n_intervals, activation=output_activation, name="output")(
            dropout_layer
        )

        model = Model(
            inputs=categorical_input_layers + [numeric_input_layer],
            outputs=output_layer,
        )
        return model

    def format_input_data(
        self,
        data: Union[None, pd.core.frame.DataFrame] = None,
        subset: Union[None, pd.core.series.Series] = None,
    ) -> List[Union[pd.core.series.Series, pd.core.frame.DataFrame]]:
        """List each categorical feature for input to own embedding layer."""
        if data is None:
            data = self.data
        subset = default_subset_to_all(subset, data)
        return split_categorical_features(
            data[subset], self.categorical_features, self.numeric_features
        )

    def train(
        self,
        params: Union[None, dict] = None,
        subset: Union[None, pd.core.series.Series] = None,
        validation_early_stopping: bool = True,
    ) -> keras.Model:
        """Train with survival loss function of Gensheimer, Narasimhan (2019).

        See https://peerj.com/articles/6257/.

        Use the AMSGrad variant of the Adam optimizer. See Reddi, Kale, and
        Kumar (2018) at https://openreview.net/forum?id=ryQu7f-RZ.

        Train until validation set performance does not improve for the given
        number of epochs or the given maximum number of epochs.

        Returns:
            A trained Keras model.
        """
        if params is None:
            params = {}
        if subset is None:
            subset = ~self.data[self.test_col] & ~self.data[self.predict_col]
        train_subset = subset & ~self.data[self.validation_col]
        valid_subset = subset & self.data[self.validation_col]
        x_train = self.format_input_data(subset=train_subset)
        x_valid = self.format_input_data(subset=valid_subset)
        y_surv = make_surv_array(
            self.data[[self.duration_col, self.max_lead_col]].min(axis=1),
            self.data[self.event_col],
            np.arange(self.n_intervals + 1),
        )
        train_weights = data[train_subset][self.weight_col] if self.weight_col else None
        valid_weights = data[valid_subset][self.weight_col] if self.weight_col else None
        model = Model(self.model.inputs, self.model.outputs)
        model.compile(
            loss=surv_likelihood(self.n_intervals), optimizer=Adam(amsgrad=True)
        )
        callbacks = []
        if validation_early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    verbose=1,
                    patience=self.config.get("PATIENCE", 4),
                    restore_best_weights=True,
                )
            )
        model.fit(
            x_train,
            y_surv[train_subset],
            batch_size=min(
                params.get("BATCH_SIZE", self.config.get("BATCH_SIZE", 512)),
                train_subset.sum(),
            ),
            epochs=params.get("MAX_EPOCHS", self.config.get("MAX_EPOCHS", 256)),
            validation_data=(x_valid, y_surv[valid_subset], valid_weights),
            verbose=2,
            callbacks=callbacks,
            sample_weight=train_weights,
        )
        return model

    def transform_features(self) -> pd.DataFrame:
        """Transform features to suit model training."""
        data = self.data.copy(deep=True)
        numeric_data = data.drop(self.reserved_cols, axis=1).select_dtypes(
            ["number", "datetime"]
        )
        minima = np.nanmin(numeric_data, axis=0)
        maxima = np.nanmax(numeric_data, axis=0)
        data[numeric_data.columns] = (numeric_data - minima) / (maxima - minima) - 0.5
        return data

    def predict(
        self,
        subset: Union[None, pd.core.series.Series] = None,
        custom_data: Union[None, pd.core.frame.DataFrame] = None,
        cumulative: bool = True,
    ) -> np.ndarray:
        """Use trained Keras model to predict observation survival rates.

        Args:
            subset: A Boolean Series that is True for observations for which
                predictions will be produced. If None, default to all
                observations.
            custom_data: A DataFrame in the same format as the input data for
                which predictions will be produced. If None, default to the
                assigned input data.
            cumulative: If True, produce cumulative survival probabilies.
                If False, produce marginal survival probabilities (i.e., one
                minus the hazard rate).

        Returns:
            A numpy array of survival probabilities by observation and lead
            length.
        """
        if cumulative:
            model = make_predictions_cumulative(self.model)
        else:
            model = make_predictions_marginal(self.model)
        return model.predict(self.format_input_data(data=custom_data, subset=subset))

    def save_model(self, file_name: str = "FFNN_Model", path: str = "") -> None:
        """Save the TensorFlow model to disk."""
        self.model.save(path + file_name + ".h5")

    def compute_shap_values(
        self, subset: Union[None, pd.core.series.Series] = None
    ) -> dict:
        """Compute SHAP values by lead length, observation, and feature.

        SHAP values for networks with embedding layers are not supported as of
        9 Jun 2020.

        Compute SHAP values for restricted mean survival time in addition to
        each lead length.

        Args:
            subset: A Boolean Series that is True for observations for which
                the shap values will be computed. If None, default to all
                observations.

        Returns:
            A dictionary of numpy arrays, each of which contains SHAP values
            for the outcome given by its key.
        """
        print(
            "SHAP values for networks with embedding layers are not "
            "supported as of 9 Jun 2020."
        )
        return None
        model = make_predictions_marginal(self.model)
        subset = default_subset_to_all(subset, self.data)
        shap_subset = split_categorical_features(
            self.data[subset], self.categorical_features, self.numeric_features
        )
        shap_subset = [np.atleast_2d(i.values) for i in shap_subset]
        interval_shap_values = shap.DeepExplainer(
            Model(model.inputs, model.output), shap_subset
        ).shap_values(shap_subset)
        model = make_predictions_cumulative(self.model)
        sum_layer = Lambda(lambda x: K.sum(x, axis=1))(model.output)
        rmst_shap_values = shap.DeepExplainer(
            Model(model.inputs, sum_layer), shap_subset
        ).shap_values(shap_subset)
        if self.categorical_features:
            interval_shap_values = [
                np.hstack(arr_list) for arr_list in interval_shap_values
            ]
            rmst_shap_values = np.hstack(rmst_shap_values)
        shap_values = {
            str(i + 1) + "_lead": arr for i, arr in enumerate(interval_shap_values)
        }
        shap_values["RMST"] = rmst_shap_values
        return shap_values

    def compute_model_uncertainty(
            self,
            subset: Union[None, pd.core.series.Series] = None,
            n_iterations: int = 50,
            params: dict = None,
            dropout_rate: float = None,
            percent_confidence: float = 0.95,
            model_precision: float = None
    ) -> pd.DataFrame:
        """Predict with MC Dropout as proposed by Gal and Ghahramani (2015). Produces prediction intervals for future observations.
        This procedure must be used with a NN model that uses dropout.

        See https://arxiv.org/abs/1506.02142.

        Args:
            subset: A Boolean Series that is True for observations for which
                predictions will be produced. If None, default to all
                observations.
            n_iterations: Number of random dropout iterations to obtain
                predictions from.
            params: A dictionary containing custom model parameters.
            dropout_rate: A dropout rate to be used, if different than default or in params.
            percent_confidence: The percent confidence of the two-sided intervals
                defined by the computed prediction intervals.
            model_precision: The inverse of the model variance to be added to the predictive uncertainty
                estimates from MC Dropout. This parameter should be hyperoptimized.

        Returns:
            A pandas DataFrame of prediction intervals for each observation, for each future time point.
        """

        subset = default_subset_to_all(subset, self.data)
        alpha = (1 - percent_confidence) / 2
        conf = str(int(percent_confidence * 100))

        individual_identifier = self.config["INDIVIDUAL_IDENTIFIER"]
        ids_in_subset = self.data[subset][individual_identifier].unique()

        self.build_model(params = params, n_intervals = self.n_intervals)
        original_forecasts = self.forecast()


        if params is None:
            params = self.config
        if dropout_rate is not None:
            params["DROPOUT_SHARE"] = dropout_rate
        if "DROPOUT_SHARE" not in params.keys():
            warn("Model must have a dropout rate specified. Specify one using dropout_rate argument",
                 UserWarning)
            return None
        dropout_model = self


        def one_dropout_prediction(self, dropout_model, params, subset) -> pd.core.frame.DataFrame:
            """Compute forecasts for one MC Dropout iteration."""

            dropout_model.config["SEED"] = randrange(1, 9999, 1)
            dropout_model.build_model(params=params, n_intervals = self.n_intervals)
            forecasts = dropout_model.forecast()
            forecasts.columns = list(map(str, np.arange(1, len(forecasts.columns) + 1, 1)))
            keep_rows = np.repeat(True, len(forecasts))
            for rw in range(len(forecasts)):
                if forecasts.index[rw] not in ids_in_subset:
                    keep_rows[rw] = False
            forecasts = forecasts[keep_rows]
            return (forecasts)


        dropout_forecasts = list()

        # do MC dropout for n_iterations
        for i in range(n_iterations):
            one_forecast = one_dropout_prediction(self, dropout_model = dropout_model,
                                                  params=params,
                                                  subset = subset)
            dropout_forecasts.append(one_forecast)

        print("Calculating forecast uncertainty")

        ### get mean forecasts
        sum_forecasts = dropout_forecasts[0]
        for i in range(1, len(dropout_forecasts)):
            sum_forecasts = sum_forecasts + dropout_forecasts[i]

        mean_forecasts = sum_forecasts / len(dropout_forecasts)

        original_forecasts.columns = mean_forecasts.columns

        def adjust_forecasts(ensemble_forecasts):
            '''Adjusts the forecasts based on difference between predictions using default loss function and custom loss function'''

            forecast_difference = original_forecasts - mean_forecasts

            for j in range(n_iterations):
                forecasts = dropout_forecasts[j]
                forecasts_set_to_zero = forecasts <= 0
                forecasts_set_to_one = forecasts >= 1

                for i in range(len(forecasts.columns)):
                    adjust = (1 - forecasts_set_to_zero.iloc[:,i].astype("int")) * (1 - forecasts_set_to_one.iloc[:,i].astype("int"))
                    forecasts.iloc[:,i] = forecasts.iloc[:,i] + forecast_difference.iloc[:,i] * adjust
                    forecasts.iloc[:,i][forecasts.iloc[:,i] > 1] = 1
                    forecasts.iloc[:,i][forecasts.iloc[:,i] < 0] = 0

                dropout_forecasts[j] = forecasts

            return dropout_forecasts

        dropout_forecasts = adjust_forecasts(dropout_forecasts)

        mean_forecasts = original_forecasts


        mean_forecasts = original_forecasts



        def get_forecast_variance() -> pd.DataFrame:
            """ Get variance across forecasts"""

            def get_forecast_variance_for_time(time=0):

                def get_variance_for_index_time(index, time=0):
                    diff_from_mean_df = dropout_forecasts[index].iloc[:, [time]] - mean_forecasts.iloc[:, [time]]
                    squared_error_df = diff_from_mean_df ** 2
                    return squared_error_df

                squared_error_sum_df = get_variance_for_index_time(0, time=time)
                for i in range(1, len(dropout_forecasts)):
                    squared_error_sum_df = squared_error_sum_df + get_variance_for_index_time(i, time=time)
                variance_df = squared_error_sum_df / len(dropout_forecasts)
                return variance_df

            variance_all_times_df = get_forecast_variance_for_time(time=0)
            for t in range(1, len(mean_forecasts.columns)):
                variance_all_times_df = pd.concat([variance_all_times_df, get_forecast_variance_for_time(time=t)],
                                                  axis=1)

            return variance_all_times_df

        def aggregate_observation_dropout_forecasts(dropout_forecasts, rw=0) -> pd.DataFrame:
            """Get collection of forecasts from MC Dropout for one observation"""

            survival_probability = pd.Series(dtype="float32")
            period = np.empty(shape=0, dtype="int")
            iteration = np.empty(shape=0, dtype="int")

            for iter in range(len(dropout_forecasts)):
                observation_forecasts = dropout_forecasts[iter].iloc[rw]
                id = dropout_forecasts[iter].index[rw]
                survival_probability = survival_probability.append(observation_forecasts)
                keys = observation_forecasts.keys()
                period = np.append(period, observation_forecasts.keys().to_numpy(dtype="int"))
                iteration = np.append(iteration, np.repeat(iter, len(keys)))

            id = np.repeat(dropout_forecasts[0].index[rw], len(iteration))
            forecast_df = pd.DataFrame({"ID": id, "iter": iteration, "period": period,
                                        "survival_prob": survival_probability})

            return forecast_df

        def get_forecast_prediction_intervals(alpha=0.025) -> pd.DataFrame:
            """Get prediction intervals for all observations for all time points"""

            prediction_intervals_df = pd.DataFrame({
                "ID": np.empty(shape=0, dtype=type(dropout_forecasts[0].index[0])),
                "Period": np.empty(shape=0, dtype="str"),
                "LB Below Median": np.empty(shape=0, dtype="float"),
                "UB Above Median": np.empty(shape=0, dtype="float"),
                "Lower" + conf + "PercentBound": np.empty(shape=0, dtype="float"),
                "Median": np.empty(shape=0, dtype="float"),
                "Mean": np.empty(shape=0, dtype="float"),
                "Upper" + conf + "PercentBound": np.empty(shape=0, dtype="float")
            })

            for rw in range(len(mean_forecasts.index)):
                forecast_df = aggregate_observation_dropout_forecasts(dropout_forecasts, rw)
                for p in range(len(forecast_df['period'].unique())):
                    period = p + 1
                    mean_forecast = mean_forecasts.iloc[rw, p]
                    period_forecasts = forecast_df[forecast_df['period'] == period]['survival_prob']
                    pred_variance = period_forecasts.var()
                    if model_precision is not None:
                        forecast_pred_variance = pred_variance + 1 / model_precision
                    else:
                        forecast_pred_variance = pred_variance
                    forecast_pred_sd = np.sqrt(forecast_pred_variance)
                    forecast_lb = np.max([0, mean_forecast + forecast_pred_sd * norm.ppf(alpha)])
                    forecast_ub = np.min([1, mean_forecast - forecast_pred_sd * norm.ppf(alpha)])
                    forecast_median = period_forecasts.quantile([0.5])
                    forecast_quantiles = period_forecasts.quantile([alpha, 0.5, 1 - alpha])

                    if model_precision is not None:
                        forecast_quantiles.iloc[0] = forecast_median + (forecast_quantiles.iloc[0] - forecast_median) * np.sqrt(forecast_pred_variance / pred_variance)
                        forecast_quantiles.iloc[2] = forecast_median + (forecast_quantiles.iloc[2] - forecast_median) * np.sqrt(forecast_pred_variance / pred_variance)

                    df = pd.DataFrame(
                        {"ID": forecast_df["ID"][0],
                         "Period": [period],
                         "LB Below Median": forecast_quantiles.iloc[0] - forecast_median,
                         "UB Above Median": forecast_quantiles.iloc[2] - forecast_median,
                         "Lower" + conf + "PercentBound": forecast_quantiles.iloc[0],
                         # "Lower" + conf + "PercentBound": forecast_lb,
                         "Median": forecast_median.iloc[0],
                         "Mean": mean_forecast,
                         # "Upper" + conf + "PercentBound": forecast_ub
                         "Upper" + conf + "PercentBound": forecast_quantiles.iloc[2]
                         }
                    )
                    prediction_intervals_df = prediction_intervals_df.append(df)
            return prediction_intervals_df

        prediction_intervals_df = get_forecast_prediction_intervals(alpha)


        def get_aggregation_prediction_intervals() -> pd.DataFrame:
            """Get prediction intervals for the expected counts for each time period."""

            aggregation_pi_df = pd.DataFrame({
                "ID": np.empty(shape=0, dtype=type(dropout_forecasts[0].index[0])),
                "Period": np.empty(shape=0, dtype="str"),
                "LB Below Median": np.empty(shape=0, dtype="float"),
                "UB Above Median": np.empty(shape=0, dtype="float"),
                "Lower" + conf + "PercentBound": np.empty(shape=0, dtype="float"),
                "Median": np.empty(shape=0, dtype="float"),
                "Mean": np.empty(shape=0, dtype="float"),
                "Upper" + conf + "PercentBound": np.empty(shape=0, dtype="float")
            })

            for p in range(len(dropout_forecasts[0].columns.unique())):
                period = p + 1
                dropout_expected_counts = []
                for d in dropout_forecasts:
                    dropout_expected_counts.append(d[d.columns[p]].sum())
                forecast_sum_series = pd.Series(dropout_expected_counts)
                forecast_sum_quantiles = forecast_sum_series.quantile([alpha, 0.5, 1 - alpha])
                df = pd.DataFrame(
                    {"ID": "Sum",
                     "Period": [period],
                     "LB Below Median": forecast_sum_quantiles.iloc[0] - forecast_sum_quantiles.iloc[1],
                     "UB Above Median": forecast_sum_quantiles.iloc[2] - forecast_sum_quantiles.iloc[1],
                     "Lower" + conf + "PercentBound": forecast_sum_quantiles.iloc[0],
                     "Median": forecast_sum_quantiles.iloc[1],
                     "Mean": forecast_sum_series.mean(),
                     "Upper" + conf + "PercentBound": forecast_sum_quantiles.iloc[2]
                     }
                )
                aggregation_pi_df = aggregation_pi_df.append(df)
            return aggregation_pi_df

        aggregation_prediction_intervals_df = get_aggregation_prediction_intervals()
        prediction_intervals_df = aggregation_prediction_intervals_df.append(prediction_intervals_df)
        prediction_intervals_df.index = np.arange(0, len(prediction_intervals_df.index))
        return prediction_intervals_df


    def plot_forecast_prediction_intervals(self, pi_df: pd.DataFrame, ID: str = "Sum") -> None:
        """Plot forecasts for future time periods with prediction intervals calculated by MC Dropout using compute_model_uncertainty()

        Args:
            pi_df: A DataFrame of prediction intervals for forecasts, generated by compute_model_uncertainty()
            ID: The ID of the observation for which to have forecasts plotted, or "Sum" for aggregated counts.
        """

        ID = str(ID)
        forecast_df = pi_df[pi_df['ID'] == ID]
        if len(forecast_df) == 0:
            try:
                ID = int(ID)
                forecast_df = pi_df[pi_df['ID'] == int(ID)]
            except ValueError:
                ID = ID
                warn("Invalid ID.", UserWarning)
                return None
        forecast_df = forecast_df.assign(Period=forecast_df['Period'].tolist())
        forecast_df = forecast_df.rename({forecast_df.columns[4]: "Lower", forecast_df.columns[7]: "Upper"}, axis = "columns")
        plt.clf()
        plt.scatter('Period', 'Mean', data=forecast_df, color="black")
        plt.fill_between(x='Period', y1='Lower', y2='Upper',
                         data=forecast_df, color='gray', alpha=0.3,
                         linewidth=0)
        plt.xlabel("Period")
        if ID == "Sum":
            plt.ylabel("Expected counts")
        else:
            plt.ylabel("Survival probability")
        plt.title("Forecasts with Prediction Intervals")
        plt.show()
        return None


class TFSurvivalModeler(TFModeler, SurvivalModeler):
    """Use TensorFlow to forecast probabilities of being observed in future periods."""

    pass


class FeedforwardNeuralNetworkModeler(TFSurvivalModeler):
    """Deprecated alias for TFSurvivalModeler"""

    warn(
        "The name 'FeedforwardNeuralNetworkModeler' is deprecated. "
        "Please use 'TFSurvivalModeler' instead.",
        DeprecationWarning,
    )


FeedForwardNeuralNetworkModeler = FeedforwardNeuralNetworkModeler


class ProportionalHazardsModeler(TFSurvivalModeler):
    """Train a proportional hazards model with embeddings for categorical features using Keras."""

    def construct_embedding_network(self) -> keras.Model:
        """Set embedding layers that feed into a single node.

        Each categorical feature passes through its own embedding layer, which
        maps whole numbers to the real line.

        The embedded values and numeric features feed directly into a single node. The
        single node feeds into a proportional hazards layer.

        Returns:
            An untrained Keras model.
        """
        if self.categorical_features:
            categorical_input_layers = [
                Input(shape=(1,), dtype="int32") for _ in self.categorical_features
            ]
            embed_layers = [
                Embedding(
                    input_dim=self.data[col].max() + 2,
                    output_dim=int(
                        (self.data[col].nunique() + 2)
                        ** self.config.get("EMBED_EXPONENT", 0)
                    ),
                    embeddings_regularizer=l2(self.config.get("EMBED_L2_REG", 2.0)),
                )(lyr)
                for (col, lyr) in zip(
                    self.categorical_features, categorical_input_layers
                )
            ]
            flatten_layers = [Flatten()(lyr) for lyr in embed_layers]
            numeric_input_layer = Input(shape=(len(self.numeric_features),))
            concat_layer = Concatenate(axis=1)(flatten_layers + [numeric_input_layer])
            dense_layer = Dense(1, use_bias=0, kernel_initializer="zeros")(concat_layer)
        else:
            categorical_input_layers = []
            numeric_input_layer = Input(shape=(len(self.numeric_features),))
            dense_layer = Dense(1, use_bias=0, kernel_initializer="zeros")(
                numeric_input_layer
            )
        output_layer = PropHazards(self.n_intervals)(dense_layer)
        model = Model(
            inputs=categorical_input_layers + [numeric_input_layer],
            outputs=output_layer,
        )
        return model

    def save_model(self, file_name: str = "PH_Model", path: str = "") -> None:
        """Save the TensorFlow model to disk."""
        self.model.save(path + file_name + ".h5")

    def hyperoptimize(self, **kwargs) -> dict:
        """Returns None for ProportionalHazardsModeler, which does not have hyperparameters"""
        warn(
            "Warning: ProportionalHazardsModeler does not have hyperparameters to optimize."
        )
        return None


class ProportionalHazardsEncodingModeler(TFSurvivalModeler):
    """Train a proportional hazards model with binary-encoded categorical features using Keras."""

    def build_model(self, n_intervals: Union[None, int] = None) -> None:
        """Train and store a neural network with a proportional hazards restriction."""
        tf.random.set_seed(self.config.get("SEED", 9999))
        if n_intervals:
            self.n_intervals = n_intervals
        else:
            self.n_intervals = self.set_n_intervals()
        for col in self.categorical_features:
            binary_encodings = binary_encode_feature(self.data[col])
            del self.data[col]
            self.data[binary_encodings.columns] = binary_encodings
            del binary_encodings
        self.data = self.data.fillna(self.config.get("NON_CAT_MISSING_VALUE", -1))
        self.model = self.construct_network()
        self.model = self.train()
        self.model = make_predictions_cumulative(self.model)

    def format_input_data(
        self,
        data: Union[None, pd.core.frame.DataFrame] = None,
        subset: Union[None, pd.core.series.Series] = None,
    ) -> List[Union[pd.core.series.Series, pd.core.frame.DataFrame]]:
        """Keep only the features and observations desired for model input."""
        if data is None:
            data = self.data
        subset = default_subset_to_all(subset, data)
        formatted_data = data.drop(self.reserved_cols, axis=1)[subset]
        for col in self.categorical_features:
            formatted_data[col] = formatted_data[col].cat.codes
        return formatted_data

    def construct_network(self) -> keras.Model:
        """Set all features to feed directly into a single node.

        The single node feeds into a proportional hazards layer.

        Returns:
            An untrained Keras model.
        """
        input_layer = Input(
            shape=(len(self.data.drop(self.reserved_cols, axis=1).columns),)
        )
        dense_layer = Dense(1, use_bias=0, kernel_initializer="zeros")(input_layer)
        output_layer = PropHazards(self.n_intervals)(dense_layer)
        model = Model(inputs=input_layer, outputs=output_layer)
        return model

    def save_model(self, file_name: str = "PH_Encoded_Model", path: str = "") -> None:
        """Save the TensorFlow model to disk."""
        self.model.save(path + file_name + ".h5")

    def hyperoptimize(self, **kwargs) -> dict:
        """Returns None for ProportionalHazardsEncodingModeler, which does not have hyperparameters"""
        warn(
            "Warning: ProportionalHazardsEncodingModeler does not have hyperparameters to optimize."
        )
        return None
