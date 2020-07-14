"""FIFE modelers based on TensorFlow, which trains neural networks."""

from typing import List, Union

from fife import survival_modeler
from fife.nnet_survival import make_surv_array, surv_likelihood, PropHazards
import numpy as np
import pandas as pd
import shap
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
    bits = int(np.log2(np.max(col)) + 1)
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
    return [data[col] for col in categorical_features] + [data[numeric_features]]


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


class FeedforwardNeuralNetworkModeler(survival_modeler.SurvivalModeler):
    """Train a neural network model using Keras.

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

    def build_model(self, n_intervals: Union[None, int] = None) -> None:
        """Train and store a neural network, freezing embeddings midway."""
        tf.random.set_seed(self.config.get("SEED", 9999))
        if n_intervals:
            self.n_intervals = n_intervals
        else:
            self.n_intervals = self.set_n_intervals()
        self.data[self.numeric_features] = self.data[self.numeric_features].fillna(
            self.config.get("NON_CAT_MISSING_VALUE", -1)
        )
        self.model = self.construct_embedding_network()
        self.model = self.train()
        if self.categorical_features:
            self.model = freeze_embedding_layers(self.model)
            self.model = self.train()
        self.model = make_predictions_cumulative(self.model)

    def construct_embedding_network(self) -> keras.Model:
        """Set embedding layers followed by alternating dropout/dense layers.

        Each categorical feature passes through its own embedding layer, which
        maps whole numbers to the real line.

        Each dense layer has a sigmoid activation function. The output layer
        has one node for each lead length.

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
            dropout_layer = Dropout(self.config.get("DROPOUT_SHARE", 0.25))(
                concat_layer
            )
        else:
            categorical_input_layers = []
            numeric_input_layer = Input(shape=(len(self.numeric_features),))
            dropout_layer = Dropout(self.config.get("DROPOUT_SHARE", 0.25))(
                numeric_input_layer
            )
        for _ in range(self.config.get("DENSE_LAYERS", 2)):
            dense_layer = Dense(
                self.config.get("NODES_PER_DENSE_LAYER", 512), activation="sigmoid"
            )(dropout_layer)
            dropout_layer = Dropout(self.config.get("DROPOUT_SHARE", 0.25))(dense_layer)
        output_layer = Dense(self.n_intervals, activation="sigmoid", name="output")(
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
        subset = survival_modeler.default_subset_to_all(subset, data)
        return split_categorical_features(
            data[subset], self.categorical_features, self.numeric_features
        )

    def train(self) -> keras.Model:
        """Train with survival loss function of Gensheimer, Narasimhan (2019).

        See https://peerj.com/articles/6257/.

        Use the AMSGrad variant of the Adam optimizer. See Reddi, Kale, and
        Kumar (2018) at https://openreview.net/forum?id=ryQu7f-RZ.

        Train until validation set performance does not improve for the given
        number of epochs or the given maximum number of epochs.

        Returns:
            A trained Keras model.
        """
        train_subset = (
            ~self.data[self.validation_col]
            & ~self.data[self.test_col]
            & ~self.data[self.predict_col]
        )
        valid_subset = (
            self.data[self.validation_col]
            & ~self.data[self.test_col]
            & ~self.data[self.predict_col]
        )
        x_train = self.format_input_data(subset=train_subset)
        x_valid = self.format_input_data(subset=valid_subset)
        y_surv = make_surv_array(
            self.data[[self.duration_col, self.max_lead_col]].min(axis=1),
            self.data[self.event_col],
            np.arange(self.n_intervals + 1),
        )
        model = Model(self.model.inputs, self.model.outputs)
        model.compile(
            loss=surv_likelihood(self.n_intervals), optimizer=Adam(amsgrad=True)
        )
        model.fit(
            x_train,
            y_surv[train_subset],
            batch_size=min(self.config.get("BATCH_SIZE", 512), train_subset.sum()),
            epochs=self.config.get("MAX_EPOCHS", 256),
            validation_data=(x_valid, y_surv[valid_subset]),
            verbose=2,
            callbacks=[
                EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    verbose=1,
                    patience=self.config.get("PATIENCE", 4),
                    restore_best_weights=True,
                )
            ],
        )
        return model

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
        subset = survival_modeler.default_subset_to_all(subset, self.data)
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
        self, subset: Union[None, pd.core.series.Series] = None, n_iterations: int = 200
    ) -> np.ndarray:
        """Predict with dropout as proposed by Gal and Ghahramani (2015).

        See https://arxiv.org/abs/1506.02142.

        Args:
            subset: A Boolean Series that is True for observations for which
                predictions will be produced. If None, default to all
                observations.
            n_iterations: Number of random dropout specifications to obtain
                predictions from.

        Returns:
            A numpy array of predictions by observation, lead length, and
            iteration.
        """
        subset = survival_modeler.default_subset_to_all(subset, self.data)
        model_inputs = split_categorical_features(
            self.data[subset], self.categorical_features, self.numeric_features
        ) + [1.0]
        predict_with_dropout = K.function(
            self.model.inputs + [K.learning_phase()], self.model.outputs
        )
        predictions = np.dstack(
            [predict_with_dropout(model_inputs)[0] for i in range(n_iterations)]
        )
        return predictions


class ProportionalHazardsModeler(FeedforwardNeuralNetworkModeler):
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


class ProportionalHazardsEncodingModeler(FeedforwardNeuralNetworkModeler):
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
        subset = survival_modeler.default_subset_to_all(subset, data)
        return data.drop(self.reserved_cols, axis=1)[subset]

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
