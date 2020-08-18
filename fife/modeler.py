"""The abstract class on which all FIFE modelers are based."""

from abc import ABC, abstractmethod
from typing import Any, Union

import numpy as np
import pandas as pd


def default_subset_to_all(
    subset: Union[None, pd.core.series.Series], data: pd.core.frame.DataFrame
) -> pd.core.series.Series:
    """Map an unspecified subset to an entirely True boolean mask."""
    if subset is None:
        return pd.Series([True] * data.shape[0], index=data.index)
    return subset


class Modeler(ABC):
    """Set template for modelers that use panel data to produce forecasts.

    Attributes:
        config (dict): User-provided configuration parameters.
        data (pd.core.frame.DataFrame): User-provided panel data.
        categorical_features (list): Column names of categorical features.
        duration_col (str): Name of the column representing the number of
            future periods observed for the given individual.
        event_col (str): Name of the column indicating whether the individual
            is observed to exit the dataset.
        predict_col (str): Name of the column indicating whether the
            observation will be used for prediction after training.
        test_col (str): Name of the column indicating whether the observation
            will be used for testing model performance after training.
        validation_col (str): Name of the column indicating whether the
            observation will be used for evaluating model performance during
            training.
        period_col (str): Name of the column representing the number of
            periods since the earliest period in the data.
        max_lead_col (str): Name of the column representing the number of
            observable future periods.
        reserved_cols (list): Column names of non-features.
        numeric_features (list): Column names of numeric features.
        n_intervals (int): The largest number of periods ahead to forecast
    """

    def __init__(
        self,
        config: Union[None, dict] = {},
        data: Union[None, pd.core.frame.DataFrame] = None,
        duration_col: str = "_duration",
        event_col: str = "_event_observed",
        predict_col: str = "_predict_obs",
        test_col: str = "_test",
        validation_col: str = "_validation",
        period_col: str = "_period",
        max_lead_col: str = "_maximum_lead",
    ) -> None:
        """Characterize data for modelling.

        Identify each column as categorical, reserved, and numeric.
        Compute the maximum lead length for modelling.

        Args:
            config: User-provided configuration parameters.
            data: User-provided panel data.
            duration_col: Name of the column representing the number of future
                periods observed for the given individual.
            event_col: Name of the column indicating whether the individual is
                observed to exit the dataset.
            predict_col: Name of the column indicating whether the observation
                will be used for prediction after training.
            test_col: Name of the column indicating whether the observation
                will be used for testing model performance after training.
            validation_col: Name of the column indicating whether the
                observation will be used for evaluating model performance
                during training.
            period_col (str): Name of the column representing the number of
                periods since the earliest period in the data.
            max_lead_col (str): Name of the column representing the number of
                observable future periods.
        """
        if (config.get("TIME_IDENTIFIER", "") == "") and data is not None:
            config["TIME_IDENTIFIER"] = data.columns[1]
            print(
                "Time identifier column name not given; assumed to be "
                f'second-leftmost column ({config["TIME_IDENTIFIER"]})'
            )
        if (config.get("INDIVIDUAL_IDENTIFIER", "") == "") and data is not None:
            config["INDIVIDUAL_IDENTIFIER"] = data.columns[0]
            print(
                "Individual identifier column name not given; assumed to be "
                f'leftmost column ({config["INDIVIDUAL_IDENTIFIER"]})'
            )
        self.config = config
        self.data = data
        self.duration_col = duration_col
        self.event_col = event_col
        self.predict_col = predict_col
        self.test_col = test_col
        self.validation_col = validation_col
        self.period_col = period_col
        self.max_lead_col = max_lead_col
        self.reserved_cols = [
            self.duration_col,
            self.event_col,
            self.predict_col,
            self.test_col,
            self.validation_col,
            self.period_col,
            self.max_lead_col,
        ]
        if self.config:
            self.reserved_cols.append(self.config["INDIVIDUAL_IDENTIFIER"])
        if self.data is not None:
            self.categorical_features = [
                col for col in self.data.select_dtypes("category")
            ]
            self.numeric_features = [
                feature
                for feature in self.data
                if feature not in (self.categorical_features + self.reserved_cols)
            ]
            self.data = self.transform_features()

    @abstractmethod
    def train(self) -> Any:
        """Train and return a model."""

    @abstractmethod
    def predict(
        self, subset: Union[None, pd.core.series.Series] = None, cumulative: bool = True
    ) -> np.ndarray:
        """Use trained model to produce observation survival probabilities."""

    @abstractmethod
    def evaluate(
        self,
        subset: Union[None, pd.core.series.Series] = None,
        threshold_positive: Union[None, str, float] = 0.5,
        share_positive: Union[None, str, float] = None,
    ) -> pd.core.frame.DataFrame:
        """Tabulate model performance metrics."""

    @abstractmethod
    def forecast(self) -> pd.core.frame.DataFrame:
        """Tabulate survival probabilities for most recent observations."""

    def transform_features(self):
        """Transform datetime features to suit model training."""
        return self.data

    def build_model(self, n_intervals: Union[None, int] = None) -> None:
        """Configure, train, and store a model."""
        if n_intervals:
            self.n_intervals = n_intervals
        else:
            self.n_intervals = self.set_n_intervals()
        self.model = self.train()

    def set_n_intervals(self) -> int:
        """Determine the maximum periods ahead the model will predict."""
        train_durations = self.data[[self.duration_col, self.max_lead_col]].min(axis=1)
        train_obs_by_lead_length = train_durations[
            (
                ~self.data[self.validation_col]
                & ~self.data[self.test_col]
                & ~self.data[self.predict_col]
            )
        ].value_counts()
        n_intervals = train_obs_by_lead_length[
            train_obs_by_lead_length > self.config.get("MIN_SURVIVORS_IN_TRAIN", 64)
        ].index.max()
        return n_intervals

    def hyperoptimize(
        self,
        n_trials: int = 64,
        rolling_validation: bool = True,
        train_subset: Union[None, pd.core.series.Series] = None,
    ) -> dict:
        """Search for hyperparameters with greater out-of-sample performance."""

    def compute_shap_values(
        self, subset: Union[None, pd.core.series.Series] = None
    ) -> dict:
        """Compute SHAP values by lead length, observation, and feature."""

    def compute_model_uncertainty(
        self, subset: Union[None, pd.core.series.Series] = None, n_iterations: int = 200
    ) -> np.ndarray:
        """Produce predictions of models modified after training."""
