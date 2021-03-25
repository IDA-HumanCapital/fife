"""FIFE modeler based on Pandas, which tabulates interacted fixed effects."""

from typing import Union
from warnings import warn

from fife.base_modelers import (
    default_subset_to_all,
    Modeler,
    SurvivalModeler,
    StateModeler,
    ExitModeler,
)
import numpy as np
import pandas as pd


class IFEModeler(Modeler):
    """Predict with mean of training observations with same values.

    Attributes:
        config (dict): User-provided configuration parameters.
        data (pd.core.frame.DataFrame): User-provided panel data.
        categorical_features (list): Column names of categorical features.
        reserved_cols (list): Column names of non-features.
        numeric_features (list): Column names of numeric features.
        n_intervals (int): The largest number of one-period intervals any
            individual is observed to survive.
        model (pd.core.frame.DataFrame): Survival rates for each combination of
            categorical values in the training data.
    """

    def train(self) -> pd.core.frame.DataFrame:
        """Compute the mean of the outcome for each combination of categorical values."""
        cell_rates = pd.DataFrame()
        train_subset = (
            ~self.data[self.validation_col]
            & ~self.data[self.test_col]
            & ~self.data[self.predict_col]
        )
        for time_horizon in range(self.n_intervals):
            data = self.label_data(time_horizon)
            data = data[train_subset]
            data = self.subset_for_training_horizon(data, time_horizon)
            grouping = self.categorical_features
            if self.objective == "multiclass":
                if self.weight_col:
                    grouped_sums = data.groupby(self.categorical_features + ["_label"])[
                        self.weight_col
                    ].sum()
                    cell_rates[
                        time_horizon + 1
                    ] = grouped_sums / grouped_sums.reset_index().groupby(
                        self.categorical_features
                    )[
                        self.weight_col
                    ].transform(
                        "sum"
                    )
                else:
                    grouped_sums = data.groupby(self.categorical_features + ["_label"])[
                        "_label"
                    ].count()
                    cell_rates[time_horizon + 1] = (
                        grouped_sums
                        / data.groupby(self.categorical_features)["_label"].count()
                    )
            else:
                if self.weight_col:
                    data["_label"] = data["_label"] * data[self.weight_col]
                    grouped_sums = data.groupby(self.categorical_features)[
                        "_label", self.weight_col
                    ].sum()
                    cell_rates[time_horizon + 1] = (
                        grouped_sums["_label"] / grouped_sums[self.weight_col]
                    )
                else:
                    cell_rates[time_horizon + 1] = data.groupby(
                        self.categorical_features
                    )["_label"].mean()
        return cell_rates

    def predict(
        self, subset: Union[None, pd.core.series.Series] = None, cumulative: bool = True
    ) -> np.ndarray:
        """Map observations to outcome means from their categorical values.

        Map observations with a combination of categorical values not seen in
        the training data to the mean of the outcome in the training set.

        Args:
            subset: A Boolean Series that is True for observations for which
                predictions will be produced. If None, default to all
                observations.
            cumulative: If True, will produce cumulative survival
                probabilities. If False, will produce marginal survival
                probabilities (i.e., one minus the hazard rate).

        Returns:
            A numpy array of outcome means by observation and lead
            length.
        """
        subset = default_subset_to_all(subset, self.data)
        predictions = self.data[subset].merge(
            self.model, how="left", on=self.categorical_features
        )
        predictions = predictions[self.model.columns]
        predictions = predictions.to_numpy()
        if self.objective == "multiclass":
            predictions = predictions.reshape(
                (self.num_class, subset.sum(), self.n_intervals), order="F"
            )
        if cumulative:
            predictions = np.cumprod(predictions, axis=-1)
        return predictions

    def save_model(self, file_name: str = "IFE_Model", path: str = "") -> None:
        """Save the pandas DataFrame model to disk."""
        self.model.to_csv(path + file_name + ".csv")

    def hyperoptimize(self, **kwargs) -> dict:
        """Returns None for InteractedFixedEffectsModeler, which does not have hyperparameters"""
        warn(
            "Warning: InteractedFixedEffectsModeler does not have hyperparameters to optimize."
        )
        return None


class IFESurvivalModeler(IFEModeler, SurvivalModeler):
    """Predict with survival rate of training observations with same values."""

    pass


class InteractedFixedEffectsModeler(IFESurvivalModeler):
    """Deprecated alias for IFESurvivalModeler"""

    warn(
        "The name 'InteractedFixedEffectsModeler' is deprecated. "
        "Please use 'IFESurvivalModeler' instead.",
        DeprecationWarning,
    )


class IFEStateModeler(IFEModeler, StateModeler):
    """Forecast the future value of a feature conditional on survival using the mean of observations with the same values."""

    pass


class IFEExitModeler(IFEModeler, ExitModeler):
    """Forecast the circumstance of exit conditional on exit using the mean of observations with the same values."""

    pass
