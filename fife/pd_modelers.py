"""FIFE modeler based on Pandas, which tabulates interacted fixed effects."""

from typing import Union

from fife import base_modelers
import numpy as np
import pandas as pd


class InteractedFixedEffectsModeler(base_modelers.SurvivalModeler):
    """Predict with survival rate of training observations with same values.

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
        """Compute survival rate for each combination of categorical values."""
        cell_rates = pd.DataFrame()
        train_subset = (
            ~self.data[self.validation_col]
            & ~self.data[self.test_col]
            & ~self.data[self.predict_col]
        )
        for time_horizon in range(self.n_intervals):
            train_data = self.data[
                (
                    self.data[[self.duration_col, self.max_lead_col]].min(axis=1)
                    + self.data[self.event_col]
                    > time_horizon
                )
                & train_subset
            ]
            train_data["survived"] = (
                train_data[[self.duration_col, self.max_lead_col]].min(axis=1)
                > time_horizon
            )
            cell_rates[time_horizon + 1] = train_data.groupby(
                self.categorical_features
            )["survived"].mean()
        return cell_rates

    def predict(
        self, subset: Union[None, pd.core.series.Series] = None, cumulative: bool = True
    ) -> np.ndarray:
        """Map observations to survival rates from their categorical values.

        Map observations with a combination of categorical values not seen in
        the training data to the mean survival rate in the training set.

        Args:
            subset: A Boolean Series that is True for observations for which
                predictions will be produced. If None, default to all
                observations.
            cumulative: If True, will produce cumulative survival
                probabilies. If False, will produce marginal survival
                probabilities (i.e., one minus the hazard rate).

        Returns:
            A numpy array of survival probabilities by observation and lead
            length.
        """
        subset = base_modelers.default_subset_to_all(subset, self.data)
        predictions = self.data[subset].merge(
            self.model, how="left", left_on=self.categorical_features, right_index=True
        )
        predictions = predictions[self.model.columns]
        predictions = predictions.fillna(predictions.mean())
        predictions = predictions.to_numpy()
        if cumulative:
            predictions = np.cumprod(predictions, axis=1)
        return predictions

    def save_model(self, file_name: str = "IFE_Model", path: str = "") -> None:
        """Save the pandas DataFrame model to disk."""
        self.model.to_csv(path + file_name + ".csv")

    def hyperoptimize(
        self,
        **kwargs
    ) -> dict:
        """Returns None for InteractedFixedEffectsModeler, which does not have hyperparameters"""
        warn("Warning: InteractedFixedEffectsModeler does not have hyperparameters to optimize.")
        return None
