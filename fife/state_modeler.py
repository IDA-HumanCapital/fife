"""The template class for FIFE state modelers."""

from collections import OrderedDict
from typing import Union

from fife.modeler import default_subset_to_all, Modeler
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score


def compute_metrics_for_categorical_outcome(
    actuals: pd.DataFrame, predictions: np.ndarray
) -> OrderedDict:
    """Evaluate predicted probabilities against actual binary outcome values.

    Args:
        actuals: A DataFrame representing one-hot-encoded actual class membership.
        predictions: A DataFrame of predicted probabilities of class membership 
            for the respective observations represented in actuals.

    Returns:
        An ordered dictionary containing a key-value pair for area under the
        receiver operating characteristic curve (AUROC).
    """
    metrics = OrderedDict()
    if actuals.sum().min() > 0:
        metrics["AUROC"] = roc_auc_score(actuals, predictions, multi_class="ovr")
    else:
        metrics["AUROC"] = np.nan
    return metrics


def compute_metrics_for_numeric_outcome(
    actuals: pd.core.series.Series, predictions: np.ndarray
) -> OrderedDict:
    """Evaluate predicted numeric values against actual outcome values.

    Args:
        actuals: A Series representing actual outcome values.
        predictions: A Series of predictions for the respective outcome values.

    Returns:
        An ordered dictionary containing a key-value pair for R-squared.
    """
    metrics = OrderedDict()
    metrics["R-squared"] = r2_score(actuals, predictions)
    return metrics


class StateModeler(Modeler):
    """Forecast the future value of a feature conditional on survival.

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
        n_intervals (int): The largest number of periods ahead to forecast.
        state_col (str): The column representing the state to forecast.
        objective (str): The model objective appropriate for the outcome type;
            "multiclass" for categorical states and "regression" for numeric
            states.
        num_class (int): The number of state categories or, if the state is
            numeric, None.
    """

    def __init__(self, state_col, **kwargs):
        """Initialize the StateModeler.

        Args:
            state_col: The column representing the state to be forecast.
            **kwargs: Arguments to Modeler.__init__().
        """
        super().__init__(**kwargs)
        self.state_col = state_col
        if self.data is not None:
            if self.state_col in self.categorical_features:
                self.objective = "multiclass"
                self.num_class = len(self.data[self.state_col].cat.categories)
            elif self.state_col in self.numeric_features:
                self.objective = "regression"
                self.num_class = None
            else:
                raise ValueError("state_col not in features.")

    def evaluate(
        self, subset: Union[None, pd.core.series.Series] = None
    ) -> pd.core.frame.DataFrame:
        """Tabulate model performance metrics.

        Args:
            subset: A Boolean Series that is True for observations over which
                the metrics will be computed. If None, default to all
                observations.

        Returns:
            A DataFrame containing, for each lead length, area under the
            receiver operating characteristic curve (AUROC) for categorical
            states, or, for numeric states, R-squared.
        """
        subset = default_subset_to_all(subset, self.data)
        predictions = self.predict(subset=subset, cumulative=False)
        metrics = []
        lead_lengths = np.arange(self.n_intervals) + 1
        for lead_length in lead_lengths:
            actuals = self.subset_for_training_horizon(self.label_data(lead_length - 1)[subset].reset_index(),
                                                       lead_length - 1)["label"]
            if self.state_col in self.categorical_features:
                metrics.append(
                    compute_metrics_for_categorical_outcome(
                        pd.get_dummies(actuals),
                        predictions[:, :, lead_length - 1].T[
                            actuals.index
                        ],
                    )
                )
            else:
                metrics.append(
                    compute_metrics_for_numeric_outcome(
                        actuals,
                        predictions[:, lead_length - 1][
                            actuals.index
                        ],
                    )
                )
        metrics = pd.DataFrame(metrics, index=lead_lengths)
        metrics.index.name = "Lead Length"
        return metrics

    def forecast(self) -> pd.core.frame.DataFrame:
        """Tabulate state probabilities for most recent observations."""
        forecasts = self.predict(subset=self.data[self.predict_col], cumulative=False)
        if self.state_col in self.categorical_features:
            columns = [
                str(i + 1) + "-period State Probabilities"
                for i in range(self.n_intervals)
            ]
            index = np.repeat(
                self.data[self.config["INDIVIDUAL_IDENTIFIER"]][
                    self.data[self.predict_col]
                ],
                forecasts.shape[0],
            )
            states = np.tile(
                self.data[self.state_col].cat.categories, forecasts.shape[1]
            )
            forecasts = np.reshape(
                forecasts,
                (forecasts.shape[0] * forecasts.shape[1], forecasts.shape[2]),
                order="F",
            )
            forecasts = pd.DataFrame(forecasts, columns=columns, index=index)
            forecasts[f"Future {self.state_col}"] = states
        else:
            columns = [
                str(i + 1) + f"-period {self.state_col} Forecast"
                for i in range(self.n_intervals)
            ]
            index = self.data[self.config["INDIVIDUAL_IDENTIFIER"]][
                self.data[self.predict_col]
            ]
            forecasts = pd.DataFrame(forecasts, columns=columns, index=index)
        return forecasts

    def subset_for_training_horizon(
        self, data: pd.DataFrame, time_horizon: int
    ) -> pd.DataFrame:
        """Return only observations where the future state is observed."""
        return data[(data[self.duration_col] > time_horizon)]

    def label_data(self, time_horizon: int) -> pd.Series:
        """Return data with the future state for each observation."""
        data = self.data.copy()
        data[self.duration_col] = data[[self.duration_col, self.max_lead_col]].min(
            axis=1
        )
        data["label"] = (
            data.groupby(self.config["INDIVIDUAL_IDENTIFIER"])[self.state_col]
            .shift(-time_horizon - 1)
        )
        if self.state_col in self.categorical_features:
            data["label"] = data["label"].cat.codes
        return data
