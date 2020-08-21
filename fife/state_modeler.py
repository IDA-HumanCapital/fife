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
    """Set template for modelers that produce state probabilities and metrics.

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
    """

    def evaluate(
        self, subset: Union[None, pd.core.series.Series] = None
    ) -> pd.core.frame.DataFrame:
        """Tabulate model performance metrics.

        Args:
            subset: A Boolean Series that is True for observations over which
                the metrics will be computed. If None, default to all
                observations.

        Returns:
            A DataFrame containing area under the receiver operating characteristic
            curve (AUROC).
        """
        subset = default_subset_to_all(subset, self.data)
        predictions = self.predict(subset=subset, cumulative=False)
        metrics = []
        lead_lengths = np.arange(self.n_intervals) + 1
        for lead_length in lead_lengths:
            actuals = (
                self.data.groupby(self.config["INDIVIDUAL_IDENTIFIER"])[
                    self.state_col
                ].shift(-lead_length)
            )[subset][self.data[subset][self.duration_col] >= lead_length]
            if self.state_col in self.categorical_features:
                metrics.append(
                    compute_metrics_for_categorical_outcome(
                        pd.get_dummies(actuals.cat.codes),
                        predictions[:, :, lead_length - 1].T[
                            self.data[subset][self.duration_col] >= lead_length
                        ],
                    )
                )
            else:
                metrics.append(
                    compute_metrics_for_numeric_outcome(
                        actuals,
                        predictions[:, lead_length - 1][
                            self.data[subset][self.duration_col] >= lead_length
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
