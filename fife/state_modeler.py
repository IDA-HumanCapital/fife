"""The template class for FIFE state modelers."""

from collections import OrderedDict
from typing import Union

from fife.modeler import default_subset_to_all, Modeler
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def compute_metrics_for_categorical_outcome(
    actuals: pd.core.series.Series,
    predictions: np.ndarray,
    threshold_positive: Union[None, str, float] = 0.5,
    share_positive: Union[None, str, float] = None,
) -> OrderedDict:
    """Evaluate predicted probabilities against actual binary outcome values.

    Args:
        actuals: A Series representing actual Boolean outcome values.
        predictions: A Series of predicted probabilities of the respective
            outcome values.
        threshold_positive: None, "predicted", or a value in [0, 1]
            representing the minimum predicted probability considered to be a
            positive prediction; Specify "predicted" to use the predicted share
            positive in each time horizon. Overridden by share_positive.
        share_positive: None, "predicted", or a value in [0, 1] representing
            the share of observations with the highest predicted probabilities
            considered to have positive predictions. Specify "predicted" to use
            the predicted share positive in each time horizon. Probability ties
            may cause share of positives in output to exceed given value.
            Overrides threshold_positive.

    Returns:
        An ordered dictionary containing key-value pairs for area under the
        receiver operating characteristic curve (AUROC), predicted and
        actual shares of observations with an outcome of True, and all
        elements of the confusion matrix.
    """
    metrics = OrderedDict()
    if actuals.any() and not actuals.all():
        metrics["AUROC"] = roc_auc_score(actuals, predictions, multi_class="ovr")
    else:
        metrics["AUROC"] = np.nan
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
        predictions = self.predict(subset=subset)
        metrics = []
        lead_lengths = np.arange(self.n_intervals) + 1
        for lead_length in lead_lengths:
            actuals = (
                self.data[subset]
                .groupby(self.config["INDIVIDUAL_IDENTIFIER"])
                .shift(-lead_length)
            )
            metrics.append(
                compute_metrics_for_categorical_outcome(
                    actuals, np.array(predictions[:, lead_length - 1].to_list())
                )
            )
        metrics = pd.DataFrame(metrics, index=lead_lengths)
        metrics.index.name = "Lead Length"
        return metrics

    def forecast(self) -> pd.core.frame.DataFrame:
        """Tabulate state probabilities for most recent observations."""
        columns = [
            str(i + 1) + "-period State Probabilities" for i in range(self.n_intervals)
        ]
        forecasts = self.predict(subset=self.data[self.predict_col])
        index = np.repeat(
                self.data[self.config["INDIVIDUAL_IDENTIFIER"]][
                    self.data[self.predict_col]
                ], forecasts.shape[0]
            )
        states = np.tile(self.data[self.state_col].cat.categories,
                         forecasts.shape[1])
        forecasts = np.reshape(forecasts,
                               (forecasts.shape[0] * forecasts.shape[1],
                                forecasts.shape[2]),
                               order="F")
        forecasts = pd.DataFrame(
            forecasts,
            columns=columns,
            index=index,
        )
        forecasts[self.state_col] = states
        return forecasts
