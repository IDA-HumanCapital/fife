"""The template class for FIFE survival modelers."""

from collections import OrderedDict
from typing import Union

from fife.modeler import default_subset_to_all, Modeler
from lifelines.utils import concordance_index
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score


def compute_metrics_for_binary_outcome(
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
        metrics["AUROC"] = roc_auc_score(actuals, predictions)
    else:
        metrics["AUROC"] = np.nan
    metrics["Predicted Share"] = predictions.mean()
    metrics["Actual Share"] = actuals.mean()
    if actuals.empty:
        (
            metrics["True Positives"],
            metrics["False Negatives"],
            metrics["False Positives"],
            metrics["True Negatives"],
        ) = [0, 0, 0, 0]
    else:
        if share_positive == "predicted":
            share_positive = predictions.mean()
        if share_positive is not None:
            threshold_positive = np.quantile(predictions, 1 - share_positive)
        elif threshold_positive == "predicted":
            threshold_positive = predictions.mean()
        (
            metrics["True Positives"],
            metrics["False Negatives"],
            metrics["False Positives"],
            metrics["True Negatives"],
        ) = (
            confusion_matrix(
                actuals, predictions >= threshold_positive, labels=[True, False]
            )
            .ravel()
            .tolist()
        )
    return metrics


class SurvivalModeler(Modeler):
    """Forecast probabilities of being observed in future periods.

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
        objective (str): The LightGBM model objective appropriate for the
            outcome type, which is "binary" for binary classification.
        num_class (int): The num_class LightGBM parameter, which is 1 for
            binary classification.
    """

    def __init__(self, **kwargs):
        """Initialize the SurvivalModeler.

        Args:
            **kwargs: Arguments to Modeler.__init__().
        """
        super().__init__(**kwargs)
        self.objective = "binary"
        self.num_class = 1

    def evaluate(
        self,
        subset: Union[None, pd.core.series.Series] = None,
        threshold_positive: Union[None, str, float] = 0.5,
        share_positive: Union[None, str, float] = None,
    ) -> pd.core.frame.DataFrame:
        """Tabulate model performance metrics.

        Args:
            subset: A Boolean Series that is True for observations over which
                the metrics will be computed. If None, default to all
                observations.
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
            A DataFrame containing, for the binary outcomes of survival to each
            lead length, area under the receiver operating characteristic
            curve (AUROC), predicted and actual shares of observations with an
            outcome of True, and all elements of the confusion matrix. Also
            includes concordance index over the restricted mean survival time.
        """
        subset = default_subset_to_all(subset, self.data)
        predictions = self.predict(subset=subset, cumulative=True)
        metrics = []
        lead_lengths = np.arange(self.n_intervals) + 1
        for lead_length in lead_lengths:
            actuals = self.subset_for_training_horizon(self.label_data(lead_length - 1)[subset].reset_index(),
                                                       lead_length - 1)["label"]
            metrics.append(
                compute_metrics_for_binary_outcome(
                    actuals,
                    predictions[:, lead_length - 1][
                        actuals.index
                    ],
                    threshold_positive=threshold_positive,
                    share_positive=share_positive,
                )
            )
        metrics = pd.DataFrame(metrics, index=lead_lengths)
        metrics.index.name = "Lead Length"
        metrics["Other Metrics:"] = ""
        concordance_index_value = concordance_index(
            self.data[subset][[self.duration_col, self.max_lead_col]].min(axis=1),
            np.sum(predictions, axis=-1),
            self.data[subset][self.event_col],
        )
        metrics["C-Index"] = np.where(metrics.index == 1, concordance_index_value, "")
        return metrics

    def forecast(self) -> pd.core.frame.DataFrame:
        """Tabulate survival probabilities for most recent observations."""
        columns = [
            str(i + 1) + "-period Survival Probability" for i in range(self.n_intervals)
        ]
        return pd.DataFrame(
            self.predict(subset=self.data[self.predict_col], cumulative=True),
            columns=columns,
            index=(
                self.data[self.config["INDIVIDUAL_IDENTIFIER"]][
                    self.data[self.predict_col]
                ]
            ),
        )

    def tabulate_survival_by_quantile(
        self, n_quantiles: int, subset: Union[None, pd.core.series.Series] = None
    ) -> pd.core.frame.DataFrame:
        """Count number survived, grouped by survival probability.

        Args:
            subset: A Boolean Series that is True for observations over which
                the counts will be computed. If None, default to all
                observations.
            n_quantiles: Number of groups. Groups will contain the same number
                of observations except to keep observations with the same
                predicted probability in the same group. Quantile 1 represents
                the individuals least likely to survive the given lead length.

        Returns:
            A DataFrame where each row contains the group size and number
            survived for each combination of lead length and group.
        """
        subset = default_subset_to_all(subset, self.data)
        predictions = self.predict(subset=subset, cumulative=True)
        actual_durations = self.data[subset][
            [self.duration_col, self.max_lead_col]
        ].min(axis=1)
        lead_lengths = np.arange(self.n_intervals) + 1
        counts = []
        for lead_length in lead_lengths:
            actuals = (
                actual_durations[self.data[self.max_lead_col] >= lead_length]
                >= lead_length
            )
            quantiles = pd.qcut(
                predictions[:, lead_length - 1][
                    self.data[subset][self.max_lead_col] >= lead_length
                ],
                n_quantiles,
                labels=False,
                duplicates="drop",
            )
            actuals = actuals.groupby(quantiles).agg(["sum", "count"])
            actuals["Lead Length"] = lead_length
            counts.append(actuals)
        counts = pd.concat(counts)
        counts.index.name = "Prediction Quantile"
        counts.columns = ["Number Survived", "Observations", "Lead Length"]
        counts = counts.reset_index()
        counts = counts[
            ["Lead Length", "Prediction Quantile", "Number Survived", "Observations"]
        ]
        counts["Prediction Quantile"] = counts["Prediction Quantile"] + 1
        return counts

    def tabulate_retention_rates(
        self, lead_periods: int, time_ids: Union[None, pd.core.series.Series] = None
    ):
        """Produce a table of actual and predicted retention rates.

        Retention rate is the share of individuals observed the given number of
        periods prior who survived to the corresponding period.

        Args:
            lead_periods: The number of periods an individual must survive to
                be considered retained.
            time_ids: Values representing the period of each observation. If
                None, default to number of periods since first period in data.

        Returns:
            Observed retention rate, predicted retention rate for periods where
            actual retention is observed in training set, and predicted
            retention rate for periods where actual retention rate is not
            observed in training set.
        """
        if time_ids is None:
            time_ids = self.data[self.period_col]
        actual_retention_rate = (
            (self.data[self.duration_col] >= lead_periods).groupby(time_ids).mean()
        )
        fitted_retention_rate = (
            pd.Series(self.predict()[:, lead_periods - 1]).groupby(time_ids).mean()
        )
        retention_rates = pd.DataFrame([actual_retention_rate, fitted_retention_rate]).T
        retention_rates.columns = ["Actual Retention Rate", "Fitted Retention Rate"]
        retention_rates["Actual Retention Rate"].iloc[-lead_periods:] = np.nan
        period_length = retention_rates.index[1] - retention_rates.index[0]
        for _ in range(lead_periods):
            retention_rates.loc[retention_rates.index.max() + period_length] = np.nan
        retention_rates = retention_rates.shift(lead_periods)
        retention_rates["Forecast Retention Rate"] = np.nan
        untrained_periods = lead_periods + self.config.get("TEST_PERIODS", 0)
        retention_rates["Forecast Retention Rate"].iloc[
            -untrained_periods:
        ] = retention_rates["Fitted Retention Rate"].iloc[-untrained_periods:]
        retention_rates["Fitted Retention Rate"].iloc[-untrained_periods:] = np.nan
        retention_rates.loc[retention_rates.index.max() + period_length] = np.nan
        retention_rates.index.name = "Period"
        return retention_rates

    def subset_for_training_horizon(
        self, data: pd.DataFrame, time_horizon: int
    ) -> pd.DataFrame:
        """Return only observations where survival would be observed."""
        return data[(data[self.duration_col] + data[self.event_col] > time_horizon)]

    def label_data(self, time_horizon: int) -> pd.DataFrame:
        """Return data with an indicator for survival for each observation."""
        data = self.data.copy()
        data[self.duration_col] = data[[self.duration_col, self.max_lead_col]].min(
            axis=1
        )
        data["label"] = data[self.duration_col] > time_horizon
        return data
