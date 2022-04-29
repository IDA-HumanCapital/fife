"""The abstract classes on which all FIFE modelers are based."""

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Union

from lifelines.utils import concordance_index
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, r2_score, roc_auc_score


def default_subset_to_all(
    subset: Union[None, pd.core.series.Series], data: pd.core.frame.DataFrame
) -> pd.core.series.Series:
    """Map an unspecified subset to an entirely True boolean mask."""
    if subset is None:
        return pd.Series([True] * data.shape[0], index=data.index)
    return subset


def compute_metrics_for_binary_outcome(
    actuals: Union[pd.Series, pd.DataFrame],
    predictions: np.ndarray,
    threshold_positive: Union[None, str, float] = 0.5,
    share_positive: Union[None, str, float] = None,
    weights: Union[None, np.ndarray] = None,
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
        weights: A 1-D array of weights with the same length as actuals and predictions.
        Each prediction contributes to metrics in proportion to its weight. If None, each prediction has the same weight.

    Returns:
        An ordered dictionary containing key-value pairs for area under the
        receiver operating characteristic curve (AUROC), predicted and
        actual shares of observations with an outcome of True, and all
        elements of the confusion matrix.
    """
    metrics = OrderedDict()
    if actuals.any() and not actuals.all():
        metrics["AUROC"] = roc_auc_score(actuals, predictions, sample_weight=weights)
    else:
        metrics["AUROC"] = np.nan
    metrics["Predicted Share"] = np.average(predictions, weights=weights)
    metrics["Actual Share"] = np.average(actuals, weights=weights)
    if actuals.empty:
        (
            metrics["True Positives"],
            metrics["False Negatives"],
            metrics["False Positives"],
            metrics["True Negatives"],
        ) = [0, 0, 0, 0]
    else:
        if share_positive == "predicted":
            share_positive = metrics["Predicted Share"]
        if share_positive is not None:
            threshold_positive = np.quantile(predictions, 1 - share_positive)
        elif threshold_positive == "predicted":
            threshold_positive = metrics["Predicted Share"]
        (
            metrics["True Positives"],
            metrics["False Negatives"],
            metrics["False Positives"],
            metrics["True Negatives"],
        ) = (
            confusion_matrix(
                actuals,
                predictions >= threshold_positive,
                labels=[True, False],
                sample_weight=weights,
            )
            .ravel()
            .tolist()
        )
    return metrics


def compute_metrics_for_categorical_outcome(
    actuals: pd.DataFrame,
    predictions: np.ndarray,
    weights: Union[None, np.ndarray] = None,
) -> OrderedDict:
    """Evaluate predicted probabilities against actual categorical outcome values.

    Args:
        actuals: A DataFrame representing one-hot-encoded actual class membership.
        predictions: A DataFrame of predicted probabilities of class membership
            for the respective observations represented in actuals.
        weights: A 1-D array of weights with the same length as actuals and predictions.
        Each prediction contributes to metrics in proportion to its weight. If None, each prediction has the same weight.

    Returns:
        An ordered dictionary containing a key-value pair for area under the
        receiver operating characteristic curve (AUROC). Classes with no positive values are not included in AUROC calculation.
    """
    metrics = OrderedDict()
    if actuals.all().any():
        metrics["AUROC"] = np.nan
    else:
        positive_cols = actuals.any()
        metrics["AUROC"] = roc_auc_score(
            actuals.loc[:, positive_cols],
            predictions[:, positive_cols],
            multi_class="ovr",
            average="weighted",
            sample_weight=weights,
        )
    return metrics


def compute_metrics_for_numeric_outcome(
    actuals: pd.core.series.Series,
    predictions: np.ndarray,
    weights: Union[None, np.ndarray] = None,
) -> OrderedDict:
    """Evaluate predicted numeric values against actual outcome values.

    Args:
        actuals: A Series representing actual outcome values.
        predictions: A Series of predictions for the respective outcome values.

    Returns:
        An ordered dictionary containing a key-value pair for R-squared.
    """
    metrics = OrderedDict()
    non_null_obs = actuals.notnull()
    metrics["R-squared"] = r2_score(
        actuals[non_null_obs],
        predictions[non_null_obs],
        sample_weight=weights[non_null_obs] if weights else None,
    )
    return metrics


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
        spell_col (str): Name of the column representing the number of
            previous spells of consecutive observations of the same individual.
        weight_col (str): Name of the column representing observation weights.
        reserved_cols (list): Column names of non-features.
        numeric_features (list): Column names of numeric features.
        n_intervals (int): The largest number of periods ahead to forecast
        allow_gaps (bool): Whether or not observations should be included for
            training and evaluation if there is a period without an observation
            between the period of observations and the last period of the
            given time horizon.
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
        spell_col: str = "_spell",
        weight_col: Union[None, str] = None,
        allow_gaps: bool = False,
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
            spell_col (str): Name of the column representing the number of
                previous spells of consecutive observations of the same individual.
            weight_col (str): Name of the column representing observation weights.
            allow_gaps (bool): Whether or not observations should be included for
                training and evaluation if there is a period without an observation
                between the period of observations and the last period of the
                given time horizon.
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
        self.spell_col = spell_col
        self.weight_col = weight_col
        self.allow_gaps = allow_gaps
        self.reserved_cols = [
            self.duration_col,
            self.event_col,
            self.predict_col,
            self.test_col,
            self.validation_col,
            self.period_col,
            self.max_lead_col,
            self.spell_col,
        ]
        if self.config:
            self.reserved_cols.append(self.config["INDIVIDUAL_IDENTIFIER"])
        if self.weight_col:
            self.reserved_cols.append(self.weight_col)
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

    @abstractmethod
    def subset_for_training_horizon(
        self, data: pd.DataFrame, time_horizon: int
    ) -> pd.DataFrame:
        """Return only observations where the outcome is observed."""

    @abstractmethod
    def label_data(self, time_horizon: int) -> pd.DataFrame:
        """Return data with an outcome label for each observation."""

    @abstractmethod
    def save_model(self, path: str = "") -> None:
        """Save model file(s) to disk."""

    def transform_features(self) -> pd.DataFrame:
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
        spell_col (str): Name of the column representing the number of
            previous spells of consecutive observations of the same individual.
        weight_col (str): Name of the column representing observation weights.
        reserved_cols (list): Column names of non-features.
        numeric_features (list): Column names of numeric features.
        n_intervals (int): The largest number of periods ahead to forecast.
        allow_gaps (bool): Whether or not observations should be included for
            training and evaluation if there is a period without an observation
            between the period of observations and the last period of the
            given time horizon.
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
                the metrics will be computed. If None, default to all test
                observations in the earliest period of the test set.
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
        if subset is None:
            subset = self.data[self.predict_col]
        predictions = self.predict(subset=subset, cumulative=(not self.allow_gaps))
        metrics = []
        lead_lengths = np.arange(self.n_intervals) + 1
        for lead_length in lead_lengths:
            actuals = self.label_data(lead_length - 1)[subset].reset_index()
            actuals = actuals[actuals[self.max_lead_col] >= lead_length]
            weights = actuals[self.weight_col].values if self.weight_col else None
            actuals = actuals["_label"]
            metrics.append(
                compute_metrics_for_binary_outcome(
                    actuals,
                    predictions[:, lead_length - 1][actuals.index],
                    threshold_positive=threshold_positive,
                    share_positive=share_positive,
                    weights=weights,
                )
            )
        metrics = pd.DataFrame(metrics, index=lead_lengths)
        metrics.index.name = "Lead Length"
        metrics["Other Metrics:"] = ""
        if (not self.allow_gaps) and (self.weight_col is None):
            concordance_index_value = concordance_index(
                self.data[subset][[self.duration_col, self.max_lead_col]].min(axis=1),
                np.sum(predictions, axis=-1),
                self.data[subset][self.event_col],
            )
            metrics["C-Index"] = np.where(
                metrics.index == 1, concordance_index_value, ""
            )
        metrics = metrics.dropna()
        return metrics

    def forecast(self) -> pd.core.frame.DataFrame:
        """Tabulate survival probabilities for most recent observations."""
        columns = [
            str(i + 1) + "-period Survival Probability" for i in range(self.n_intervals)
        ]
        return pd.DataFrame(
            self.predict(
                subset=self.data[self.predict_col], cumulative=(not self.allow_gaps)
            ),
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
        predictions = self.predict(subset=subset, cumulative=(not self.allow_gaps))
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
        if self.allow_gaps:
            return data[data[self.max_lead_col] > time_horizon]
        return data[(data[self.duration_col] + data[self.event_col] > time_horizon)]

    def label_data(self, time_horizon: int) -> pd.DataFrame:
        """Return data with an indicator for survival for each observation."""
        data = self.data.copy()
        data[self.duration_col] = data[[self.duration_col, self.max_lead_col]].min(
            axis=1
        )
        if self.allow_gaps:
            ids = data[
                [self.config["INDIVIDUAL_IDENTIFIER"], self.config["TIME_IDENTIFIER"]]
            ]
            ids[self.config["TIME_IDENTIFIER"]] = (
                ids[self.config["TIME_IDENTIFIER"]] - time_horizon - 1
            )
            ids["_label"] = True
            data = data.merge(
                ids,
                how="left",
                on=[
                    self.config["INDIVIDUAL_IDENTIFIER"],
                    self.config["TIME_IDENTIFIER"],
                ],
            )
            data["_label"] = data["_label"].fillna(False)
        else:
            data["_label"] = data[self.duration_col] > time_horizon
        return data


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
        spell_col (str): Name of the column representing the number of
            previous spells of consecutive observations of the same individual.
        weight_col (str): Name of the column representing observation weights.
        reserved_cols (list): Column names of non-features.
        numeric_features (list): Column names of numeric features.
        n_intervals (int): The largest number of periods ahead to forecast.
        state_col (str): The column representing the state to forecast.
        objective (str): The model objective appropriate for the outcome type;
            "multiclass" for categorical states and "regression" for numeric
            states.
        class_values (pd.Int64Index): The state categories.
        num_class (int): The number of state categories or, if the state is
            numeric, None.
    """

    def __init__(self, state_col, **kwargs):
        """Initialize the StateModeler.

        Args:
            state_col: The column representing the state to forecast.
            **kwargs: Arguments to Modeler.__init__().
        """
        super().__init__(**kwargs)
        self.state_col = state_col
        if self.data is not None:
            if self.state_col in self.categorical_features:
                self.objective = "multiclass"
                self.class_values = self.data[self.state_col].cat.categories
                self.num_class = len(self.class_values)
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
                the metrics will be computed. If None, default to all test
                observations in the earliest period of the test set.

        Returns:
            A DataFrame containing, for each lead length, area under the
            receiver operating characteristic curve (AUROC) for categorical
            states, or, for numeric states, R-squared.
        """
        if subset is None:
            subset = self.data[self.predict_col]
        predictions = self.predict(subset=subset, cumulative=False)
        metrics = []
        lead_lengths = np.arange(self.n_intervals) + 1
        for lead_length in lead_lengths:
            actuals = self.subset_for_training_horizon(
                self.label_data(lead_length - 1)[subset].reset_index(), lead_length - 1
            )
            weights = actuals[self.weight_col].values if self.weight_col else None
            actuals = actuals["_label"]
            if self.objective == "multiclass":
                actuals = pd.DataFrame(
                    {label: actuals == label for label in range(self.num_class)},
                    index=actuals.index,
                )
                metrics.append(
                    compute_metrics_for_categorical_outcome(
                        actuals,
                        predictions[:, :, lead_length - 1].T[actuals.index],
                        weights=weights,
                    )
                )
            else:
                metrics.append(
                    compute_metrics_for_numeric_outcome(
                        actuals,
                        predictions[:, lead_length - 1][actuals.index],
                        weights=weights,
                    )
                )
        metrics = pd.DataFrame(metrics, index=lead_lengths)
        metrics.index.name = "Lead Length"
        return metrics

    def forecast(self) -> pd.core.frame.DataFrame:
        """Tabulate state probabilities for most recent observations."""
        forecasts = self.predict(subset=self.data[self.predict_col], cumulative=False)
        if self.objective == "multiclass":
            columns = [
                str(i + 1) + "-period State Probabilities"
                for i in range(self.n_intervals)
            ]
            ids = np.repeat(
                self.data[self.config["INDIVIDUAL_IDENTIFIER"]][
                    self.data[self.predict_col]
                ],
                forecasts.shape[0],
            )
            states = np.tile(self.class_values, forecasts.shape[1])
            forecasts = np.reshape(
                forecasts,
                (forecasts.shape[0] * forecasts.shape[1], forecasts.shape[2]),
                order="F",
            )
            index = pd.MultiIndex.from_arrays(
                [ids, states],
                names=[
                    self.config["INDIVIDUAL_IDENTIFIER"],
                    f"Future {self.state_col}",
                ],
            )
            forecasts = pd.DataFrame(forecasts, columns=columns, index=index)
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
        if self.allow_gaps:
            return data[data[self.max_lead_col] > time_horizon].dropna(
                subset=["_label"]
            )
        return data[(data[self.duration_col] > time_horizon)]

    def label_data(self, time_horizon: int) -> pd.Series:
        """Return data with the future state for each observation."""
        data = self.data.copy()
        data[self.duration_col] = data[[self.duration_col, self.max_lead_col]].min(
            axis=1
        )
        ids = data[
            [
                self.config["INDIVIDUAL_IDENTIFIER"],
                self.period_col,
                self.state_col,
            ]
        ]
        ids[self.period_col] = ids[self.period_col] - time_horizon - 1
        ids = ids.rename({self.state_col: "_label"}, axis=1)
        if self.objective == "multiclass":
            ids["_label"] = ids["_label"].cat.codes
        data = data.merge(
            ids,
            how="left",
            on=[self.config["INDIVIDUAL_IDENTIFIER"], self.period_col],
        )
        return data


class ExitModeler(StateModeler):
    """Forecast the circumstance of exit conditional on exit."""

    def __init__(self, exit_col, **kwargs):
        """Initialize the ExitModeler.

        Args:
            exit_col: The column representing the exit circumstance to forecast.
            **kwargs: Arguments to Modeler.__init__().
        """
        super().__init__(exit_col, **kwargs)
        if self.objective == "multiclass":
            self.class_values = self.data[
                (self.data["_duration"] == 0) & (self.data["_event_observed"] == True)
            ][self.state_col].unique()
            self.data[self.state_col] = self.data[
                self.state_col
            ].cat.reorder_categories(
                list(self.class_values)
                + [
                    cat
                    for cat in self.data[self.state_col].cat.categories
                    if cat not in self.class_values
                ]
            )
            self.num_class = len(self.class_values)
        self.exit_col = self.state_col
        if self.data is not None:
            if self.state_col in self.categorical_features:
                self.categorical_features.remove(self.state_col)
            elif self.state_col in self.numeric_features:
                self.numeric_features.remove(self.state_col)

    def subset_for_training_horizon(
        self, data: pd.DataFrame, time_horizon: int
    ) -> pd.DataFrame:
        """Return only observations where exit is observed at the given time horizon."""
        if self.allow_gaps:
            ids = data[
                [self.config["INDIVIDUAL_IDENTIFIER"], self.config["TIME_IDENTIFIER"]]
            ]
            ids[self.config["TIME_IDENTIFIER"]] = (
                ids[self.config["TIME_IDENTIFIER"]] - time_horizon - 1
            )
            ids["exit"] = False
            data = data.merge(
                ids,
                how="left",
                on=[
                    self.config["INDIVIDUAL_IDENTIFIER"],
                    self.config["TIME_IDENTIFIER"],
                ],
            )
            data = data[data["exit"].isna() & (data[self.max_lead_col] > time_horizon)]
            data = data.drop("exit", axis=1)
        else:
            data = data[
                data[self.event_col] & (data[self.duration_col] == time_horizon)
            ]
        return data

    def label_data(self, time_horizon: int) -> pd.Series:
        """Return data with the exit circumstance for each observation."""
        data = self.data.copy()
        data[self.duration_col] = data[[self.duration_col, self.max_lead_col]].min(
            axis=1
        )
        if self.objective == "multiclass":
            temp_exit_col = data[self.exit_col]
            data[self.exit_col] = data[self.exit_col].cat.codes
        data["_label"] = data.groupby(
            [self.config["INDIVIDUAL_IDENTIFIER"], self.spell_col]
        )[self.exit_col].transform("last")
        if self.objective == "multiclass":
            data[self.exit_col] = temp_exit_col
        return data
