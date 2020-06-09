"""The abstract class on which all FIFE modelers are based."""

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, List, Union

from lifelines.utils import concordance_index
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score


def default_subset_to_all(subset: Union[None, pd.core.series.Series],
                          data: pd.core.frame.DataFrame
                          ) -> pd.core.series.Series:
    """Map an unspecified subset to an entirely True boolean mask."""
    if subset is None:
        return pd.Series([True] * data.shape[0], index=data.index)
    return subset


def compute_metrics_for_binary_outcome(
        actuals: pd.core.series.Series,
        predictions: pd.core.series.Series,
        threshold_positive: Union[None, float] = 0.5,
        share_positive: Union[None, float] = None) -> OrderedDict:
    """Evaluate predicted probabilities against actual binary outcome values.

    Args:
        actuals: A Series representing actual Boolean outcome values.
        predictions: A Series of predicted probabilities of the respective
            outcome values.
        threshold_positive: None or a value in [0, 1] representing the minimum
            predicted probability considered to be a positive prediction;
            overridden by share_positive.
        share_positive: None or a value in [0, 1] representing the share of
            observations with the highest predicted probabilities considered to
            have positive predictions; probability ties may cause share of
            positives in output to exceed given value; overrides
            threshold_positive.

    Returns:
        An ordered dictionary containing key-value pairs for area under the
        receiver operating characteristic curve (AUROC), predicted and
        actual shares of observations with an outcome of True, and all
        elements of the confusion matrix.
    """
    metrics = OrderedDict()
    if actuals.any() and not actuals.all():
        metrics['AUROC'] = roc_auc_score(actuals, predictions)
    else:
        metrics['AUROC'] = np.nan
    metrics['Predicted Share'] = predictions.mean()
    metrics['Actual Share'] = actuals.mean()
    if actuals.empty:
        metrics['True Positives'], metrics['False Negatives'], \
            metrics['False Positives'], metrics['True Negatives'] = \
            [0, 0, 0, 0]
    else:
        if share_positive is not None:
            threshold_positive = np.quantile(predictions, 1 - share_positive)
        metrics['True Positives'], metrics['False Negatives'], \
            metrics['False Positives'], metrics['True Negatives'] = \
            confusion_matrix(actuals, predictions >= threshold_positive,
                             labels=[True, False]).ravel().tolist()
    return metrics


class SurvivalModeler(ABC):
    """Set template for modelers to produce survival probabilities and metrics.

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
        reserved_cols (list): Column names of non-features.
        numeric_features (list): Column names of numeric features.
        n_intervals (int): The largest number of one-period intervals any
            individual is observed to survive.
    """

    def __init__(self,
                 config: Union[None, dict] = {},
                 data: Union[None, pd.core.frame.DataFrame] = None,
                 categorical_features: Union[None, List[str]] = None,
                 duration_col: str = '_duration',
                 event_col: str = '_event_observed',
                 predict_col: str = '_predict_obs',
                 test_col: str = '_test',
                 validation_col: str = '_validation') -> None:
        """Characterize data for modelling.

        Identify each column as categorical, reserved, and numeric.
        Compute the maximum lead length for modelling.

        Args:
            config: User-provided configuration parameters.
            data: User-provided panel data.
            categorical_features: Column names of categorical features.
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
        """
        if (config.get('TIME_IDENTIFIER', '') == '') and data is not None:
            config['TIME_IDENTIFIER'] = data.columns[1]
            print('Time identifier column name not given; assumed to be '
                  f'second-leftmost column ({config["TIME_IDENTIFIER"]})')
        if (config.get('INDIVIDUAL_IDENTIFIER', '') == '') and data is not None:
            config['INDIVIDUAL_IDENTIFIER'] = data.columns[0]
            print('Individual identifier column name not given; assumed to be '
                  f'leftmost column ({config["INDIVIDUAL_IDENTIFIER"]})')
        self.config = config
        self.data = data
        self.duration_col = duration_col
        self.event_col = event_col
        self.predict_col = predict_col
        self.test_col = test_col
        self.validation_col = validation_col
        if config:
            self.reserved_cols = [self.duration_col,
                                  self.event_col,
                                  self.predict_col,
                                  self.test_col,
                                  self.validation_col,
                                  self.config.get('INDIVIDUAL_IDENTIFIER')]
            if self.data is not None and categorical_features is not None:
                self.categorical_features = [
                    col for col in categorical_features if col in self.data]
                self.numeric_features = [
                    feature for feature in self.data if feature not in
                    (categorical_features + self.reserved_cols)]

    @abstractmethod
    def train(self) -> Any:
        """Train and return a model."""

    @abstractmethod
    def predict(self,
                subset: Union[None, pd.core.series.Series] = None,
                cumulative: bool = True) -> np.ndarray:
        """Use trained model to produce observation survival probabilities."""

    def build_model(self) -> None:
        """Configure, train, and store a model."""
        self.n_intervals = self.set_n_intervals()
        self.model = self.train()

    def set_n_intervals(self) -> int:
        """Determine the maximum periods ahead the model will predict."""
        train_obs_by_lead_length = self.data[
            (~self.data[self.validation_col] &
             ~self.data[self.test_col] &
             ~self.data[self.predict_col])][self.duration_col].value_counts()
        n_intervals = (train_obs_by_lead_length[
            train_obs_by_lead_length > self.config.get('MIN_SURVIVORS_IN_TRAIN', 64)]
                       .index.max())
        return n_intervals

    def forecast(self) -> pd.core.frame.DataFrame:
        """Tabulate survival probabilities for most recent observations."""
        columns = [str(i + 1) + '-period Survival Probability' for
                   i in range(self.n_intervals)]
        return pd.DataFrame(
            self.predict(subset=self.data[self.predict_col], cumulative=True),
            columns=columns,
            index=(self.data[self.config['INDIVIDUAL_IDENTIFIER']]
                   [self.data[self.predict_col]]))

    def evaluate(self, subset: Union[None, pd.core.series.Series] = None,
                 threshold_positive: Union[None, float] = 0.5,
                 share_positive: Union[None, float] = None,
                 predict_beyond_subset: bool = False
                 ) -> pd.core.frame.DataFrame:
        """Tabulate model performance metrics.

        Args:
            subset: A Boolean Series that is True for observations over which
                the metrics will be computed. If None, default to all
                observations.
            threshold_positive: None or a value in [0, 1] representing the
                minimum predicted probability considered to be a positive
                prediction; overridden by share_positive.
            share_positive: None or a value in [0, 1] representing the share of
                observations with the highest predicted probabilities
                considered to have positive predictions; probability ties may
                cause share of positives in output to exceed given value;
                overrides threshold_positive.
            predict_beyond_subset: Whether or not predictions for time
                horizons that extend beyond the last period of the given subset
                will be evaluated.

        Returns:
            A DataFrame containing, for the binary outcomes of survival to each
            lead length, area under the receiver operating characteristic
            curve (AUROC), predicted and actual shares of observations with an
            outcome of True, and all elements of the confusion matrix. Also
            includes concordance index over the restricted mean survival time.
        """
        subset = default_subset_to_all(subset, self.data)
        predictions = self.predict(subset=subset, cumulative=True)
        actual_durations = self.data[subset][self.duration_col]
        factorized_time_ids = pd.factorize(
            self.data[self.config['TIME_IDENTIFIER']], sort=True)[0]
        if predict_beyond_subset:
            maximum_period = factorized_time_ids.max()
        else:
            maximum_period = factorized_time_ids[subset].max()
        maximum_lead = maximum_period - factorized_time_ids[subset]
        metrics = []
        lead_lengths = np.arange(self.n_intervals) + 1
        for lead_length in lead_lengths:
            actuals = (actual_durations[maximum_lead >= lead_length] >=
                       lead_length)
            metrics.append(compute_metrics_for_binary_outcome(
                actuals,
                predictions[:, lead_length - 1][maximum_lead >= lead_length],
                threshold_positive=threshold_positive,
                share_positive=share_positive))
        metrics = pd.DataFrame(metrics, index=lead_lengths)
        metrics.index.name = 'Lead Length'
        metrics['Other Metrics:'] = ''
        concordance_index_value = concordance_index(
            actual_durations,
            np.sum(predictions, axis=-1),
            self.data[subset][self.event_col])
        metrics['C-Index'] = np.where(metrics.index == 1,
                                      concordance_index_value, '')
        return metrics

    def tabulate_survival_by_quantile(
            self, subset: Union[None, pd.core.series.Series] = None,
            n_quantiles: int = 5,
            predict_beyond_subset: bool = True) -> pd.core.frame.DataFrame:
        """Count number survived, grouped by survival probability.

        Args:
            subset: A Boolean Series that is True for observations over which
                the counts will be computed. If None, default to all
                observations.
            n_quantiles: Number of groups. Groups will contain the same number
                of observations except to keep observations with the same
                predicted probability in the same group. Quantile 1 represents
                the individuals least likely to survive the given lead length.
            predict_beyond_subset: Whether or not predictions for time
                horizons that extend beyond the last period of the given subset
                will be evaluated.

        Returns:
            A DataFrame where each row contains the group size and number
            survived for each combination of lead length and group.
        """
        subset = default_subset_to_all(subset, self.data)
        predictions = self.predict(subset=subset, cumulative=True)
        actual_durations = self.data[subset][self.duration_col]
        factorized_time_ids = pd.factorize(
            self.data[self.config['TIME_IDENTIFIER']], sort=True)[0]
        if predict_beyond_subset:
            maximum_period = factorized_time_ids.max()
        else:
            maximum_period = factorized_time_ids[subset].max()
        maximum_lead = maximum_period - factorized_time_ids[subset]
        lead_lengths = np.arange(self.n_intervals) + 1
        counts = []
        for lead_length in lead_lengths:
            actuals = (actual_durations[maximum_lead >= lead_length] >=
                       lead_length)
            quantiles = pd.qcut(
                predictions[:, lead_length - 1][maximum_lead >= lead_length],
                n_quantiles, labels=False, duplicates='drop')
            actuals = actuals.groupby(quantiles).agg(['sum', 'count'])
            actuals['Lead Length'] = lead_length
            counts.append(actuals)
        counts = pd.concat(counts)
        counts.index.name = 'Prediction Quantile'
        counts.columns = ['Number Survived', 'Observations', 'Lead Length']
        counts = counts.reset_index()
        counts = counts[['Lead Length', 'Prediction Quantile',
                         'Number Survived', 'Observations']]
        counts['Prediction Quantile'] = counts['Prediction Quantile'] + 1
        return counts

    def tabulate_retention_rates(
            self, lead_periods: int,
            time_ids: Union[None, pd.core.series.Series] = None):
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
        factorized_time_ids = (pd.factorize(
            self.data[self.config['TIME_IDENTIFIER']], sort=True)[0])
        if time_ids is None:
            time_ids = factorized_time_ids
        person_ids = self.data[self.config['INDIVIDUAL_IDENTIFIER']]
        durations = ((pd.Series(factorized_time_ids).groupby(person_ids)
                      .transform(max)) - factorized_time_ids)
        actual_retention_rate = ((durations >= lead_periods).groupby(time_ids)
                                 .mean())
        forecasted_retention_rate = \
            (pd.Series(self.predict()[:, lead_periods - 1])
             .groupby(time_ids).mean())
        retention_rates = pd.DataFrame([actual_retention_rate,
                                        forecasted_retention_rate]).T
        retention_rates.columns = ['Actual Retention Rate',
                                   'Fitted Retention Rate']
        retention_rates['Actual Retention Rate'].iloc[-lead_periods:] = np.nan
        period_length = retention_rates.index[1] - retention_rates.index[0]
        for _ in range(lead_periods):
            retention_rates.loc[
                retention_rates.index.max() + period_length] = np.nan
        retention_rates = retention_rates.shift(lead_periods)
        retention_rates['Forecast Retention Rate'] = np.nan
        untrained_periods = lead_periods + self.config.get('TEST_PERIODS', 0)
        retention_rates['Forecast Retention Rate'
                        ].iloc[-untrained_periods:] = \
            retention_rates['Fitted Retention Rate'].iloc[-untrained_periods:]
        retention_rates['Fitted Retention Rate'
                        ].iloc[-untrained_periods:] = np.nan
        retention_rates.loc[
            retention_rates.index.max() + period_length] = np.nan
        retention_rates.index.name = 'Period'
        return retention_rates

    def hyperoptimize(self,
                      n_trials: int = 64,
                      rolling_validation: bool = True,
                      train_subset: Union[None, pd.core.series.Series] = None) -> dict:
        """Search for hyperparameters with greater out-of-sample performance."""

    def compute_shap_values(
            self, subset: Union[None, pd.core.series.Series] = None) -> dict:
        """Compute SHAP values by lead length, observation, and feature."""

    def compute_model_uncertainty(
            self, subset: Union[None, pd.core.series.Series] = None,
            n_iterations: int = 200) -> np.ndarray:
        """Produce predictions of models modified after training."""
