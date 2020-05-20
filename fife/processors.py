"""Data processing functions and classes for FIFE."""

from typing import List, Tuple, Union

import numpy as np
import pandas as pd


def check_column_consistency(data: pd.core.frame.DataFrame,
                             colname: str) -> None:
    """Assert column exists, has no missing values, and is not constant."""
    assert colname in data, (
        f'{colname} not in data')
    assert all(data[colname].notnull()), (
        f'{colname} has missing values')
    assert any(data[colname][1:].values != data[colname][:-1].values), (
        f'{colname} does not have multiple unique values')


def deduplicate_column_values(data: pd.core.frame.DataFrame,
                              reserved_cols: List[str] = [],
                              max_obs: int = 65536) -> pd.core.frame.DataFrame:
    """Delete columns with the same values as a later column.

    Args:
        df: A DataFrame.
        reserved_cols: Names of columns to exclude from deduplication.
        max_obs: The number of observations to sample if df has more than that
            many observations.

    Returns:
        A DataFrame containing only the last instance of each unique column.
    """
    comparison_data = data.drop(reserved_cols, axis=1).sample(
        n=min(max_obs, data.shape[0]), replace=False)
    deduplicated_cols = list(
        comparison_data.T.drop_duplicates(keep='last').index)
    deduplicated_data = data[deduplicated_cols + reserved_cols]
    duplicated_cols = [col for col in data if
                       col not in deduplicated_data.columns]
    if duplicated_cols:
        print(f'{", ".join(duplicated_cols)} dropped for having identical ' +
              'values as another feature')
    return deduplicated_data


def normalize_numeric_feature(
        col: pd.core.series.Series,
        excluded_obs: Union[None, pd.core.series.Series] = None
        ) -> Tuple[pd.core.series.Series, List[float]]:
    """Scale numeric values to their empirical range.

    Args:
        col: A numeric Series.
        excluded_obs: True for observations to exclude when computing min/max.

    Returns:
        - A Series of floats.
        - A list containing the minimum and maximum values among the included
          observations.
    """
    if excluded_obs is None:
        minimum = np.nanmin(col)
        maximum = np.nanmax(col)
    else:
        minimum = np.nanmin(col[~excluded_obs])
        maximum = np.nanmax(col[~excluded_obs])
    normalized_col = process_numeric_feature(col, minimum, maximum)
    numeric_range = [minimum, maximum]
    return normalized_col, numeric_range


def process_numeric_feature(col: pd.core.series.Series,
                            minimum: float,
                            maximum: float) -> pd.core.series.Series:
    """Scale numeric values to a given range.

    Args:
        col: A numeric Series.
        minimum: The value to map to -0.5.
        maximum: The value to map to 0.5.

    Returns:
        A Series of floats.
    """
    return (col - minimum) / (maximum - minimum) - 0.5


def factorize_categorical_feature(
        col: pd.core.series.Series,
        excluded_obs: Union[None, pd.core.series.Series] = None
        ) -> Tuple[pd.core.series.Series, dict]:
    """Map categorical values to unsigned integers.

    Args:
        col: A Series.
        excluded_obs: True for observations to exclude from creating the map.

    Returns:
        - A pandas Series of unsigned integers.
        - A dictionary mapping each unique value among the included
          observations and np.nan to an integer and any other value not in the
          Series to 0.
    """
    if excluded_obs is None:
        cat_map = produce_categorical_map(col)
    else:
        cat_map = produce_categorical_map(col[~excluded_obs])
    factorized_col = process_categorical_feature(col, cat_map)
    return factorized_col, cat_map


def produce_categorical_map(col: pd.core.series.Series) -> dict:
    """Return a map from categorical values to unsigned integers.

    Zero is reserved for values not seen in data used to create map.

    Args:
        col: A Series.

    Returns:
        A dictionary mapping each unique value in the Series and np.nan to an
        integer and any other value not in the Series to zero.
    """
    _, cat_map = pd.factorize(col.values, sort=True)
    cat_map = {val: (i + 1) for i, val in enumerate(cat_map)}
    cat_map['unrecognized'] = 0
    cat_map[np.nan] = len(cat_map)
    return cat_map


def process_categorical_feature(col: pd.core.series.Series,
                                cat_map: dict) -> pd.core.frame.DataFrame:
    """Map categorical values to unsigned integers.

    Args:
        col: A pandas Series.
        cat_map: A dict containing a key for each unique value in col.

    Returns:
        A pandas Series of unsigned integers.
    """
    col = col.map(cat_map).fillna(0)
    n_bits = 8
    while col.max() >= 2 ** n_bits:
        n_bits *= 2
    col = col.astype('uint' + str(n_bits))
    return col


class DataProcessor:
    """Prepare data by identifying features as degenerate or categorical."""

    def __init__(self,
                 config: Union[None, dict] = None,
                 data: Union[None, pd.core.frame.DataFrame] = None) -> None:
        """Initialize the DataProcessor.

        Args:
            config: A dictionary of configuration parameters.
            data: A DataFrame to be processed.
        """
        self.config = config
        self.data = data

    def drop_degenerate_features(self) -> pd.core.frame.DataFrame:
        """Drop constant features or those with too many missing values."""
        thresh = int((1 - self.config['MAX_NULL_SHARE']) * self.data.shape[0])
        data = self.data.dropna(thresh=thresh, axis=1)
        for col in data.columns:
            if all(data[col][1:].values == data[col][:-1].values):
                del data[col]
                print(f'Feature {col} dropped for having zero variance')
        return data

    def identify_categorical_features(self) -> Tuple[List[str], List[str]]:
        """Split the list of features into categorical and numeric."""
        categorical_cols = []
        numeric_cols = []
        for col in self.data:
            if col == self.config.get('INDIVIDUAL_IDENTIFIER'):
                pass
            elif col.endswith(
                    tuple(self.config.get('CATEGORICAL_SUFFIXES', []))):
                categorical_cols.append(col)
                if col.endswith(
                        tuple(self.config.get('NUMERIC_SUFFIXES', []))):
                    print(f'{col} matches categorical and numeric suffixes; '
                          'identified as categorical')
            elif (col in self.data.select_dtypes(include=['datetime']) or
                  col == self.config.get('TIME_IDENTIFIER')):
                numeric_cols.append(col)
            elif col in self.data.select_dtypes(exclude=['number']):
                categorical_cols.append(col)
                if col.endswith(
                        tuple(self.config.get('NUMERIC_SUFFIXES', []))):
                    print(f'{col} matches numeric suffix but is non-numeric; '
                          'identified as categorical')
            elif (col.endswith(
                    tuple(self.config.get('NUMERIC_SUFFIXES', []))) or
                  (self.data[col].nunique() >
                   self.config['MAX_UNIQUE_NUMERIC_CATS'])):
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
        return categorical_cols, numeric_cols


class PanelDataProcessor(DataProcessor):
    """Ready panel data for modelling.

    Attributes:
        config (dict): User-provided configuration parameters.
        data (pd.core.frame.DataFrame): Processed panel data.
        raw_subset (pd.core.frame.DataFrame): An unprocessed sample from the
            final period of data. Useful for displaying meaningful values in
            SHAP plots.
        categorical_maps (dict): Contains for each categorical feature a map
            from each unique value to a whole number.
        numeric_ranges (pd.core.frame.DataFrame): Contains for each numeric
            feature the maximum and minimum value in the training set.
    """

    def build_processed_data(self) -> None:
        """Clean, augment, and store a panel dataset and related information.

        - Sort data by individual and time.
        - Drop degenerate features.
        - Label subsets for prediction, validation, and testing.
        - Compute survival duration and if departure is observed.
        - Store a subset of the raw input data from the final period.
        - Map categorical features to unsigned integers.
        - Scale numeric features.
        - Drop duplicated features.
        """
        self.check_panel_consistency()
        self.data = self.sort_panel_data()
        self.data = self.drop_degenerate_features()
        categorical_features, numeric_features = \
            self.identify_categorical_features()
        self.data = self.data[categorical_features
                              + numeric_features
                              + [self.config['INDIVIDUAL_IDENTIFIER']]]
        self.data['_predict_obs'] = self.flag_final_periods(1)
        self.data['_validation'] = self.flag_validation_individuals()
        if self.config.get('TEST_PERIODS', 0) > 0:
            self.data['_test'] = self.flag_final_periods(
                self.config['TEST_PERIODS'])
            self.data['_event_observed'] = self.flag_event_observed(
                group=self.data['_test'])
            self.data['_duration'] = self.compute_survival_duration(
                group=self.data['_test'])
        else:
            self.data['_test'] = False
            self.data['_event_observed'] = self.flag_event_observed()
            self.data['_duration'] = self.compute_survival_duration()
        sample_size = self.config.get('SHAP_SAMPLE_SIZE', 0)
        self.raw_subset = (self.data[self.data['_predict_obs']]
                           .sample(n=sample_size).sort_index())
        held_out_obs = (self.data['_validation'] | self.data['_test'] |
                        self.data['_predict_obs'])
        self.categorical_maps = {}
        for col in categorical_features:
            self.data[col], self.categorical_maps[col] = \
                factorize_categorical_feature(self.data[col], held_out_obs)
        numeric_ranges = {}
        for col in numeric_features:
            self.data[col], numeric_ranges[col] = \
                normalize_numeric_feature(
                    self.data[col],
                    self.data['_test'] | self.data['_validation'])
        self.numeric_ranges = pd.DataFrame.from_dict(
            numeric_ranges, orient='index', columns=['Minimum', 'Maximum'])
        self.data = deduplicate_column_values(
            self.data, ['_duration', '_event_observed',
                        '_predict_obs', '_test', '_validation',
                        self.config['INDIVIDUAL_IDENTIFIER']])

    def check_panel_consistency(self) -> None:
        """Ensure observations have unique individual-period combinations."""
        check_column_consistency(self.data,
                                 self.config['INDIVIDUAL_IDENTIFIER'])
        check_column_consistency(self.data,
                                 self.config['TIME_IDENTIFIER'])
        ids = [self.config['INDIVIDUAL_IDENTIFIER'],
               self.config['TIME_IDENTIFIER']]
        assert not any(self.data.duplicated(subset=ids)), (
            'One or more individuals have multiple observations for '
            'a single time value')

    def sort_panel_data(self) -> pd.core.frame.DataFrame:
        """Sort the data by individual, then by period."""
        return self.data.sort_values(
            [self.config['INDIVIDUAL_IDENTIFIER'],
             self.config['TIME_IDENTIFIER']]).reset_index(drop=True)

    def flag_final_periods(self, n_periods: int) -> pd.core.series.Series:
        """Flag observations from the most recent periods."""
        period_cutoff = pd.factorize(self.data[self.config['TIME_IDENTIFIER']],
                                     sort=True)[1][-n_periods]
        return self.data[self.config['TIME_IDENTIFIER']] >= period_cutoff

    def flag_validation_individuals(self) -> pd.core.series.Series:
        """Flag observations from a random share of individuals."""
        unique_ids = self.data[self.config['INDIVIDUAL_IDENTIFIER']].unique()
        size = int(self.config['VALIDATION_SHARE'] * unique_ids.shape[0])
        validation_ids = np.random.choice(unique_ids,
                                          size=size,
                                          replace=False)
        return (self.data[self.config['INDIVIDUAL_IDENTIFIER']]
                .isin(validation_ids))

    def flag_event_observed(self, group: pd.core.series.Series = None
                            ) -> pd.core.series.Series:
        """Flag observations from individuals observed to depart the data."""
        time_id = self.config['TIME_IDENTIFIER']
        individual_id = self.data[self.config['INDIVIDUAL_IDENTIFIER']]
        last_period = self.data.groupby(individual_id)[time_id].transform(max)
        if group is None:
            return last_period < self.data[time_id].max()
        return last_period < self.data.groupby(group)[time_id].transform(max)

    def compute_survival_duration(self, group: pd.core.series.Series = None
                                  ) -> pd.core.series.Series:
        """Count future observations of the individual at each observation."""
        factorized_time_ids = pd.factorize(
            self.data[self.config['TIME_IDENTIFIER']], sort=True)[0]
        groups = [self.data[self.config['INDIVIDUAL_IDENTIFIER']]]
        if group is not None:
            groups.append(group)
        return (pd.Series(factorized_time_ids).groupby(groups).transform(max)
                - factorized_time_ids)

    def process_new_data(self, new_data: pd.core.frame.DataFrame
                         ) -> pd.core.frame.DataFrame:
        """Apply existing categorical maps and numeric ranges to new data."""
        data = new_data.copy(deep=True)
        for col, categorical_map in self.categorical_maps:
            if col in data:
                data[col] = process_categorical_feature(data[col],
                                                        categorical_map)
        for col in self.numeric_ranges.index:
            if col in data:
                minimum = self.numeric_ranges.loc[col]['Minimum']
                maximum = self.numeric_ranges.loc[col]['Maximum']
                data[col] = process_numeric_feature(data[col],
                                                    minimum, maximum)
        return data
