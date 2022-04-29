"""Data processing functions and classes for FIFE."""

from typing import List, Tuple, Union

import dask
import numpy as np
import pandas as pd

from warnings import warn


def check_column_consistency(data: pd.core.frame.DataFrame, colname: str) -> None:
    """Assert column exists, has no missing values, and is not constant."""
    assert colname in data, f"{colname} not in data"
    assert all(data[colname].notnull()), f"{colname} has missing values"
    assert any(
        data[colname][1:].values != data[colname][:-1].values
    ), f"{colname} does not have multiple unique values"


def deduplicate_column_values(
    data: pd.core.frame.DataFrame, reserved_cols: List[str] = [], max_obs: int = 65536
) -> pd.core.frame.DataFrame:
    """Delete columns with the same values as a later column.

    Args:
        df: A DataFrame.
        reserved_cols: Names of columns to exclude from deduplication.
        max_obs: The number of observations to sample if df has more than that
            many observations.

    Returns:
        A DataFrame containing only the last instance of each unique column.
    """
    warn(
        "Warning: deduplicate_column_values is deprecated. "
        "If you wish to deduplicate columns, do so prior to initializing a processor.",
        DeprecationWarning,
    )
    comparison_data = data.drop(reserved_cols, axis=1).sample(
        n=min(max_obs, data.shape[0]), replace=False
    )
    deduplicated_cols = list(comparison_data.T.drop_duplicates(keep="last").index)
    deduplicated_data = data[reserved_cols + deduplicated_cols]
    duplicated_cols = [col for col in data if col not in deduplicated_data.columns]
    if duplicated_cols:
        print(
            f'{", ".join(duplicated_cols)} dropped for having identical '
            + "values as another feature"
        )
    return deduplicated_data


def normalize_numeric_feature(
    col: pd.core.series.Series, excluded_obs: Union[None, pd.core.series.Series] = None
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
    warn(
        "Warning: normalize_numeric_feature is deprecated. "
        "Modelers now use transform_features upon initialization instead.",
        DeprecationWarning,
    )
    if excluded_obs is None:
        col_subset = col[col.notnull()]
    else:
        col_subset = col[(~excluded_obs) & col.notnull()]
    if len(col_subset) > 0:
        minimum = np.nanmin(col_subset)
        maximum = np.nanmax(col_subset)
    else:
        minimum = np.nan
        maximum = np.nan
    normalized_col = process_numeric_feature(col, minimum, maximum)
    numeric_range = [minimum, maximum]
    return normalized_col, numeric_range


def process_numeric_feature(
    col: pd.core.series.Series, minimum: float, maximum: float
) -> pd.core.series.Series:
    """Scale numeric values to a given range.

    Args:
        col: A numeric Series.
        minimum: The value to map to -0.5.
        maximum: The value to map to 0.5.

    Returns:
        A Series of floats.
    """
    warn(
        "Warning: process_numeric_feature is deprecated. "
        "Modelers now use transform_features upon initialization instead.",
        DeprecationWarning,
    )
    return (col - minimum) / (maximum - minimum) - 0.5


def factorize_categorical_feature(
    col: pd.core.series.Series, excluded_obs: Union[None, pd.core.series.Series] = None
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
    warn(
        "Warning: factorize_categorical_feature is deprecated. "
        "Processors now convert categorical features to pandas Categorical type.",
        DeprecationWarning,
    )
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
    warn(
        "Warning: produce_categorical_map is deprecated. "
        "Processors now convert categorical features to pandas Categorical type.",
        DeprecationWarning,
    )
    _, cat_map = pd.factorize(col.values, sort=True)
    cat_map = {val: (i + 1) for i, val in enumerate(cat_map)}
    cat_map["unrecognized"] = 0
    cat_map[np.nan] = len(cat_map)
    return cat_map


def process_categorical_feature(
    col: pd.core.series.Series, cat_map: dict
) -> pd.core.frame.DataFrame:
    """Map categorical values to unsigned integers.

    Args:
        col: A pandas Series.
        cat_map: A dict containing a key for each unique value in col.

    Returns:
        A pandas Series of unsigned integers.
    """
    warn(
        "Warning: process_categorical_feature is deprecated. "
        "Processors now convert categorical features to pandas Categorical type.",
        DeprecationWarning,
    )
    col = col.map(cat_map).fillna(0)
    n_bits = 8
    while col.max() >= 2**n_bits:
        n_bits *= 2
    col = col.astype("uint" + str(n_bits))
    return col


class DataProcessor:
    """Prepare data by identifying features as degenerate or categorical."""

    def __init__(
        self,
        config: Union[None, dict] = {},
        data: Union[None, pd.core.frame.DataFrame] = None,
    ) -> None:
        """Initialize the DataProcessor.

        Args:
            config: A dictionary of configuration parameters.
            data: A DataFrame to be processed.
        """
        if (config.get("INDIVIDUAL_IDENTIFIER", "") == "") and data is not None:
            config["INDIVIDUAL_IDENTIFIER"] = data.columns[0]
            print(
                "Individual identifier column name not given; assumed to be "
                f'leftmost column ({config["INDIVIDUAL_IDENTIFIER"]})'
            )
        self.config = config
        self.data = data

    def is_degenerate(self, col: str) -> bool:
        """Determine if a feature is constant or has too many missing values."""
        if self.data[col].isnull().mean() >= self.config.get("MAX_NULL_SHARE", 0.999):
            return True
        if self.data[col].nunique(dropna=False) < 2:
            return True
        return False

    def is_categorical(self, col: str) -> bool:
        """Determine if the given feature should be processed as categorical, as opposed to numeric."""
        if col.endswith(tuple(self.config.get("CATEGORICAL_SUFFIXES", []))):
            if col.endswith(tuple(self.config.get("NUMERIC_SUFFIXES", []))):
                print(
                    f"{col} matches categorical and numeric suffixes; "
                    "identified as categorical"
                )
            return True
        if col in self.data.select_dtypes(
            include=["datetime"]
        ) or col == self.config.get("TIME_IDENTIFIER"):
            return False
        if col in self.data.select_dtypes(exclude=["number"]):
            if col.endswith(tuple(self.config.get("NUMERIC_SUFFIXES", []))):
                print(
                    f"{col} matches numeric suffix but is non-numeric; "
                    "identified as categorical"
                )
            return True
        if col.endswith(tuple(self.config.get("NUMERIC_SUFFIXES", []))) or (
            self.data[col].nunique() > self.config.get("MAX_UNIQUE_NUMERIC_CATS", 1024)
        ):
            return False
        return True


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

    def __init__(
        self,
        config: Union[None, dict] = {},
        data: Union[None, pd.core.frame.DataFrame] = None,
    ) -> None:
        """Initialize the PanelDataProcessor.

        Args:
            config: A dictionary of configuration parameters.
            data: A DataFrame to be processed.
        """
        if (config.get("TIME_IDENTIFIER", "") == "") and data is not None:
            config["TIME_IDENTIFIER"] = data.columns[1]
            print(
                "Time identifier column name not given; assumed to be "
                f'second-leftmost column ({config["TIME_IDENTIFIER"]})'
            )
        super().__init__(config, data)

    def build_processed_data(self, parallelize: bool = True) -> None:
        """Clean, augment, and store a panel dataset and related information.

        - Sort data by individual and time.
        - Drop degenerate features.
        - Label subsets for prediction, validation, and testing.
        - Compute survival duration and if departure is observed.
        - Store a subset of the raw input data from the final period.
        - Map categorical features to unsigned integers.
        - Scale numeric features.
        """
        self.check_panel_consistency()
        self.data = self.sort_panel_data()
        self.process_all_columns(parallelize=parallelize)
        self.build_reserved_cols()

    def process_all_columns(self, parallelize: bool = True) -> None:
        """Split, process, and merge all data columns."""
        data_dict = {}
        if parallelize:
            for colname in self.data:
                data_dict[colname] = dask.delayed(self.process_single_column)(colname)
            data_dict = dask.compute(data_dict)[0]
        else:
            for colname in self.data:
                data_dict[colname] = self.process_single_column(colname)
        data_dict = {key: val for key, val in data_dict.items() if val is not None}
        self.data = pd.DataFrame.from_dict(data_dict)

    def process_single_column(self, colname: str) -> Union[None, pd.core.series.Series]:
        """Apply data cleaning functions to an individual data column."""
        if colname == self.config["INDIVIDUAL_IDENTIFIER"]:
            return self.data[colname]
        elif self.is_degenerate(colname):
            print(f'Column "{colname}" is degenerate and will be dropped.')
            return None
        elif self.is_categorical(colname):
            return (
                self.data[colname]
                .astype("category")
                .cat.add_categories("NaN")
                .fillna("NaN")
            )
        else:
            return self.data[colname]

    def build_reserved_cols(self):
        """Add data split and outcome-related columns to the data."""
        self.data["_period"] = pd.factorize(
            self.data[self.config["TIME_IDENTIFIER"]], sort=True
        )[0]
        max_test_intervals = int((len(set(self.data["_period"])) - 1) / 2)
        test_intervals = self.config.get(
            "TEST_INTERVALS", self.config.get("TEST_PERIODS", 0) - 1
        )

        if test_intervals > max_test_intervals:
            warn(
                f"The specified value for TEST_INTERVALS of {test_intervals} was too high and will not allow for enough training periods. It was automatically reduced to {max_test_intervals}"
            )
            test_intervals = max_test_intervals

        self.data["_test"] = (self.data["_period"] + test_intervals) >= self.data[
            "_period"
        ].max()
        self.data["_validation"] = (
            self.flag_validation_individuals() & ~self.data["_test"]
        )
        observation_max_period = self.data.groupby("_test")["_period"].transform("max")
        self.data["_maximum_lead"] = (
            observation_max_period
            - self.data["_period"]
            + (observation_max_period < self.data["_period"].max())
        )
        del observation_max_period
        gaps = (
            self.data.groupby(self.config["INDIVIDUAL_IDENTIFIER"])["_period"].shift()
            < self.data["_period"] - 1
        )
        spells = gaps.groupby(self.data[self.config["INDIVIDUAL_IDENTIFIER"]]).cumsum()
        self.data["_duration"] = spells.groupby(
            [self.data[self.config["INDIVIDUAL_IDENTIFIER"]], spells]
        ).cumcount(ascending=False)
        self.data["_spell"] = spells
        self.data["_event_observed"] = (
            self.data["_duration"] < self.data["_maximum_lead"]
        )
        self.data["_predict_obs"] = (
            self.data["_period"] + test_intervals + 1
        ) == self.data["_period"].max()

    def check_panel_consistency(self) -> None:
        """Ensure observations have unique individual-period combinations."""
        check_column_consistency(self.data, self.config["INDIVIDUAL_IDENTIFIER"])
        check_column_consistency(self.data, self.config["TIME_IDENTIFIER"])
        ids = [self.config["INDIVIDUAL_IDENTIFIER"], self.config["TIME_IDENTIFIER"]]
        assert not any(self.data.duplicated(subset=ids)), (
            "One or more individuals have multiple observations for "
            "a single time value"
        )

    def sort_panel_data(self) -> pd.core.frame.DataFrame:
        """Sort the data by individual, then by period."""
        return self.data.sort_values(
            [self.config["INDIVIDUAL_IDENTIFIER"], self.config["TIME_IDENTIFIER"]]
        ).reset_index(drop=True)

    def flag_validation_individuals(self) -> pd.core.series.Series:
        """Flag observations from a random share of individuals."""
        unique_ids = self.data[self.config["INDIVIDUAL_IDENTIFIER"]].unique()
        size = int(self.config.get("VALIDATION_SHARE", 0.25) * unique_ids.shape[0])
        validation_ids = np.random.choice(unique_ids, size=size, replace=False)
        return self.data[self.config["INDIVIDUAL_IDENTIFIER"]].isin(validation_ids)
