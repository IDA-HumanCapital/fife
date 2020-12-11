"""Define pytest fixtures for FIFE unit testing."""

import datetime as dt

from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def setup_config():
    """Create configuration dictionary of model parameters."""
    config_dict = {
        "BATCH_SIZE": 512,
        "CATEGORICAL_SUFFIXES": [],
        "DATA_FILE_NAME": "Input_Data",
        "DATA_FILE_PATH": "",
        "DENSE_LAYERS": 2,
        "DROPOUT_SHARE": 0.25,
        "EMBED_EXPONENT": 0,
        "EMBED_L2_REG": 2.0,
        "FILE_SUFFIX": "",
        "FIXED_EFFECT_FEATURES": [],
        "INDIVIDUAL_IDENTIFIER": "SSNSCR",
        "MAX_EPOCHS": 256,
        "MAX_NULL_SHARE": 0.999,
        "MAX_UNIQUE_NUMERIC_CATS": 1024,
        "MIN_SURVIVORS_IN_TRAIN": 64,
        "NON_CAT_MISSING_VALUE": -1,
        "NOTES_FOR_LOG": "No config notes specified",
        "NODES_PER_DENSE_LAYER": 512,
        "NUMERIC_SUFFIXES": [],
        "PATIENCE": 4,
        "PROPORTIONAL_HAZARDS": False,
        "QUANTILES": 5,
        "RETENTION_INTERVAL": 1,
        "SEED": 9999,
        "SHAP_PLOT_ALPHA": 0.5,
        "SHAP_SAMPLE_SIZE": 128,
        "TEST_PERIODS": 0,
        "TIME_IDENTIFIER": "FILE_DATE",
        "TREE_MODELS": True,
        "VALIDATION_SHARE": 0.25,
    }
    return config_dict


@pytest.fixture
def fabricate_forecasts():
    """Create faux forecasts for testing."""
    np.random.seed(9999)
    actual_array = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1])
    predicted_array = pd.Series(np.arange(0.1, 0.99, 0.1))
    faux_forecasts = dict()
    faux_forecasts["AUROC=1"] = [actual_array, predicted_array]
    faux_forecasts["AUROC=0"] = [
        actual_array,
        predicted_array.sort_values(ascending=False),
    ]
    faux_forecasts["empty actual"] = [pd.Series([]), predicted_array]
    return faux_forecasts


@pytest.fixture
def setup_dataframe():
    """Create unbalanced panel dataframe with various feature types."""
    # Specify parameters of data
    np.random.seed(9999)
    num_individuals = 1000

    base_date = dt.date(2000, 1, 1)
    max_months_from_base_date = 100
    mean_service_months = 48
    min_service_months = 1
    trend_after_date = base_date + relativedelta(
        months=round(max_months_from_base_date / 2)
    )

    values = []
    for i in np.arange(num_individuals):

        # Determine start date for individual
        indiv_first_obs_months_from_base_date = np.random.randint(
            max_months_from_base_date
        )
        add_years, add_months = divmod(indiv_first_obs_months_from_base_date, 12)
        start_date = base_date + relativedelta(years=add_years, months=add_months)

        # Determine separation date for individual
        max_service_months = (
            max_months_from_base_date - indiv_first_obs_months_from_base_date
        )
        if mean_service_months > max_service_months:
            actual_service_months = max_service_months
        else:
            actual_service_months = round(
                np.random.triangular(
                    min_service_months, mean_service_months, max_service_months
                )
            )
        separation_date = start_date + relativedelta(months=actual_service_months)

        # Specify variables that are assigned once per individual
        nonmixed_categorical_var = np.random.choice(["A", "B", "C", "D", "E", "F"])
        consistent_mixed_categorical_var = np.random.choice(
            ["E1", "E2", "E3", "O4", "O5", "O6"]
        )
        inconsistent_mixed_categorical_var = np.random.choice(
            ["Q", "R", "S", "T", 1, 2, 3]
        )
        completely_null_var = np.NaN
        partially_null_categorical_var = np.random.choice(
            ["FF", "GG", "HH", "II", "JJ"]
        )
        nonneg_trend_numeric_var = np.random.uniform(low=1, high=10)
        constant_categorical_var = "Z"

        date = start_date
        while date <= separation_date:

            # Specify variables assigned to each individual in every period
            nonneg_uniform_numeric_var = np.random.uniform(low=0, high=250000)
            nonneg_normal_numeric_var = np.random.normal(loc=45, scale=10)
            nonneg_triangular_numeric_var = np.random.triangular(10, 50, 90)
            neg_normal_numeric_var = np.random.normal(loc=-5, scale=20)
            nonneg_binomial_numeric_var = np.random.binomial(1, 0.65)
            duplicate_nonneg_uniform_numeric_var = nonneg_uniform_numeric_var
            partially_null_numeric_var = np.random.choice(
                [10, 20, np.NaN, 40, 50, 60, np.NaN, 80]
            )
            if np.random.binomial(1, 0.01) == 1:
                partially_null_categorical_var = np.NaN
            if np.random.binomial(1, 0.8) == 1 and partially_null_numeric_var == 40:
                partially_null_numeric_var = np.NaN

            # Create record
            values.append(
                [
                    i + 1,
                    date,
                    nonneg_uniform_numeric_var,
                    nonneg_normal_numeric_var,
                    nonneg_triangular_numeric_var,
                    neg_normal_numeric_var,
                    nonneg_binomial_numeric_var,
                    nonmixed_categorical_var,
                    consistent_mixed_categorical_var,
                    inconsistent_mixed_categorical_var,
                    completely_null_var,
                    partially_null_categorical_var,
                    partially_null_numeric_var,
                    nonneg_trend_numeric_var,
                    constant_categorical_var,
                    duplicate_nonneg_uniform_numeric_var,
                ]
            )

            # Specify interactions among variables
            if nonneg_uniform_numeric_var > 200000:
                neg_normal_numeric_var -= 3
                consistent_mixed_categorical_var = "O6"
            if nonneg_uniform_numeric_var < 50000:
                nonneg_normal_numeric_var += 10
            if nonneg_normal_numeric_var < 25 or nonneg_normal_numeric_var > 45:
                nonneg_triangular_numeric_var /= 2
                if nonneg_triangular_numeric_var < 12:
                    inconsistent_mixed_categorical_var = "Q"
                elif nonneg_triangular_numeric_var < 14:
                    inconsistent_mixed_categorical_var = "S"
            if nonneg_triangular_numeric_var > 65:
                if neg_normal_numeric_var < 0:
                    nonmixed_categorical_var = "C"
                elif nonneg_binomial_numeric_var == 1:
                    nonmixed_categorical_var = "F"
                elif partially_null_numeric_var == np.NaN:
                    nonmixed_categorical_var = "B"
                    partially_null_categorical_var = "GG"
            if consistent_mixed_categorical_var in ["E1", "O4"]:
                if partially_null_numeric_var == np.NaN:
                    nonneg_uniform_numeric_var = 135000
                elif nonmixed_categorical_var == "E":
                    nonneg_normal_numeric_var = 22

            # Add time series trend
            if date > trend_after_date:
                nonneg_uniform_numeric_var += 100
                nonneg_normal_numeric_var += 0.08
                nonneg_triangular_numeric_var -= 0.05
                nonneg_trend_numeric_var += 3

            # Increment date
            date += relativedelta(months=1)

    # Create dataframe with all records
    col_names = [
        "SSNSCR",
        "FILE_DATE",
        "nonneg_uniform_numeric_var",
        "nonneg_normal_numeric_var",
        "nonneg_triangular_numeric_var",
        "neg_normal_numeric_var",
        "nonneg_binomial_numeric_var",
        "nonmixed_categorical_var",
        "consistent_mixed_categorical_var",
        "inconsistent_mixed_categorical_var",
        "completely_null_var",
        "partially_null_categorical_var",
        "partially_null_numeric_var",
        "nonneg_trend_numeric_var",
        "constant_categorical_var",
        "duplicate_nonneg_uniform_numeric_var",
    ]
    data = pd.DataFrame(values, columns=col_names)

    # Add validation, test, and predict_obs flags
    data["_validation"] = pd.Series(
        np.random.choice([True, False], size=len(data), p=[0.1, 0.9])
    )
    data["_test"] = (
        pd.Series(np.random.choice([True, False], size=len(data), p=[0.1, 0.9]))
        & ~data["_validation"]
    )
    data["_predict_obs"] = (
        pd.Series(np.random.choice([True, False], size=len(data), p=[0.1, 0.9]))
        & ~data["_validation"]
        & ~data["_test"]
    )

    # Add period column
    data["_period"] = pd.factorize(data["FILE_DATE"], sort=True)[0]

    # Add maximum lead column
    data["_maximum_lead"] = (
        data.groupby("_test")["_period"].transform("max") - data["_period"]
    )

    # Add survival duration column
    data = data.sort_values(["SSNSCR", "FILE_DATE"])
    gaps = data.groupby("SSNSCR")["_period"].shift() < data["_period"] - 1
    spells = gaps.groupby(data["SSNSCR"]).cumsum()
    data["_duration"] = spells.groupby([data["SSNSCR"], spells]).cumcount(
        ascending=False
    )

    # Add event observed flag
    data["_event_observed"] = (
        data.groupby(data["SSNSCR"])["FILE_DATE"].transform("max")
        < data["FILE_DATE"].max()
    )

    # Add spell column
    data["_spell"] = 0

    return data


@pytest.fixture
def setup_ffnnm_dataframe():
    """Create unbalanced panel for FeedforwardNeuralNetworkModeler."""
    # Specify parameters of data
    np.random.seed(9999)
    num_individuals = 1000

    base_date = dt.date(2000, 1, 1)
    max_months_from_base_date = 100
    mean_service_months = 48
    min_service_months = 1

    values = []
    for i in np.arange(num_individuals):

        # Determine start date for individual
        indiv_first_obs_months_from_base_date = np.random.randint(
            max_months_from_base_date
        )
        add_years, add_months = divmod(indiv_first_obs_months_from_base_date, 12)
        start_date = base_date + relativedelta(years=add_years, months=add_months)

        # Determine separation date for individual
        max_service_months = (
            max_months_from_base_date - indiv_first_obs_months_from_base_date
        )
        if mean_service_months > max_service_months:
            actual_service_months = max_service_months
        else:
            actual_service_months = round(
                np.random.triangular(
                    min_service_months, mean_service_months, max_service_months
                )
            )
        separation_date = start_date + relativedelta(months=actual_service_months)

        # Specify variables that are assigned once per individual
        nonmixed_categorical_var = np.random.choice(["A", "B", "C", "D", "E", "F"])
        consistent_mixed_categorical_var = np.random.choice(
            ["E1", "E2", "E3", "O4", "O5", "O6"]
        )

        date = start_date
        while date <= separation_date:

            # Specify variables assigned to each individual in every period
            nonneg_normal_numeric_var = np.random.normal(loc=0, scale=0.1)
            nonneg_triangular_numeric_var = np.random.triangular(0, 0.5, 1)

            # Create record
            values.append(
                [
                    i + 1,
                    date,
                    nonneg_normal_numeric_var,
                    nonneg_triangular_numeric_var,
                    nonmixed_categorical_var,
                    consistent_mixed_categorical_var,
                ]
            )

            # Increment date
            date += relativedelta(months=1)

    # Create dataframe with all records
    col_names = [
        "SSNSCR",
        "FILE_DATE",
        "nonneg_normal_numeric_var",
        "nonneg_triangular_numeric_var",
        "nonmixed_categorical_var",
        "consistent_mixed_categorical_var",
    ]
    data = pd.DataFrame(values, columns=col_names)

    # Add validation, test, and predict_obs flags
    data["_validation"] = pd.Series(
        np.random.choice([True, False], size=len(data), p=[0.1, 0.9])
    )
    data["_test"] = (
        pd.Series(np.random.choice([True, False], size=len(data), p=[0.1, 0.9]))
        & ~data["_validation"]
    )
    data["_predict_obs"] = (
        pd.Series(np.random.choice([True, False], size=len(data), p=[0.1, 0.9]))
        & ~data["_validation"]
        & ~data["_test"]
    )

    # Add period column
    data["_period"] = pd.factorize(data["FILE_DATE"], sort=True)[0]

    # Add maximum lead column
    data["_maximum_lead"] = (
        data.groupby("_test")["_period"].transform("max") - data["_period"]
    )

    # Add survival duration column
    data = data.sort_values(["SSNSCR", "FILE_DATE"])
    gaps = data.groupby("SSNSCR")["_period"].shift() < data["_period"] - 1
    spells = gaps.groupby(data["SSNSCR"]).cumsum()
    data["_duration"] = spells.groupby([data["SSNSCR"], spells]).cumcount(
        ascending=False
    )

    # Add event observed flag
    data["_event_observed"] = (
        data.groupby(data["SSNSCR"])["FILE_DATE"].transform("max")
        < data["FILE_DATE"].max()
    )

    # Add spell column
    data["_spell"] = 0

    # Factorize categorical features
    data["nonmixed_categorical_var"], _ = pd.factorize(
        data["nonmixed_categorical_var"], sort=True
    )
    data["consistent_mixed_categorical_var"], _ = pd.factorize(
        data["consistent_mixed_categorical_var"]
    )

    return data
