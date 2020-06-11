"""Conduct unit testing for fife.processors module."""

from fife import processors
import numpy as np
import pandas as pd
import pytest


def test_deduplicate_column_values(setup_dataframe):
    """Test that deduplicate_column_values removes duplicate columns."""
    errors_list = []
    data = processors.deduplicate_column_values(setup_dataframe)
    if data.T.duplicated().any():
        errors_list.append("Duplicate columns remain.")
    if (
        "nonneg_uniform_numeric_var" in data.columns
        and "duplicate_nonneg_uniform_numeric_var" in data.columns
    ):
        errors_list.append("Named duplicate columns remain.")
    assert not errors_list, "Errors occurred: \n{}".format("\n".join(errors_list))


def test_normalize_numeric_feature(setup_dataframe):
    """Test that normalize_numeric_feature returns normalized numeric columns
    and dataframe with numeric ranges.
    """
    errors_list = []
    df_numeric_vars = setup_dataframe.select_dtypes(include=["number"])
    test_cols = [x for x in df_numeric_vars.columns if x not in ["SSNSCR", "FILE_DATE"]]
    for col in test_cols:
        if col not in ["completely_null_var", "partially_null_numeric_var"]:
            excluded_obs = pd.Series([False] * len(df_numeric_vars[col]))
            normalized_col, numeric_range = processors.normalize_numeric_feature(
                df_numeric_vars[col], excluded_obs
            )
            if not normalized_col.between(-0.5, 0.5).all():
                errors_list.append(
                    f"Normalized values outside range " f"[-0.5, 0.5] for column {col}."
                )
            if not numeric_range[0] == df_numeric_vars[col].min():
                errors_list.append(
                    f"Returned minimum value does not match "
                    f"actual minimum value for column {col}."
                )
            if not numeric_range[1] == df_numeric_vars[col].max():
                errors_list.append(
                    f"Returned maximum value does not match "
                    f"actual maximum value for column {col}."
                )
    assert not errors_list, "Errors occurred: \n{}".format("\n".join(errors_list))


def test_process_numeric_feature(setup_dataframe):
    """Test that process_numeric_feature function scales numeric columns
    to range [-0.5, 0.5].
    """
    errors_list = []
    df_numeric_vars = setup_dataframe.select_dtypes(include=["number"])
    test_cols = [x for x in df_numeric_vars.columns if x not in ["SSNSCR", "FILE_DATE"]]
    for col in test_cols:
        if col in ["completely_null_var", "partially_null_numeric_var"]:
            df_numeric_vars[col] = df_numeric_vars[col].fillna(0)
        feature_min = df_numeric_vars[col].min()
        feature_max = df_numeric_vars[col].max()
        scaled_col = processors.process_numeric_feature(
            df_numeric_vars[col], feature_min, feature_max
        )
        if not scaled_col.between(-0.5, 0.5).all() and col != "completely_null_var":
            errors_list.append(
                f"Returned values outside range [-0.5, 0.5] " f"for column {col}."
            )
    assert not errors_list, "Errors occurred: \n{}".format("\n".join(errors_list))


def test_factorize_categorical_feature(setup_dataframe):
    """Test that factorize_categorical_feature function returns a factorized
    categorical column and corresponding map.
    """
    errors_list = []
    df_categorical_vars = setup_dataframe.select_dtypes(exclude=["number"])
    test_cols = [
        x for x in df_categorical_vars.columns if x not in ["SSNSCR", "FILE_DATE"]
    ]
    for col in test_cols:
        exclude_obs = pd.Series([False] * len(df_categorical_vars[col]))
        factorized_col, cat_map = processors.factorize_categorical_feature(
            df_categorical_vars[col], exclude_obs
        )
        if len(factorized_col) != len(df_categorical_vars[col]):
            errors_list.append(
                f"Length of returned column {col} does not "
                f"match original column length."
            )
        if not pd.api.types.is_numeric_dtype(factorized_col):
            errors_list.append(f"Returned column {col} is not a numeric " f"data type.")
        if not isinstance(cat_map, dict):
            errors_list.append(f"Returned cat_map is not a dictionary.")
        if len(set(cat_map)) < df_categorical_vars[col].nunique():
            errors_list.append(
                f"Returned cat_map for column {col} has "
                f"fewer unique values than original column."
            )
        if len(set(cat_map)) < factorized_col.nunique():
            errors_list.append(
                f"Returned cat_map for column {col} has "
                f"fewer unique values than factorized column."
            )
    assert not errors_list, "Errors occurred: \n{}".format("\n".join(errors_list))


def test_produce_categorical_map(setup_dataframe):
    """Test that produce_categorical_map returns a dictionary mapping
    categorical values to integers.
    """
    errors_list = []
    df_categorical_vars = setup_dataframe.select_dtypes(exclude=["number"])
    test_cols = [
        x for x in df_categorical_vars.columns if x not in ["SSNSCR", "FILE_DATE"]
    ]
    for col in test_cols:
        cat_map = processors.produce_categorical_map(df_categorical_vars[col])
        if not isinstance(cat_map, dict):
            errors_list.append(f"Returned cat_map is not a dictionary.")
        if len(set(cat_map)) < df_categorical_vars[col].nunique():
            errors_list.append(
                f"Returned cat_map for column {col} has "
                f"fewer unique values than original column."
            )
    assert not errors_list, "Errors occurred: \n{}".format("\n".join(errors_list))


def test_process_categorical_feature(setup_dataframe):
    """Test that process_categorical_feature maps categorical values to
    unsigned integers.
    """
    errors_list = []
    df_categorical_vars = setup_dataframe.select_dtypes(exclude=["number"])
    test_cols = [
        x for x in df_categorical_vars.columns if x not in ["SSNSCR", "FILE_DATE"]
    ]
    for col in test_cols:
        _, cat_map = pd.factorize(df_categorical_vars[col], sort=True)
        cat_map = {val: (i + 1) for i, val in enumerate(cat_map)}
        cat_map[np.nan] = len(cat_map)
        factorized_col = processors.process_categorical_feature(
            df_categorical_vars[col], cat_map
        )
        if len(factorized_col) != len(df_categorical_vars[col]):
            errors_list.append(
                f"Length of returned column {col} does not "
                f"match original column length."
            )
        if not pd.api.types.is_numeric_dtype(factorized_col):
            errors_list.append(f"Returned column {col} is not a numeric " f"data type.")
    assert not errors_list, "Errors occurred: \n{}".format("\n".join(errors_list))


def test_DataProcessor(setup_config, setup_dataframe):
    """Test that DataProcessor binds config and data arguments."""
    data_processor = processors.DataProcessor(setup_config, setup_dataframe)
    assert data_processor.config == setup_config and data_processor.data.equals(
        setup_dataframe
    )


def test_drop_degenerate_features(setup_config, setup_dataframe):
    """Test that drop_degenerate_features drops an invariant feature."""
    data_processor = processors.DataProcessor(config=setup_config, data=setup_dataframe)
    data = data_processor.drop_degenerate_features()
    assert "constant_var" not in data.columns


def test_identify_categorical_features(setup_config, setup_dataframe):
    """Test that identify_categorical_features correctly identifies
    categorical and numeric sets of features.
    """
    reserved_cat_cols = [
        "_duration",
        "_event_observed",
        "_predict_obs",
        "_test",
        "_validation",
        "_period",
        "_maximum_lead",
    ]
    reserved_num_cols = ["FILE_DATE"]
    non_reserved_cols = [
        x
        for x in setup_dataframe.columns
        if x not in reserved_cat_cols + reserved_num_cols
    ]
    setup_cat_cols = [x for x in non_reserved_cols if "categorical" in x] + [
        "completely_null_var"
    ]
    setup_num_cols = [x for x in non_reserved_cols if "numeric" in x]
    num_vars_treated_as_cat_vars = []
    for num_col in setup_num_cols:
        if (
            setup_dataframe[num_col].nunique()
            <= setup_config["MAX_UNIQUE_NUMERIC_CATS"]
        ):
            num_vars_treated_as_cat_vars.append(num_col)
    setup_cat_cols = setup_cat_cols + num_vars_treated_as_cat_vars + reserved_cat_cols
    setup_num_cols = [
        x for x in setup_num_cols if x not in num_vars_treated_as_cat_vars
    ] + reserved_num_cols
    data_processor = processors.DataProcessor(config=setup_config, data=setup_dataframe)
    cat_cols, numeric_cols = data_processor.identify_categorical_features()
    assert set(setup_cat_cols) == set(cat_cols)
    assert set(setup_num_cols) == set(numeric_cols)


def test_PanelDataProcessor(setup_config, setup_dataframe):
    """Test that PanelDataProcessor binds config and data arguments."""
    data_processor = processors.PanelDataProcessor(
        config=setup_config, data=setup_dataframe
    )
    assert data_processor.config == setup_config and data_processor.data.equals(
        setup_dataframe
    )


def test_check_panel_consistency(setup_config, setup_dataframe):
    """Test that check_panel_consistency raises an error if an
    observation is duplicated.
    """
    data_processor = processors.PanelDataProcessor(
        config=setup_config, data=setup_dataframe
    )
    data_processor.data = data_processor.data.append(data_processor.data.iloc[[1], :])
    with pytest.raises(AssertionError):
        data_processor.check_panel_consistency()


def test_sort_panel_data(setup_config, setup_dataframe):
    """Test that sort_panel_data re-sorts swapped observations."""
    data_processor = processors.PanelDataProcessor(
        config=setup_config, data=setup_dataframe
    )
    first_two_rows_already_sorted = data_processor.data.iloc[[1, 2], :].copy()
    data_processor.data.iloc[[1, 2], :] = data_processor.data.iloc[[2, 1], :]
    data_processor.data = data_processor.sort_panel_data()
    assert data_processor.data.iloc[[1, 2], :].equals(first_two_rows_already_sorted)


def test_flag_validation_individuals(setup_config, setup_dataframe):
    """Test that validation set is given share of observations and contains
    all observations of each individual therein.
    """
    data_processor = processors.PanelDataProcessor(
        config=setup_config, data=setup_dataframe
    )
    error_tolerance = 0.1
    data_processor.data["validation"] = data_processor.flag_validation_individuals()
    share_in_validation_sample = np.mean(data_processor.data["validation"])
    share_approximately_correct = (
        (data_processor.config["VALIDATION_SHARE"] - error_tolerance)
        <= share_in_validation_sample
    ) and (
        share_in_validation_sample
        <= (data_processor.config["VALIDATION_SHARE"] + error_tolerance)
    )
    rates_individuals_within_validation_group = data_processor.data.groupby(
        data_processor.config["INDIVIDUAL_IDENTIFIER"]
    )["validation"].mean()
    individual_consistently_in_validation_group = (
        rates_individuals_within_validation_group == 1
    ) | (rates_individuals_within_validation_group == 0)
    assert share_approximately_correct
    assert np.mean(individual_consistently_in_validation_group) == 1


SEED = 9999
np.random.seed(SEED)
