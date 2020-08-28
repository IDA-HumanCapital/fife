"""Conduct unit testing for fife.processors module."""

from fife import processors
import numpy as np
import pandas as pd
import pytest


def test_DataProcessor(setup_config, setup_dataframe):
    """Test that DataProcessor binds config and data arguments."""
    data_processor = processors.DataProcessor(setup_config, setup_dataframe)
    assert data_processor.config == setup_config and data_processor.data.equals(
        setup_dataframe
    )


def test_is_degenerate(setup_config, setup_dataframe):
    """Test that is_degenerate identifies an invariant feature as degenerate."""
    data_processor = processors.DataProcessor(config=setup_config, data=setup_dataframe)
    assert data_processor.is_degenerate("constant_categorical_var")


def test_is_categorical(setup_config, setup_dataframe):
    """Test that is_categorical correctly identifies
    categorical and numeric features.
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
    for col in setup_cat_cols:
        assert data_processor.is_categorical(col)
    for col in setup_num_cols:
        assert ~data_processor.is_categorical(col)


def test_PanelDataProcessor(setup_config, setup_dataframe):
    """Test that PanelDataProcessor binds config and data arguments."""
    data_processor = processors.PanelDataProcessor(
        config=setup_config, data=setup_dataframe
    )
    assert data_processor.config == setup_config and data_processor.data.equals(
        setup_dataframe
    )


if False:

    def test_process_all_columns(setup_config, setup_dataframe):
        """Test that PanelDataProcessor.process_all_columns() replaces the data
        attribute of PanelDataProcessor instance with a Pandas Dataframe."""
        errors_list = []
        for parallelize in [True, False]:
            data_processor = processors.PanelDataProcessor(
                config=setup_config, data=setup_dataframe
            )
            data_processor.process_all_columns(parallelize=parallelize)
            if not isinstance(data_processor.data, pd.DataFrame):
                errors_list.append(
                    f"Data attribute returned when parallelize={parallelize} "
                    "is not an instance of pd.DataFrame."
                )
        assert not errors_list, "Errors occurred: \n{}".format("\n".join(errors_list))


def test_process_single_column(setup_config, setup_dataframe):
    """Test that PanelDataProcessor.process_single_column() drops degenerate
    columns, correctly casts categorical columns, and does not modify individual
    identifier column."""
    errors_list = []
    indiv_id_col = setup_config["INDIVIDUAL_IDENTIFIER"]
    degenerate_cols = [
        col
        for col in setup_dataframe
        if (setup_dataframe[col].isnull().all()) | (setup_dataframe[col].nunique() < 2)
    ]
    categorical_cols = [
        col
        for col in setup_dataframe
        if ("categorical_var" in col) & (col not in degenerate_cols)
    ]
    data_processor = processors.PanelDataProcessor(
        config=setup_config, data=setup_dataframe
    )
    processed_col = data_processor.process_single_column(indiv_id_col)
    if not processed_col.equals(setup_dataframe[indiv_id_col]):
        errors_list.append("Individual identifier column {indiv_id_col} modified.")
    for degenerate_col in degenerate_cols:
        processed_col = data_processor.process_single_column(degenerate_col)
        if processed_col is not None:
            errors_list.append(
                f"Degenerate column {degenerate_col} not dropped from dataframe."
            )
    for categorical_col in categorical_cols:
        processed_col = data_processor.process_single_column(categorical_col)
        if not isinstance(processed_col.dtype, pd.api.types.CategoricalDtype):
            errors_list.append(
                f"Categorical column {categorical_col} not cast to categorical dtype."
            )
    assert not errors_list, "Errors occurred: \n{}".format("\n".join(errors_list))


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
