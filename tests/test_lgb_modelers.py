"""Conduct unit testing for fife.lgb_modelers module."""

from fife import lgb_modelers
import lightgbm as lgb
import numpy as np
import pandas as pd


def test_gbtm_init():
    """Test that GradientBoostedTreesModeler can be instantiated
    without arguments.
    """
    errors_list = []
    try:
        modeler = lgb_modelers.GradientBoostedTreesModeler()
        if not vars(modeler):
            errors_list.append(
                "GradientBoostedTreesModeler instantiated "
                "object does not have attributes."
            )
    except Exception:
        errors_list.append("GradientBoostedTreesModeler could not " "be instantiated.")
    assert not errors_list, "Errors occurred: \n{}".format("\n".join(errors_list))


def test_gbtm_hyperoptimize(setup_config, setup_dataframe):
    """Test that GradientBoostedTreesModeler.hyperoptimize() returns a nested
    dictionary of parameters.
    """
    errors_list = []
    cat_features_list = setup_dataframe.select_dtypes(
        include=["object"]
    ).columns.tolist()
    for cat_feature in cat_features_list:
        setup_dataframe[cat_feature] = setup_dataframe[cat_feature].astype("category")
    setup_dataframe["FILE_DATE"] = pd.Series(
        pd.factorize(setup_dataframe["FILE_DATE"])[0]
    )
    subset_training_obs = (
        ~setup_dataframe["_validation"]
        & ~setup_dataframe["_test"]
        & ~setup_dataframe["_predict_obs"]
    )
    training_obs_lead_lengths = setup_dataframe[subset_training_obs][
        "_duration"
    ].value_counts()
    n_intervals = training_obs_lead_lengths[
        training_obs_lead_lengths > setup_config["MIN_SURVIVORS_IN_TRAIN"]
    ].index.max()
    modeler = lgb_modelers.GradientBoostedTreesModeler(
        config=setup_config,
        data=setup_dataframe,
    )
    modeler.n_intervals = n_intervals
    params = modeler.hyperoptimize(2)
    if len(params.keys()) != n_intervals:
        errors_list.append(
            "Number of parameter sets returned does not "
            "match number of lead lengths."
        )
    for k, v in params.items():
        if not isinstance(v, dict):
            errors_list.append(f"Parameter set {v} (with key {k}) " f"is not a dict.")
    assert not errors_list, "Errors occurred: \n{}".format("\n".join(errors_list))


def test_gbtm_train(setup_config, setup_dataframe):
    """Test that GradientBoostedTreesModeler.train() returns a list
    containing a trained model for each time interval.
    """
    errors_list = []
    cat_features_list = setup_dataframe.select_dtypes(
        include=["object"]
    ).columns.tolist()
    for cat_feature in cat_features_list:
        setup_dataframe[cat_feature] = setup_dataframe[cat_feature].astype("category")
    setup_dataframe["FILE_DATE"] = pd.Series(
        pd.factorize(setup_dataframe["FILE_DATE"])[0]
    )
    subset_training_obs = (
        ~setup_dataframe["_validation"]
        & ~setup_dataframe["_test"]
        & ~setup_dataframe["_predict_obs"]
    )
    training_obs_lead_lengths = setup_dataframe[subset_training_obs][
        "_duration"
    ].value_counts()
    n_intervals = training_obs_lead_lengths[
        training_obs_lead_lengths > setup_config["MIN_SURVIVORS_IN_TRAIN"]
    ].index.max()
    modeler = lgb_modelers.GradientBoostedTreesModeler(
        config=setup_config,
        data=setup_dataframe,
    )
    modeler.n_intervals = n_intervals
    models_list = modeler.train()
    if len(models_list) != n_intervals:
        errors_list.append(
            "Number of models trained and saved does not "
            "match number of lead lengths."
        )
    for index, model in enumerate(models_list):
        if not isinstance(model, lgb.basic.Booster):
            errors_list.append(
                f"Model #{index + 1} (of {len(models_list)}) "
                f"is not an instance of lgb.basic.Booster."
            )
        if not model.num_trees() > 0:
            errors_list.append(
                f"Model #{index + 1} (of {len(models_list)}) "
                f"was not trained (num_trees == 0)."
            )
        if not model.params:
            errors_list.append(
                f"Model #{index + 1} (of {len(models_list)}) "
                f"does not have any training parameters."
            )
    assert not errors_list, "Errors occurred: \n{}".format("\n".join(errors_list))


def test_gbtm_train_single_model(setup_config, setup_dataframe):
    """Test that GradientBoostedTreesModeler.train_single_model() returns a
    trained model of type lgb.basic.Booster.
    """
    errors_list = []
    cat_features_list = setup_dataframe.select_dtypes(
        include=["object"]
    ).columns.tolist()
    for cat_feature in cat_features_list:
        setup_dataframe[cat_feature] = setup_dataframe[cat_feature].astype("category")
    setup_dataframe["FILE_DATE"] = pd.Series(
        pd.factorize(setup_dataframe["FILE_DATE"])[0]
    )
    subset_training_obs = (
        ~setup_dataframe["_validation"]
        & ~setup_dataframe["_test"]
        & ~setup_dataframe["_predict_obs"]
    )
    training_obs_lead_lengths = setup_dataframe[subset_training_obs][
        "_duration"
    ].value_counts()
    n_intervals = training_obs_lead_lengths[
        training_obs_lead_lengths > setup_config["MIN_SURVIVORS_IN_TRAIN"]
    ].index.max()
    modeler = lgb_modelers.GradientBoostedTreesModeler(
        config=setup_config,
        data=setup_dataframe,
    )
    for time_horizon in range(n_intervals):
        model = modeler.train_single_model(time_horizon=time_horizon)
        if not isinstance(model, lgb.basic.Booster):
            errors_list.append(
                f"Model for time horizon {time_horizon} "
                f"is not an instance of lgb.basic.Booster."
            )
        if not model.num_trees() > 0:
            errors_list.append(
                f"Model for time horizon {time_horizon} "
                f"was not trained (num_trees == 0)."
            )
        if not model.params:
            errors_list.append(
                f"Model for time horizon {time_horizon} "
                f"does not have any training parameters."
            )
    assert not errors_list, "Errors occurred: \n{}".format("\n".join(errors_list))


def test_gbtm_predict(setup_config, setup_dataframe):
    """Test that GradientBoostedTreesModeler.predict() returns
    survival probabilities for all observations and time intervals.
    """
    errors_list = []
    reserved_cols = [
        "_test",
        "_validation",
        "_predict_obs",
        "_duration",
        "_event_observed",
        "_period",
        "_maximum_lead",
        "_spell",
        "SSNSCR",
    ]
    cat_features_list = setup_dataframe.select_dtypes(
        include=["object"]
    ).columns.tolist()
    for cat_feature in cat_features_list:
        setup_dataframe[cat_feature] = setup_dataframe[cat_feature].astype("category")
    numeric_features_list = [
        x for x in setup_dataframe.columns if x not in cat_features_list + reserved_cols
    ]
    setup_dataframe["FILE_DATE"] = pd.Series(
        pd.factorize(setup_dataframe["FILE_DATE"])[0]
    )
    subset_training_obs = (
        ~setup_dataframe["_validation"]
        & ~setup_dataframe["_test"]
        & ~setup_dataframe["_predict_obs"]
    )
    subset_validation_obs = (
        setup_dataframe["_validation"]
        & ~setup_dataframe["_test"]
        & ~setup_dataframe["_predict_obs"]
    )
    training_obs_lead_lengths = setup_dataframe[subset_training_obs][
        "_duration"
    ].value_counts()
    n_intervals = training_obs_lead_lengths[
        training_obs_lead_lengths > setup_config["MIN_SURVIVORS_IN_TRAIN"]
    ].index.max()
    models_list = []
    for time_horizon in np.arange(n_intervals):
        train_data = setup_dataframe[
            (
                setup_dataframe["_duration"] + setup_dataframe["_event_observed"]
                > time_horizon
            )
            & subset_training_obs
        ]
        train_data = lgb.Dataset(
            train_data[cat_features_list + numeric_features_list],
            label=train_data["_duration"] > time_horizon,
        )
        validation_data = setup_dataframe[
            (
                setup_dataframe["_duration"] + setup_dataframe["_event_observed"]
                > time_horizon
            )
            & subset_validation_obs
        ]
        validation_data = train_data.create_valid(
            validation_data[cat_features_list + numeric_features_list],
            label=validation_data["_duration"] > time_horizon,
        )
        model = lgb.train(
            {"objective": "binary"},
            train_data,
            num_boost_round=setup_config["MAX_EPOCHS"],
            early_stopping_rounds=setup_config["PATIENCE"],
            valid_sets=[validation_data],
            valid_names=["validation_set"],
            categorical_feature=cat_features_list,
            verbose_eval=True,
        )
        models_list.append(model)
    modeler = lgb_modelers.GradientBoostedTreesModeler(
        config=setup_config,
        data=setup_dataframe,
    )
    modeler.model = models_list
    predictions_dict = {}
    subset_bool_series = pd.Series(setup_dataframe["SSNSCR"].mod(2)).astype("bool")
    predictions_dict["cumulative"] = modeler.predict()
    predictions_dict["cumulative_subset"] = modeler.predict(subset=subset_bool_series)
    predictions_dict["marginal"] = modeler.predict(cumulative=False)
    predictions_dict["marginal_subset"] = modeler.predict(
        subset=subset_bool_series, cumulative=False
    )
    for case, predictions in predictions_dict.items():
        if predictions.shape[1] != n_intervals:
            errors_list.append(
                f"Number of periods in predictions array "
                f"({case}) does not match number of "
                f"intervals passed to predict() method."
            )
        if (predictions < 0).any() or (predictions > 1).any():
            errors_list.append(
                f"One or more of the predicted survival "
                f"probabilities ({case}) are outside of "
                f"the range [0, 1]."
            )
        if case in ["cumulative", "marginal"]:
            if predictions.shape[0] != len(setup_dataframe):
                errors_list.append(
                    f"Number of observations in predictions array ({case}) "
                    f"does not match number of observations passed to "
                    f"predict() method."
                )
        if case in ["cumulative_subset", "marginal_subset"]:
            if predictions.shape[0] != subset_bool_series.sum():
                errors_list.append(
                    f"Number of observations in predictions array ({case}) "
                    f"does not match number of observations passed to "
                    f"predict() method."
                )
    if (predictions_dict["cumulative"] > predictions_dict["marginal"]).any():
        errors_list.append(
            "Cumulative probability predictions exceed "
            "marginal probability predictions."
        )
    if (
        predictions_dict["cumulative_subset"] > predictions_dict["marginal_subset"]
    ).any():
        errors_list.append(
            "Cumulative subset probability predictions "
            "exceed marginal subset probability predictions."
        )
    if (
        predictions_dict["cumulative"][:, 0] != predictions_dict["marginal"][:, 0]
    ).any():
        errors_list.append(
            "Cumulative probability predictions do not "
            "match marginal probability predictions for "
            "first time interval."
        )
    if (
        predictions_dict["cumulative_subset"][:, 0]
        != predictions_dict["marginal_subset"][:, 0]
    ).any():
        errors_list.append(
            "Cumulative subset probability predictions do "
            "not match marginal subset probability "
            "predictions for first time interval."
        )
    assert not errors_list, "Errors occurred: \n{}".format("\n".join(errors_list))


def test_gbtm_build_model(setup_config, setup_dataframe):
    """Test that GradientBoostedTreesModeler.build_model()
    creates 'n_intervals' and 'model' attributes.
    """
    errors_list = []
    cat_features_list = setup_dataframe.select_dtypes(
        include=["object"]
    ).columns.tolist()
    for cat_feature in cat_features_list:
        setup_dataframe[cat_feature] = setup_dataframe[cat_feature].astype("category")
    setup_dataframe["FILE_DATE"] = pd.Series(
        pd.factorize(setup_dataframe["FILE_DATE"])[0]
    )
    modeler = lgb_modelers.GradientBoostedTreesModeler(
        config=setup_config,
        data=setup_dataframe,
    )
    if hasattr(modeler, "n_intervals"):
        errors_list.append(
            'GradientBoostedTreeModeler object has "n_intervals" attribute '
            'before calling "build_model" method.'
        )
    if hasattr(modeler, "model"):
        errors_list.append(
            'GradientBoostedTreeModeler object has "model" attribute '
            'before calling "build_model" method.'
        )
    modeler.build_model()
    if not hasattr(modeler, "n_intervals"):
        errors_list.append(
            '"Build_model" method does not create "n_intervals" attribute '
            "for GradientBoostedTreeModeler object."
        )
    if not hasattr(modeler, "model"):
        errors_list.append(
            '"Build_model" method does not create "model" attribute '
            "for GradientBoostedTreeModeler object."
        )
    assert not errors_list, "Errors occurred: \n{}".format("\n".join(errors_list))


def test_gbtm_set_n_intervals(setup_config, setup_dataframe):
    """Test that GradientBoostedTreesModeler.set_n_intervals() returns
    a value within the range of duration values.
    """
    errors_list = []
    min_duration = setup_dataframe["_duration"].min()
    max_duration = setup_dataframe["_duration"].max()
    modeler = lgb_modelers.GradientBoostedTreesModeler(
        config=setup_config, data=setup_dataframe
    )
    n_intervals = modeler.set_n_intervals()
    if not min_duration <= n_intervals <= max_duration:
        errors_list.append(
            'Returned "n_intervals" value is outside the ' "range of duration values."
        )
    assert not errors_list, "Errors occurred: \n{}".format("\n".join(errors_list))


SEED = 9999
np.random.seed(SEED)
