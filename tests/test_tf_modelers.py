"""Conduct unit testing for fife.tf_modelers module."""

from fife import tf_modelers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten
import numpy as np
import pandas as pd


def test_binary_encode_feature(setup_config, setup_dataframe):
    """Test that binary_encode_feature function returns encoded columns."""
    errors_list = []
    df_numeric_vars = setup_dataframe.select_dtypes(include=["number"])
    test_cols = [x for x in df_numeric_vars.columns if x not in ["SSNSCR", "FILE_DATE"]]
    for col in test_cols:
        if df_numeric_vars[col].isnull().all():
            continue
        col_is_categorical = (
            df_numeric_vars[col].nunique() <= setup_config["MAX_UNIQUE_NUMERIC_CATS"]
        )
        if col_is_categorical:
            binary_cols = tf_modelers.binary_encode_feature(df_numeric_vars[col])
            if len(binary_cols.columns) != int(
                np.log2(max(df_numeric_vars[col].max(), 1)) + 1
            ):
                errors_list.append(
                    f"Incorrect number of encoded columns returned for {col}."
                )
            binary_cols["sum"] = binary_cols.sum(axis=1)
            if binary_cols["sum"].max() > int(
                np.log2(max(df_numeric_vars[col].max(), 1)) + 1
            ):
                errors_list.append(f"Returned columns encoded incorrectly for {col}.")
    assert not errors_list, "Errors occurred: \n{}".format("\n".join(errors_list))


def test_split_categorical_features(setup_dataframe):
    """Test that split_categorical_features function separates categorical
    columns from numeric columns.
    """
    errors_list = []
    categorical_features = setup_dataframe.select_dtypes(
        exclude=["number"]
    ).columns.tolist()
    for cat_feature in categorical_features:
        setup_dataframe[cat_feature] = setup_dataframe[cat_feature].astype("category")
    numeric_features = [
        x for x in setup_dataframe.columns if x not in categorical_features
    ]
    split_features_list = tf_modelers.split_categorical_features(
        setup_dataframe.copy(), categorical_features, numeric_features
    )
    series_in_split_features_list = [
        x for x in split_features_list if isinstance(x, pd.Series)
    ]
    if len(series_in_split_features_list) != len(categorical_features):
        errors_list.append(
            f"Number of split categorical series does not "
            f"match number of categorical columns in "
            f"original dataframe."
        )
    dataframe_in_split_features_list = [
        x for x in split_features_list if isinstance(x, pd.DataFrame)
    ]
    if len(dataframe_in_split_features_list) != 1:
        errors_list.append(f"Incorrect number of dataframes returned.")
    elif len(dataframe_in_split_features_list[0].columns) != len(numeric_features):
        errors_list.append(
            f"Number of numeric columns in returned "
            f"dataframe does not match number of numeric "
            f"columns in original dataframe."
        )
    elif not dataframe_in_split_features_list[0].equals(
        setup_dataframe[numeric_features]
    ):
        errors_list.append(
            f"Returned numeric dataframe does not match " f"original numeric dataframe."
        )
    assert not errors_list, "Errors occurred: \n{}".format("\n".join(errors_list))


def test_freeze_embedding_layers():
    """Test that embedding layers have trainable property set to false."""
    input_length = 4
    input_dim = 16
    output_dim = 2
    weights = np.random.normal(size=(input_dim, output_dim))
    embedding_layer = Embedding(
        input_dim=input_dim,
        output_dim=output_dim,
        input_length=input_length,
        weights=[weights],
        trainable=True,
        name="embedding_layer",
    )
    input_layer = Input(shape=(input_length,), dtype="int32", name="main_input")
    lyr = embedding_layer(input_layer)
    lyr = Flatten()(lyr)
    output_layer = Dense(1, activation="sigmoid")(lyr)
    model = Model(inputs=input_layer, outputs=output_layer)
    model = tf_modelers.freeze_embedding_layers(model)
    layer_out = [
        layer.trainable for layer in model.layers if type(layer).__name__ == "Embedding"
    ]
    assert not any(layer_out)


def test_make_predictions_cumulative():
    """Test that make_predictions_cumulative returns an instance of
    tf_modelers.CumulativeProduct.
    """
    input_layer = Input(shape=(32,))
    output_layer = Dense(32)(input_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    model = tf_modelers.make_predictions_cumulative(model)
    assert isinstance(model.layers[-1], tf_modelers.CumulativeProduct)


def test_make_predictions_marginal():
    """Test that make_predictions_marginal does not remove a layer if
    the layer is not a CumulativeProduct layer.
    """
    input_layer = Input(shape=(32,))
    output_layer = Dense(32)(input_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    marginal_model = tf_modelers.make_predictions_marginal(model)
    assert len(marginal_model.layers) == len(model.layers)


def test_ffnnm_init(setup_ffnnm_dataframe, setup_config):
    """Test that FeedforwardNeuralNetworkModeler instantiates properly."""
    errors_list = []
    assertions = []
    config = setup_config
    data = setup_ffnnm_dataframe
    try:
        if config["INDIVIDUAL_IDENTIFIER"] == "":
            config["INDIVIDUAL_IDENTIFIER"] = data.columns[0]
        if config["TIME_IDENTIFIER"] == "":
            config["TIME_IDENTIFIER"] = data.columns[1]
        modeler = tf_modelers.FeedforwardNeuralNetworkModeler(config=config, data=data)
        assertions.append(
            isinstance(modeler, tf_modelers.FeedforwardNeuralNetworkModeler)
        )
        if assertions[-1] is False:
            errors_list.append("Modeler did not instantiate properly")
    except Exception as error:
        assertions.append(False)
        errors_list.append(str(error))
    assertion = all(assertions)
    assert assertion, "Errors occurred: \n{}".format("\n".join(errors_list))


def test_ffnnm_hyperoptimize(setup_ffnnm_dataframe, setup_config):
    """Test that FeedforwardNeuralNetworkModeler.hyperoptimize() returns a
    dictionary of parameters.
    """
    errors_list = []
    cat_features_list = setup_ffnnm_dataframe.select_dtypes(
        include=["object"]
    ).columns.tolist()
    for cat_feature in cat_features_list:
        setup_ffnnm_dataframe[cat_feature] = setup_ffnnm_dataframe[cat_feature].astype(
            "category"
        )
    setup_ffnnm_dataframe["FILE_DATE"] = pd.Series(
        pd.factorize(setup_ffnnm_dataframe["FILE_DATE"])[0]
    )
    subset_training_obs = (
        ~setup_ffnnm_dataframe["_validation"]
        & ~setup_ffnnm_dataframe["_test"]
        & ~setup_ffnnm_dataframe["_predict_obs"]
    )
    training_obs_lead_lengths = setup_ffnnm_dataframe[subset_training_obs][
        "_duration"
    ].value_counts()
    n_intervals = training_obs_lead_lengths[
        training_obs_lead_lengths > setup_config["MIN_SURVIVORS_IN_TRAIN"]
    ].index.max()
    modeler = tf_modelers.FeedforwardNeuralNetworkModeler(
        config=setup_config,
        data=setup_ffnnm_dataframe,
    )
    modeler.n_intervals = n_intervals
    params = modeler.hyperoptimize(2)
    if not isinstance(params, dict):
        errors_list.append(f"Parameter set is not a dict.")
    assert not errors_list, "Errors occurred: \n{}".format("\n".join(errors_list))


def test_ffnnm_construct_embedding_network(setup_ffnnm_dataframe, setup_config):
    """Test that FeedforwardNeuralNetworkModeler.construct_embedding_network()
    returns a Keras training model."""
    errors_list = []
    assertions = []
    config = setup_config
    data = setup_ffnnm_dataframe
    subset_training_obs = ~data["_validation"] & ~data["_test"] & ~data["_predict_obs"]
    train_obs_lead_lengths = data[subset_training_obs]["_duration"].value_counts()
    n_intervals = train_obs_lead_lengths[
        train_obs_lead_lengths > config["MIN_SURVIVORS_IN_TRAIN"]
    ].index.max()
    categorical_features = [
        "nonmixed_categorical_var",
        "consistent_mixed_categorical_var",
    ]
    for col in categorical_features:
        data[col] = data[col].astype("category")
    data["FILE_DATE"], _ = pd.factorize(data["FILE_DATE"])
    try:
        if config["INDIVIDUAL_IDENTIFIER"] == "":
            config["INDIVIDUAL_IDENTIFIER"] = data.columns[0]
        if config["TIME_IDENTIFIER"] == "":
            config["TIME_IDENTIFIER"] = data.columns[1]
        modeler = tf_modelers.FeedforwardNeuralNetworkModeler(config=config, data=data)
        modeler.n_intervals = n_intervals
        modeler.model = modeler.construct_embedding_network()
        assertions.append(isinstance(modeler.model, Model))
        if not assertions[-1]:
            errors_list.append("Model not of type tensorflow.keras.Model")
    except Exception as error:
        assertions.append(False)
        errors_list.append(str(error))
    assertion = all(assertions)
    assert assertion, "Errors occurred: \n{}".format("\n".join(errors_list))


def test_ffnnm_train(setup_ffnnm_dataframe, setup_config):
    """Test that FeedforwardNeuralNetworkModeler.train() returns a Keras
    training model and that the modeler's weights change during training."""
    errors_list = []
    assertions = []
    config = setup_config
    data = setup_ffnnm_dataframe
    subset_training_obs = ~data["_validation"] & ~data["_test"] & ~data["_predict_obs"]
    train_obs_lead_lengths = data[subset_training_obs]["_duration"].value_counts()
    n_intervals = train_obs_lead_lengths[
        train_obs_lead_lengths > config["MIN_SURVIVORS_IN_TRAIN"]
    ].index.max()
    categorical_features = [
        "nonmixed_categorical_var",
        "consistent_mixed_categorical_var",
    ]
    for col in categorical_features:
        data[col] = data[col].astype("category")
    data["FILE_DATE"], _ = pd.factorize(data["FILE_DATE"])
    try:
        modeler = tf_modelers.FeedforwardNeuralNetworkModeler(config=config, data=data)
        modeler.n_intervals = n_intervals
        modeler.data[modeler.numeric_features] = modeler.data[
            modeler.numeric_features
        ].fillna(modeler.config["NON_CAT_MISSING_VALUE"])
        modeler.model = modeler.construct_embedding_network()
        weights_pretrain = modeler.model.get_weights()
        modeler.model = modeler.train()
        weights_posttrain = modeler.model.get_weights()
        no_change_in_weights_list = []
        for i, weight_pretrain in enumerate(weights_pretrain):
            no_change_in_weights_i = np.equal(weight_pretrain, weights_posttrain[i])
            no_change_in_weights_list.append(np.all(no_change_in_weights_i))
        weights_are_training = not all(no_change_in_weights_list)
        assertions.append(weights_are_training)
        if not assertions[-1]:
            errors_list.append(
                "Model weights have not changed, suggesting " "failure to train"
            )
        assertions.append(isinstance(modeler.model, Model))
        if not assertions[-1]:
            errors_list.append("Model not of type tensorflow.keras.Model")
    except Exception as error:
        assertions.append(False)
        errors_list.append(str(error))
    assertion = all(assertions)
    assert assertion, "Errors occurred: \n{}".format("\n".join(errors_list))


SEED = 9999
np.random.seed(SEED)
