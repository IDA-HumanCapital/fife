#!/usr/bin/env python3
"""Execute default FIFE pipeline.

This script trains a discrete-time survival model on unbalanced panel data,
then forecasts survival for the observations in the most recent period.
It creates and populates Intermediate and Output folders in the given
directory.

See the accompanying README file for further details and instructions,
including documentation of all configuration parameters.
"""

import json
import os
from time import time
from typing import Type

from fife import lgb_modelers, processors, tf_modelers, utils
from fife.base_modelers import Modeler
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve


def main():
    """Execute default FIFE pipeline from data to survival forecasts and metrics."""
    checkpoint_time = time()
    config = parse_config()
    if config.get("EXIT_COL_PATH"):
        raise NotImplementedError(
            "Forecasting exit circumstances from the command line is not yet supported. Try LGBExitModeler from the FIFE Python package."
        )
    if config.get("STATE_COL"):
        raise NotImplementedError(
            "Forecasting future state from the command line is not yet supported. Try LGBStateModeler from the FIFE Python package."
        )
    utils.make_results_reproducible(config["SEED"])
    utils.redirect_output_to_log(path=config["RESULTS_PATH"])
    utils.print_copyright()
    utils.print_config(config)
    data = read_data(config)
    print(f"I/O setup time: {time() - checkpoint_time} seconds")

    checkpoint_time = time()
    data_processor = processors.PanelDataProcessor(config, data)
    data_processor.build_processed_data()
    utils.save_intermediate_data(
        data_processor.data,
        "Processed_Data",
        file_format="pickle",
        path=config["RESULTS_PATH"],
    )
    print(f"Data processing time: {time() - checkpoint_time} seconds")

    by_feature = config.get("BY_FEATURE", "")
    if by_feature != "" and by_feature not in data.columns:
        raise ValueError(
            "The selected feature for 'BY_FEATURE' is not in the dataset. Check spelling or the original dataset to ensure that you are entering the correct feature name."
        )

    checkpoint_time = time()
    utils.ensure_folder_existence(f'{config["RESULTS_PATH"]}/Intermediate/Models')
    test_intervals = config.get("TEST_INTERVALS", config.get("TEST_PERIODS", 0) - 1)
    if config.get("TREE_MODELS"):
        modeler_class = lgb_modelers.LGBSurvivalModeler
    elif config.get("PROPORTIONAL_HAZARDS"):
        modeler_class = tf_modelers.ProportionalHazardsModeler
    else:
        modeler_class = tf_modelers.FeedforwardNeuralNetworkModeler
    modeler = modeler_class(config=config, data=data_processor.data)
    modeler.n_intervals = (
        test_intervals if test_intervals > 0 else modeler.set_n_intervals()
    )
    if not config.get("TIME_ID_AS_FEATURE"):
        modeler.numeric_features.remove(config["TIME_IDENTIFIER"])
    if config.get("HYPER_TRIALS", 0) > 0:
        params = modeler.hyperoptimize(config["HYPER_TRIALS"])
    else:
        params = None
    modeler.build_model(n_intervals=modeler.n_intervals, params=params)
    modeler.save_model(path=f"{config['RESULTS_PATH']}/Intermediate/Models/")
    print(f"Model training time: {time() - checkpoint_time} seconds")

    checkpoint_time = time()
    if test_intervals > 0:

        # Save metrics
        max_test_intervals = int((len(set(modeler.data["_period"])) - 1) / 2)

        evaluation_subset = modeler.data["_period"] == (
            modeler.data["_period"].max() - min(test_intervals, max_test_intervals)
        )

        if by_feature != "":
            values = list(set(data[by_feature]))

            for feature_value in values:

                evaluation_subset_by_feature = modeler.data[by_feature] == feature_value
                evaluation_subset_comparison = (
                    evaluation_subset & evaluation_subset_by_feature
                )

                utils.save_output_table(
                    modeler.evaluate(evaluation_subset_comparison),
                    f"Metrics_{feature_value}",
                    path=config["RESULTS_PATH"],
                )

        utils.save_output_table(
            modeler.evaluate(evaluation_subset),
            "Metrics",
            path=config["RESULTS_PATH"],
        )

        # Save counts by quantile
        utils.save_output_table(
            modeler.tabulate_survival_by_quantile(
                n_quantiles=config["QUANTILES"],
                subset=evaluation_subset,
            ),
            "Counts_by_Quantile",
            index=False,
            path=config["RESULTS_PATH"],
        )

        # Save forecast errors
        actuals = np.array(
            [
                modeler.data[evaluation_subset]["_duration"] > time_horizon
                for time_horizon in range(test_intervals)
            ]
        ).T
        predictions = modeler.predict(evaluation_subset)
        utils.save_output_table(
            pd.DataFrame(
                predictions - actuals,
                columns=[
                    f"{time_horizon + 1}-period Forecast Error"
                    for time_horizon in range(test_intervals)
                ],
            ),
            "Forecast_Errors",
            index=False,
            path=config["RESULTS_PATH"],
        )

        # Save calibration errors
        predicted_share, actual_share = calibration_curve(
            actuals.flatten(), predictions.flatten(), n_bins=8, strategy="quantile"
        )
        calibration_errors = pd.DataFrame(
            [predicted_share, actual_share, actual_share - predicted_share]
        ).T
        calibration_errors.columns = [
            "Predicted Share",
            "Actual Share",
            "Calibration Error",
        ]
        calibration_errors.index.name = "Quantile"
        calibration_errors.index = calibration_errors.index + 1
        utils.save_output_table(
            calibration_errors,
            "Calibration_Errors",
            path=config["RESULTS_PATH"],
        )

    else:

        # Save forecasts
        individual_predictions = modeler.forecast()
        utils.save_output_table(
            individual_predictions, "Survival_Curves", path=config["RESULTS_PATH"]
        )

        # Save aggregated forecasts with uncertainty intervals
        utils.save_output_table(
            utils.compute_aggregation_uncertainty(individual_predictions),
            "Aggregate_Survival_Bounds",
            index=False,
            path=config["RESULTS_PATH"],
        )

        # Save and plot actual, fitted, and forecasted retention rates
        lead_periods = config["RETENTION_INTERVAL"]
        time_ids = pd.factorize(
            modeler.data[modeler.config["TIME_IDENTIFIER"]], sort=True
        )[0]
        retention_rates = modeler.tabulate_retention_rates(
            lead_periods=lead_periods, time_ids=time_ids
        )
        utils.save_output_table(
            retention_rates, "Retention_Rates", path=config["RESULTS_PATH"]
        )
        axes = retention_rates.plot()
        axes.set_ylabel(f"{lead_periods}-period Retention Rate")
        earliest_period = data_processor.data[
            data_processor.config["TIME_IDENTIFIER"]
        ].min()
        axes.set_xlabel(f"Periods Since {earliest_period}")
        utils.save_plot("Retention_Rates", path=config["RESULTS_PATH"])

        # Plot SHAP values for a subset of observations in the final period
        sample_size = config.get("SHAP_SAMPLE_SIZE", 0)
        if (
            isinstance(modeler, (lgb_modelers.GradientBoostedTreesModeler))
            and sample_size > 0
        ):
            shap_observations = (
                modeler.data[modeler.data["_predict_obs"]]
                .sample(n=sample_size)
                .sort_index()
            )
            subset = modeler.data.index.isin(shap_observations.index)
            shap_values = modeler.compute_shap_values(subset=subset)
            utils.plot_shap_values(
                shap_values,
                shap_observations[
                    modeler.categorical_features + modeler.numeric_features
                ],
                modeler.data[subset][
                    modeler.categorical_features + modeler.numeric_features
                ],
                config["TIME_IDENTIFIER"],
                path=config["RESULTS_PATH"],
            )

    print(f"Output production time: {time() - checkpoint_time} seconds")


def parse_config() -> dict:
    """Parse configuration parameters specified in the command line."""
    parser = utils.FIFEArgParser()
    args = parser.parse_args()
    config = {}
    if args.CONFIG_PATH:
        with open(args.CONFIG_PATH, "r") as file:
            config.update(json.load(file))
    config.update({k: v for k, v in vars(args).items() if v is not None})
    return config


def read_data(config: dict) -> pd.DataFrame:
    """Read the input dataset as specified in config or inferred from current directory."""
    if "DATA_FILE_PATH" not in config.keys():
        valid_suffixes = (".csv", ".csv.gz", ".p", ".pkl", ".h5")
        candidate_data_files = [
            file for file in os.listdir() if file.endswith(valid_suffixes)
        ]
        assert len(candidate_data_files) >= 1, (
            "No data files found in current directory. "
            f"Valid data file suffixes are {valid_suffixes}. "
            "If you want to use data in another directory, "
            "please specify the DATA_FILE_PATH."
        )
        assert len(candidate_data_files) <= 1, (
            "Multiple data files found in current directory. "
            "please specify the DATA_FILE_PATH."
        )
        print(f"Using {candidate_data_files[0]} as data file.")
        config["DATA_FILE_PATH"] = candidate_data_files[0]
    data = utils.import_data_file(config["DATA_FILE_PATH"])
    return data


if __name__ == "__main__":
    main()
