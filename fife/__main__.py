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
import pickle
import sys
from time import time

from fife import lgb_modelers, pd_modelers, processors, tf_modelers, utils
import pandas as pd


def main():
    """Execute default FIFE pipeline from data to forecasts and metrics."""
    checkpoint_time = time()

    # Read configuration parameters
    parser = utils.FIFEArgParser()
    args = parser.parse_args()
    config = {}
    if args.CONFIG_PATH:
        with open(args.CONFIG_PATH, "r") as file:
            config.update(json.load(file))
    config.update({k: v for k, v in vars(args).items() if v is not None})

    # Ensure reproducibility
    utils.make_results_reproducible(config["SEED"])
    utils.redirect_output_to_log(path=config["RESULTS_PATH"])
    print("Produced using FIFE: Finite-Interval Forecasting Engine")
    print("Copyright (c) 2018 - 2020, Institute for Defense Analyses (IDA)")
    print("Please cite using the suggested citation in the LICENSE file.\n")
    utils.print_config(config)

    # Read data file
    if "DATA_FILE_PATH" not in config.keys():
        valid_suffixes = (".csv", ".csv.gz", ".p", "pkl", ".h5")
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
    print(f"I/O setup time: {time() - checkpoint_time} seconds")
    checkpoint_time = time()

    # Process data
    data_processor = processors.PanelDataProcessor(config, data)
    data_processor.build_processed_data()
    print(f"Data processing time: {time() - checkpoint_time} seconds")
    checkpoint_time = time()

    # Save intermediate data
    utils.save_intermediate_data(
        data_processor.data,
        "Processed_Data",
        file_format="pickle",
        path=config["RESULTS_PATH"],
    )

    # Train and save model
    utils.ensure_folder_existence(f'{config["RESULTS_PATH"]}/Intermediate/Models')
    if config["TREE_MODELS"]:
        modeler = lgb_modelers.GradientBoostedTreesModeler(
            config=config, data=data_processor.data,
        )
        modeler.n_intervals = modeler.set_n_intervals()
        if config.get("HYPER_TRIALS", 0) > 0:
            params = modeler.hyperoptimize(config["HYPER_TRIALS"])
        else:
            params = None
        modeler.build_model(n_intervals=modeler.n_intervals, params=params)
        for i, lead_specific_model in enumerate(modeler.model):
            lead_path = (
                f'{config["RESULTS_PATH"]}/Intermediate/Models/'
                f"{i + 1}-lead_GBT_Model.json"
            )
            with open(lead_path, "w") as file:
                json.dump(lead_specific_model.dump_model(), file, indent=4)
    elif config.get("PROPORTIONAL_HAZARDS"):
        modeler = tf_modelers.ProportionalHazardsModeler(
            config=config, data=data_processor.data,
        )
        modeler.build_model()
        modeler.model.save(f'{config["RESULTS_PATH"]}/Intermediate/Models/PH_Model.h5')
    else:
        modeler = tf_modelers.FeedforwardNeuralNetworkModeler(
            config=config, data=data_processor.data,
        )
        modeler.n_intervals = modeler.set_n_intervals()
        if config.get("HYPER_TRIALS", 0) > 0:
            params = modeler.hyperoptimize(config["HYPER_TRIALS"])
        else:
            params = None
        modeler.build_model(n_intervals=modeler.n_intervals, params=params)
        modeler.model.save(
            f'{config["RESULTS_PATH"]}/Intermediate/Models/FFNN_Model.h5'
        )
    print(f"Model training time: {time() - checkpoint_time} seconds")
    checkpoint_time = time()

    # Save metrics or forecasts
    test_intervals = modeler.config.get(
        "TEST_INTERVALS", modeler.config.get("TEST_PERIODS", 0) - 1
    )
    if test_intervals > 0:

        # Save metrics
        evaluation_subset = modeler.data["_period"] == (
            modeler.data["_period"].max() - test_intervals
        )
        utils.save_output_table(
            modeler.evaluate(evaluation_subset), "Metrics", path=config["RESULTS_PATH"],
        )

        # Save counts by quantile
        utils.save_output_table(
            modeler.tabulate_survival_by_quantile(
                n_quantiles=config["QUANTILES"], subset=evaluation_subset,
            ),
            "Counts_by_Quantile",
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


if __name__ == "__main__":
    main()
