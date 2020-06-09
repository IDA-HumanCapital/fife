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

    # Read config file, if any
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as file:
            config = json.load(file)
    else:
        print('No configuration file specified.')
        candidate_configs = [file for file in os.listdir()
                             if file.endswith('.json')]
        if len(candidate_configs) == 0:
            print('No json files found in current directory. '
                  'Will proceed with default configuration. ')
            config = {}
        else:
            assert len(candidate_configs) <= 1, (
                ('Multiple json files found in current directory. '
                 'Please specify a configuration file in your command, '
                 'e.g., "fife example_config.json".'))
            print(f'Using {candidate_configs[0]} as configuration file.')
            with open(candidate_configs[0], 'r') as file:
                config = json.load(file)

    # Use default values of config parameters not specified
    DEFAULT_CONFIG = {'SEED': 9999,
                      'RESULTS_PATH': 'FIFE_results',
					  'TREE_MODELS': True,
					  'RETENTION_INTERVAL': 1,
					  'QUANTILES': 5,
                      'SHAP_SAMPLE_SIZE': 128,
                      'FIXED_EFFECT_FEATURES': []}
    for k, v in DEFAULT_CONFIG.items():
        if k not in config.keys():
            config[k] = v

    # Ensure reproducibility
    utils.make_results_reproducible(config['SEED'])
    utils.redirect_output_to_log(path=config['RESULTS_PATH'])
    print('Produced using FIFE: Finite-Interval Forecasting Engine')
    print('Copyright (c) 2018 - 2020, Institute for Defense Analyses (IDA)')
    print('Please cite using the suggested citation in the LICENSE file.\n')
    utils.print_config(config)

    # Read data file
    if 'DATA_FILE_PATH' not in config.keys():
        valid_suffixes = ('.csv', '.csv.gz', '.p', 'pkl', '.h5')
        candidate_data_files = [file for file in os.listdir()
                                if file.endswith(valid_suffixes)]
        assert len(candidate_data_files) >= 1, (
            ('No data files found in current directory. '
             f'Valid data file suffixes are {valid_suffixes}. '
             'If you want to use data in another directory, '
             'please specify the DATA_FILE_PATH in a config file.'))
        assert len(candidate_configs) <= 1, (
            ('Multiple data files found in current directory. '
             'please specify the DATA_FILE_PATH in a config file.'))
        print(f'Using {candidate_data_files[0]} as data file.')
        config['DATA_FILE_PATH'] = candidate_data_files[0]
    data = utils.import_data_file(config['DATA_FILE_PATH'])
    print(f'I/O setup time: {time() - checkpoint_time} seconds')
    checkpoint_time = time()

    # Process data
    data_processor = processors.PanelDataProcessor(config, data)
    data_processor.build_processed_data()
    print(f'Data processing time: {time() - checkpoint_time} seconds')
    checkpoint_time = time()

    # Save intermediate files
    utils.save_maps(data_processor.categorical_maps,
                    'Categorical_Maps',
                    path=config['RESULTS_PATH'])
    utils.save_maps(data_processor.numeric_ranges,
                    'Numeric_Ranges',
                    path=config['RESULTS_PATH'])
    utils.save_intermediate_data(data_processor.data,
                                 'Processed_Data',
                                 file_format='pickle',
                                 path=config['RESULTS_PATH'])

    # Train and save model
    utils.ensure_folder_existence(
        f'{config["RESULTS_PATH"]}/Intermediate/Models')
    categorical_features = list(data_processor.categorical_maps.keys())
    if config['TREE_MODELS']:
        modeler = \
            lgb_modelers.GradientBoostedTreesModeler(
                config=config, data=data_processor.data,
                categorical_features=categorical_features)
        modeler.build_model()
        for i, lead_specific_model in enumerate(modeler.model):
            lead_path = (f'{config["RESULTS_PATH"]}/Intermediate/Models/'
                         f'{i + 1}-lead_GBT_Model.json')
            with open(lead_path, 'w') as file:
                json.dump(lead_specific_model.dump_model(), file, indent=4)
    elif config.get('PROPORTIONAL_HAZARDS'):
        modeler = \
            tf_modelers.ProportionalHazardsModeler(
                config=config, data=data_processor.data,
                categorical_features=categorical_features)
        modeler.build_model()
        modeler.model.save(
            f'{config["RESULTS_PATH"]}/Intermediate/Models/PH_Model.h5')
    else:
        modeler = \
            tf_modelers.FeedforwardNeuralNetworkModeler(
                config=config, data=data_processor.data,
                categorical_features=categorical_features)
        modeler.build_model()
        modeler.model.save(
            f'{config["RESULTS_PATH"]}/Intermediate/Models/FFNN_Model.h5')
    print(f'Model training time: {time() - checkpoint_time} seconds')
    checkpoint_time = time()

    # Save metrics and forecasts
    utils.save_output_table(
        modeler.evaluate(modeler.data['_validation'] & ~modeler.data['_test']),
        'Metrics', path=config['RESULTS_PATH'])
    individual_predictions = modeler.forecast()
    utils.save_output_table(individual_predictions,
                            'Survival_Curves',
                            path=config['RESULTS_PATH'])
    utils.save_output_table(
        utils.compute_aggregation_uncertainty(individual_predictions),
        'Aggregate_Survival_Bounds', index=False, path=config['RESULTS_PATH'])

    # Save and plot retention rates
    lead_periods = config['RETENTION_INTERVAL']
    time_ids = pd.factorize(
        modeler.data[modeler.config['TIME_IDENTIFIER']], sort=True)[0]
    retention_rates = modeler.tabulate_retention_rates(
        lead_periods=lead_periods, time_ids=time_ids)
    utils.save_output_table(retention_rates,
                            'Retention_Rates',
                            path=config['RESULTS_PATH'])
    axes = retention_rates.plot()
    axes.set_ylabel(f'{lead_periods}-period Retention Rate')
    earliest_period = data_processor.numeric_ranges.loc[
        data_processor.config["TIME_IDENTIFIER"], "Minimum"]
    axes.set_xlabel(f'Periods Since {earliest_period}')
    utils.save_plot('Retention_Rates',
                    path=config['RESULTS_PATH'])

    # Save event counts by quantile
    utils.save_output_table(modeler.tabulate_survival_by_quantile(
        n_quantiles=config['QUANTILES'],
		subset=modeler.data['_validation'] & ~modeler.data['_test']),
                            'Counts_by_Quantile',
                            index=False,
                            path=config['RESULTS_PATH'])

    # Plot SHAP values for a subset of observations in the final period
    if isinstance(modeler, (lgb_modelers.GradientBoostedTreesModeler)):
        subset = modeler.data.index.isin(data_processor.raw_subset.index)
        shap_values = modeler.compute_shap_values(subset=subset)
        utils.plot_shap_values(
            shap_values,
            data_processor.raw_subset[modeler.categorical_features
                                      + modeler.numeric_features],
            modeler.data[subset][modeler.categorical_features
                                 + modeler.numeric_features],
            config['TIME_IDENTIFIER'],
            path=config['RESULTS_PATH'])

    # Save metrics for interacted fixed effects model
    if ((set() < set(config['FIXED_EFFECT_FEATURES']) <=
         set(data_processor.data))):
        ife_modeler = \
            pd_modelers.InteractedFixedEffectsModeler(
                config=config, data=data_processor.data,
                categorical_features=categorical_features)
        ife_modeler.build_model()
        with open(f'{config["RESULTS_PATH"]}Intermediate/Models/IFE_Model.p',
                  'wb') as file:
            pickle.dump(ife_modeler.model, file)
        subset = ife_modeler.data['_validation'] & ~ife_modeler.data['_test']
        utils.save_output_table(ife_modeler.evaluate(subset),
                                'IFE_Metrics',
                                path=config['RESULTS_PATH'])
        ife_quantiles = ife_modeler.tabulate_survival_by_quantile(
            subset, n_quantiles=config['QUANTILES'])
        utils.save_output_table(ife_quantiles,
                                'IFE_Counts_by_Quantile',
                                index=False,
                                path=config['RESULTS_PATH'])

    print(f'Output production time: {time() - checkpoint_time} seconds')


if __name__ == '__main__':
    main()
