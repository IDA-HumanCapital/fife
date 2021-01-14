"""
This script runs a Monte Carlo experiment to test the performance of FIFE's exit modeler on a known dgp.
"""

import math, json
import os

from fife import utils
from fife.lgb_modelers import LGBSurvivalModeler, LGBStateModeler, LGBExitModeler
from fife.processors import PanelDataProcessor
import numpy as np
import pandas as pd
from tests_performance.Data_Fabrication import fabricate_data


def run_FIFE(df, seed, model, test_intervals):
    """
    :param df: Data frame created before
    :param seed: seed for replication
    :param model: 'base', 'state', or 'exit'.
    :param test_intervals: the number of intervals to remove from the training set for testing purposes later
    :return: modeler, the model created and forecasts, the spreadsheet of forecasted probabilities
    """
    assert model in ['base', 'state', 'exit'], "Model must be 'base', 'state', or 'exit'!"

    utils.make_results_reproducible(seed)
    if model == 'base':
        df = df.drop('exit_type', axis='columns')
        modeler = LGBSurvivalModeler(data=df)
    elif model == 'state':
        modeler = LGBStateModeler(data=df, state_col='exit_type')
    else:
        modeler = LGBExitModeler(data=df, exit_col='exit_type')
    modeler.build_model()

    # Pull out the final estimates to test later
    evaluation_subset = modeler.data["_period"] == (
            modeler.data["_period"].max() - test_intervals
    )
    modeler.evaluate(evaluation_subset)
    forecasts = modeler.forecast()

    return modeler, forecasts


def run_simulation(N_SIMULATIONS, N_PERSONS=100, N_PERIODS=10, N_EXTRA_FEATURES=0, EXIT_PROB=.5, SEED=None):
    if SEED is not None:
        np.random.seed(SEED)

    with open(os.path.join(os.getcwd(), "tests_performance", "config.json"), 'rb') as p:
        config = json.load(p)

    for i in range(N_SIMULATIONS):
        data = fabricate_data(N_PERSONS=N_PERSONS,
                              N_PERIODS=N_PERIODS,
                              k=N_EXTRA_FEATURES,
                              exit_prob=EXIT_PROB)
        numeric_suffixes = ['X1', 'X3']
        if N_EXTRA_FEATURES > 0:
            for items in range(4, N_EXTRA_FEATURES + 4):
                temp = 'X{}'.format(items)
                numeric_suffixes.append(temp)

        config['NUMERIC_SUFFIXES'] = numeric_suffixes

        data_processor = PanelDataProcessor(config={'NUMERIC_SUFFIXES': numeric_suffixes, 'MIN_SURVIVORS_IN_TRAIN': 5,
                                                    'TEST_INTERVALS': math.ceil(N_PERIODS / 3)}, data=data)
        data_processor.build_processed_data()

        # Take a holdout sample of individuals
        #  ssns = data['ID'].unique()
        # holdout_ssns = np.random.choice(ssns, size=math.ceil(len(ssns) * HOLDOUT_SIZE / 100)).tolist()
        # holdout = data_processor.data.loc[data_processor.data['ID'].isin(holdout_ssns)]
        # prediction = data_processor.data.loc[~data_processor.data['ID'].isin(holdout_ssns)]

        base_modeler = run_FIFE(data_processor.data, SEED, 'base', math.ceil(N_PERIODS / 3))
        state_modeler = run_FIFE(data_processor.data, SEED, 'state', math.ceil(N_PERIODS / 3))
        exit_modeler = run_FIFE(data_processor.data, SEED, 'exit', math.ceil(N_PERIODS / 3))


if __name__ == '__main__':
    run_simulation(100, N_PERSONS=100, N_PERIODS=30, SEED=999)
