"""
This script runs a Monte Carlo experiment to test the performance of FIFE's exit modeler on a known dgp.
"""

import json
import math
import os
import random
import numpy as np

from fife import utils
from fife.lgb_modelers import LGBSurvivalModeler, LGBStateModeler, LGBExitModeler
from fife.processors import PanelDataProcessor
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
        modeler = LGBSurvivalModeler(config={'MIN_SURVIVORS_IN_TRAIN': 5}, data=df)
    elif model == 'state':
        modeler = LGBStateModeler(config={'MIN_SURVIVORS_IN_TRAIN': 5}, data=df, state_col='exit_type')
    else:
        modeler = LGBExitModeler(config={'MIN_SURVIVORS_IN_TRAIN': 5}, data=df, exit_col='exit_type')
    modeler.build_model()

    # Pull out the final estimates to test later
    evaluation_subset = modeler.data["_period"] == (
            modeler.data["_period"].max() - test_intervals
    )
    evaluations = modeler.evaluate(evaluation_subset)
    forecasts = modeler.forecast()

    return forecasts, evaluations


def run_simulation(PATH, N_SIMULATIONS, MODEL= 'exit', N_PERSONS=400, N_PERIODS=40, N_EXTRA_FEATURES=0, EXIT_PROB=.2, SEED=None):
    """
    This script runs a Monte Carlo simulation of various FIFE models. The results of the evaluations and forecasts
    are saved in csvs.
    :param PATH: The file-path where forecasting and evaluation .csvs will be saved
    :param MODEL: 'base', 'state', or 'exit'; the type of FIFE model to run
    :param N_SIMULATIONS: The number of simulations to run
    :param N_PERSONS: The number of people created in the dataset
    :param N_PERIODS: The number of possible time periods in the dataset
    :param N_EXTRA_FEATURES: How many features over 3 to include. If more are included, they are random.
    :param EXIT_PROB: The probability an individual exits at the end of each time period
    :param SEED: A number to set the random seed
    :return: None, but saves 3 .csvs: The forecasts, evaluations, and created datasets.
    """

    if SEED is not None:
        np.random.seed(SEED)

    with open(os.path.join(os.getcwd(), "tests_performance", "config.json"), 'rb') as p:
        config = json.load(p)

    forecasts = []
    evaluations = []
    datas = []

    # Create a list of random seeds so that all of the data isn't the same.
    random_seeds = np.random.randint(N_SIMULATIONS*10, size=N_SIMULATIONS)

    for i in range(N_SIMULATIONS):
        data = fabricate_data(N_PERSONS=N_PERSONS,
                              N_PERIODS=N_PERIODS,
                              k=N_EXTRA_FEATURES,
                              SEED= random_seeds[i],
                              exit_prob=EXIT_PROB)
        numeric_suffixes = ['X1', 'X3']
        if N_EXTRA_FEATURES > 0:
            for items in range(4, N_EXTRA_FEATURES + 4):
                temp = 'X{}'.format(items)
                numeric_suffixes.append(temp)

        config['NUMERIC_SUFFIXES'] = numeric_suffixes

        data_processor = PanelDataProcessor(config={'NUMERIC_SUFFIXES': numeric_suffixes,
                                                    'TEST_INTERVALS': math.ceil(N_PERIODS / 3)}, data=data)
        data_processor.build_processed_data()

        forecast, evaluation = run_FIFE(data_processor.data, SEED, MODEL, math.ceil(N_PERIODS / 3))

        # Append this information to a df for forecasts, evaluations, and data
        forecast['model_type'] = MODEL
        forecast['run'] = i

        evaluation['model_type'] = MODEL
        evaluation['run'] = i

        data_processor.data['run'] = i

        forecasts.append(forecast)
        evaluations.append(evaluation)
        datas.append(data_processor.data)







if __name__ == '__main__':
    run_simulation('something', N_SIMULATIONS= 100, N_PERSONS=100, N_PERIODS=30, SEED=999)
