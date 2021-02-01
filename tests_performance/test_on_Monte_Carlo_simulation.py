"""
This script runs a Monte Carlo experiment to test the performance of FIFE's exit modeler on a known dgp.
"""

import json
import math
import os
import sys
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from fife.lgb_modelers import LGBSurvivalModeler, LGBStateModeler, LGBExitModeler
from fife.processors import PanelDataProcessor
from tests_performance.Data_Fabrication import fabricate_data


def eval_chi_square(true_df, forecasts, dgp=1):
    '''
    Return chi^2 information comparing forecasts with the actual probabilities.
    :param true_df: Pandas DF, the data on which the model's forecasts are created.
    :param forecasts: Pandas DF, forecasts from model.forecast() in FIFE
    :param type_test: str, 'vector' or 'single'. Represents method of calculating Chi^2.
    :param dgp: 1 or 2. The type of data generation we did.
    :return: A pandas df with
    '''

    # Add information about true values based on rules of the DGP to the true_df
    true_df = true_df[['ID', 'X1']]
    if dgp == 1:
        conditions = [(true_df['X1'] == 'A'), (true_df['X1'] == 'B'), (true_df['X1'] == 'C')]
        values_x = [0.7, 0.2, 0.1]
        values_y = [0.2, 0.7, 0.2]
        values_z = [0.1, 0.1, 0.7]
        groups = [1, 2, 3]
        true_df['prob_exit_X'] = np.select(conditions, values_x)
        true_df['prob_exit_Y'] = np.select(conditions, values_y)
        true_df['prob_exit_Z'] = np.select(conditions, values_z)
        true_df['groups'] = np.select(conditions, groups)
    else:
        true_df['prob_exit_X'] = np.where(true_df['X1'] == 'A', 0.6, 1 / 3)
        true_df['prob_exit_Y'] = np.where(true_df['X1'] == 'A', 0.3, 1 / 3)
        true_df['prob_exit_Z'] = np.where(true_df['X1'] == 'A', 0.1, 1 / 3)
        groups = [1, 2]
        true_df['groups'] = np.select(true_df['X1'] == 'A', groups)

    true_df = true_df.drop('X1', axis=1).rename({'prob_exit_X': 'X', 'prob_exit_Y': 'Y', 'prob_exit_Z': 'Z'}, axis=1)
    true_df = pd.melt(true_df, id_vars=['ID', 'groups'], value_vars=['X', 'Y', 'Z'], var_name='Future exit_type',
                      value_name='prob_exit')

    # In this dgp, the probability of exit stays the same at each period, so it doesn't matter which period we're at.
    forecasts = forecasts.reset_index()
    test_stat = forecasts.merge(true_df, how='left', on=['ID', 'Future exit_type'])
    period_cols = [i for i in test_stat.columns if '-period' in i]

    # We want to return the average chi^2 probability over individuals by time period
    # Calculate Chi-Squared test statistic for each period and individual
    test_stat[period_cols] = test_stat[period_cols].sub(test_stat['prob_exit'], axis=0)

    # Calculate mean by original groups (whether X1 is A, B, or C) and the values. This will be the 9 groups
    # representing the combinations of I_m and d in the formula
    test_stat = test_stat.groupby(['groups', 'prob_exit'])[period_cols].mean().reset_index()

    for i in period_cols:
        test_stat[i] = (test_stat[i] ** 2) / test_stat['prob_exit']

    test_stat = test_stat.groupby('groups')[period_cols].mean().reset_index()

    # Now, sum the values and then find the $\chi^{2}$ probabilities from the CDF of $\chi^{2}$
    test_stat = (test_stat[period_cols].sum(axis=0)).reset_index().rename({'index': 'Lead Length', 0: 'Chi Squared'},
                                                                          axis=1)
    test_stat['Lead Length'] = test_stat['Lead Length'].str.split('-', expand=True)[0]
    test_stat['Chi Squared'] = 1 - stats.chi2.cdf(test_stat['Chi Squared'], 2)

    return test_stat


def run_FIFE(df, model, test_intervals, dgp):
    """
    :param df: Data frame created before
    :param seed: seed for replication
    :param model: 'base', 'state', or 'exit'.
    :param test_intervals: the number of intervals to remove from the training set for testing purposes later
    :param dgp: 1 or 2. The type of data generating process we used.
    :return: modeler, the model created and forecasts, the spreadsheet of forecasted probabilities
    """
    assert model in ['base', 'state', 'exit'], "Model must be 'base', 'state', or 'exit'!"

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
    chi_squared = eval_chi_square(df[modeler.data['_predict_obs']], forecasts, dgp=dgp)

    return forecasts, evaluations, chi_squared


def run_simulation(PATH, N_SIMULATIONS=1000, MODEL='exit', N_PERSONS=10000, N_PERIODS=40, N_EXTRA_FEATURES=0,
                   EXIT_PROB=0.2, SEED=None, dgp=1):
    """
    This script runs a Monte Carlo simulation of various FIFE models. The results of the evaluations and forecasts
    are saved in csvs.
    :param PATH: The file-path where forecasting and evaluation file with .csvs will be saved
    :param MODEL: 'base', 'state', or 'exit'; the type of FIFE model to run
    :param N_SIMULATIONS: The number of simulations to run
    :param N_PERSONS: The number of people created in the dataset
    :param N_PERIODS: The number of possible time periods in the dataset
    :param N_EXTRA_FEATURES: How many features over 3 to include. If more are included, they are random.
    :param EXIT_PROB: The probability an individual exits at the end of each time period
    :param SEED: A number to set the random seed
    :param dgp: data fabrication dgp (see Data_Fabrication.py)
    :return: None, but saves 3 .csvs: The forecasts, evaluations, and created datasets.
    """

    assert MODEL in ['base', 'state', 'exit'], "MODEL must be of type 'base', 'state', or 'exit'!"

    today = str(date.today())
    PATH = os.path.join(PATH, '{}_{}'.format(MODEL, today))

    if SEED is not None:
        np.random.seed(SEED)

    with open(os.path.join(os.getcwd(), "tests_performance", "config.json"), 'rb') as p:
        config = json.load(p)

    forecasts = []
    evaluations = []
    chi_squares = []

    for i in tqdm(range(N_SIMULATIONS)):
        data = fabricate_data(N_PERSONS=N_PERSONS,
                              N_PERIODS=N_PERIODS,
                              k=N_EXTRA_FEATURES,
                              exit_prob=EXIT_PROB,
                              dgp=dgp)
        numeric_suffixes = ['X2', 'X3']
        if N_EXTRA_FEATURES > 0:
            for items in range(4, N_EXTRA_FEATURES + 4):
                temp = 'X{}'.format(items)
                numeric_suffixes.append(temp)

        config['NUMERIC_SUFFIXES'] = numeric_suffixes

        data_processor = PanelDataProcessor(config={'NUMERIC_SUFFIXES': numeric_suffixes,
                                                    'TEST_INTERVALS': math.ceil(N_PERIODS / 3)}, data=data)
        data_processor.build_processed_data()

        try:
            forecast, evaluation, chi_squared = run_FIFE(data_processor.data, MODEL, math.ceil(N_PERIODS / 3), dgp=dgp)

            # Append this information to a df for forecasts, evaluations, and data
            forecast['run'] = i
            evaluation['run'] = i
            chi_squared['run'] = i

            forecasts.append(forecast)
            evaluations.append(evaluation)
            chi_squares.append(chi_squared)

        except ValueError:
            # Fix for if there is a problem with the validation set being empty for some periods of time until the bug
            # is removed
            continue

    # Save information about this run in the provided path
    os.makedirs(os.path.join(PATH), exist_ok=True)
    forecasts = pd.concat(forecasts).reset_index()
    evaluations = pd.concat(evaluations).reset_index()
    chi_squares = pd.concat(chi_squares).reset_index()

    forecasts.to_csv(os.path.join(PATH, 'forecasts_{}.csv'.format(MODEL)), index=False)
    evaluations.to_csv(os.path.join(PATH, 'evaluations_{}.csv'.format(MODEL)), index=False)
    chi_squares.to_csv(os.path.join(PATH, 'chi_squared_{}.csv'.format(MODEL)), index=False)

    original_stdout = sys.stdout
    with open(os.path.join(PATH, 'run_information.txt'), 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print(
            'Run information: \nModel: {}\nN_SIMULATIONS: {}\n N_PERSONS: {}\n N_PERIODS: {}\nEXIT_PROB: {}'.format(
                MODEL, N_SIMULATIONS, N_PERSONS, N_PERIODS, EXIT_PROB))
        sys.stdout = original_stdout  # Reset the standard output to its original value


if __name__ == '__main__':
    PATH = r'X:\Human Capital Group\Sponsored Projects\4854 DoN FIFE Extensions\Code\FIFE_Testing'
    run_simulation(PATH=PATH, SEED=999)

