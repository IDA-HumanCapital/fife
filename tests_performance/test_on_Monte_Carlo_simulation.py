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
from Data_Fabrication import fabricate_data

def run_FIFE(df, holdout, seed, model):
    """
    :param df:
    :param model: 'base', 'state', or 'exit'.
    :return:
    """
    assert model in ['base', 'state', 'exit'], "Model must be 'base', 'state', or 'exit'!"

    utils.make_results_reproducible(seed)
    if model == 'base':
        df = df.drop('exit_type', axis='columns')
        holdout = holdout.drop('exit_type', axis='columns')
        modeler = LGBSurvivalModeler(data=df)
    elif model == 'state':
        modeler = LGBStateModeler(data=df, state_col='exit_type')
    else:
        modeler = LGBExitModeler(data=df, exit_col='exit_type')
    modeler.build_model()
    forecasts = modeler.forecast()
    modeler.evaluate(holdout)

    return modeler


def run_simulation(N_SIMULATIONS, HOLDOUT_SIZE=10, N_PERSONS=100, N_PERIODS=10, N_EXTRA_FEATURES=0, EXIT_PROB = .5, SEED=None):
    if SEED is not None:
        np.random.seed(SEED)

    with open(os.path.join(os.getcwd(), "config.json"), 'rb') as p:
        config = json.load(p)

    for i in range(N_SIMULATIONS):
        data = fabricate_data(N_PERSONS=N_PERSONS, 
                              N_PERIODS=N_PERIODS, 
                              k=N_EXTRA_FEATURES, 
                              exit_prob=EXIT_PROB)
        numeric_suffixes = ['X1', 'X3']  # Why do we need this?
        if N_EXTRA_FEATURES > 0:
            for items in range(4, N_EXTRA_FEATURES + 4):
                temp = 'X{}'.format(items)
                numeric_suffixes.append(temp)

        config['NUMERIC_SUFFIXES'] = numeric_suffixes

        data_processor = PanelDataProcessor(config= {'NUMERIC_SUFFIXES': numeric_suffixes}, data=data)
        data_processor.build_processed_data()

        # Take a holdout sample of individuals
        ssns = data['ID'].unique()
        holdout_ssns = np.random.choice(ssns, size=math.ceil(len(ssns) * HOLDOUT_SIZE / 100)).tolist()
        holdout = data_processor.data.loc[data_processor.data['ID'].isin(holdout_ssns)]
        prediction = data_processor.data.loc[~data_processor.data['ID'].isin(holdout_ssns)]

        base_modeler = run_FIFE(prediction, holdout, SEED, 'base')
        state_modeler = run_FIFE(prediction, holdout, SEED, 'state')
        exit_modeler = run_FIFE(prediction, holdout, SEED, 'exit')




if __name__ == '__main__':
    df = fabricate_data(N_PERSONS=100, N_PERIODS=10, k=2, exit_prob=.5)