""" This script generates descriptive statistics and plots about the results of the Monte Carlo Simulation."""
import os
from datetime import date

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tests_performance.test_on_Monte_Carlo_simulation import run_simulation


def generate_plots(df, PATH, type='auroc'):
    """
    This function creates plots that investigate the results of the models we made.
    :param df: A csv file of evaluation information.
    :param PATH: path to where plots should be saved.
    :param type: 'auroc' or 'chi_squared'
    :return: None, but save plots describing the results of the model.
    """

    assert type in ['auroc', 'chi_squared'], "We are looking at 'auroc' or 'chi_squared' only."

    if type == 'auroc':
        col = 'AUROC'
    else:
        col = 'Chi Squared'

    # Look at average at each lead length and graph this.
    plt.close()
    sns.lineplot(x='Lead Length', y=col, data=df).set_ylabel("Average {}".format(col))
    plt.tight_layout()
    plt.savefig(os.path.join(PATH, 'average_runs_{}.png'.format(col)))

    plt.close()
    sns.lineplot(x='Lead Length', y=col, data=df, units='run', estimator=None).set_ylabel(col)
    sns.lineplot(x='Lead Length', y=col, data=df, ci=None, color='black')
    if col=='Chi Squared':
        sns.lineplot(x='Lead Length', y=col, data=df, ci=None, color='black')
    plt.tight_layout()
    plt.savefig(os.path.join(PATH, 'all_runs_{}.png'.format(col)))


def run_evaluations(PATH=None, MODEL='exit'):
    """
    This is the workhouse function that
    :param path: default None. The path to the data files to evaluate. If None is used, a Monte carlo simulation is run
    first.
    :param MODEL: base', 'state', or 'exit'; the type of FIFE model we're evaluating.
    :return: Nothing, but saves evaluation in forms of plots and descriptive statistics.
    """

    assert MODEL in ['base', 'state', 'exit'], "MODEL must be of type 'base', 'state', or 'exit'!"

    if PATH is None:  # we need to run the simulation first before creating evaluations
        PATH = r'X:\Human Capital Group\Sponsored Projects\4854 DoN FIFE Extensions\Code\FIFE_Testing'
        run_simulation(PATH=PATH, SEED=999)
        today = str(date.today())
        PATH = os.path.join(PATH, '{}_{}'.format(MODEL, today))

    evals = pd.read_csv(os.path.join(PATH, 'evaluations_{}.csv'.format(MODEL)))
    chi_square = pd.read_csv(os.path.join(PATH, 'chi_squared_{}.csv'.format(MODEL)))

    PATH = os.path.join(PATH, 'plots')
    os.makedirs(PATH, exist_ok=True)

    # Create AUROC evaluation plots
    generate_plots(evals, PATH, type='auroc')
    generate_plots(chi_square, PATH, type='chi_squared')


if __name__ == '__main__':
    run_evaluations()
