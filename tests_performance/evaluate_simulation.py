""" This script generates descriptive statistics and plots about the results of the Monte Carlo Simulation."""
import os
from datetime import date

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tests_performance.test_on_Monte_Carlo_simulation import run_simulation


def evaluate_results(df, PATH, type):
    """
    This function creates plots that investigate the results of the models we made.
    :param df: A csv file of evaluation information.
    :param PATH: path to where plots should be saved.
    :param type: str, 'auroc' or 'chi_squared'.
    :return: Dataframe with the descriptive statistics of AUROC.
    """

    if type == 'auroc':
        col = 'AUROC'
        name = 'AUROC'
        path_name = 'AUROC'

    elif type == 'chi_squared':
        col = 'p-value'
        name = r'$\chi^{2}$ Probability'
        path_name = 'chi_squared'
    else:
        NameError('Invalid type!')

    # remove NaN AUROC to determine if we should make a lineplot of average leads
    df = df.loc[df[col].notnull()]

    if df['Lead Length'].max() > 2:
        # Look at average at each lead length and graph this.
        plt.close()
        sns.lineplot(x='Lead Length', y=col, data=df).set_ylabel("Average {}".format(name))
        plt.tight_layout()
        plt.savefig(os.path.join(PATH, 'average_runs_{}.png'.format(path_name)))

        # Make spaghetti plot of lines
        plt.close()
        sns.lineplot(x='Lead Length', y=col, data=df, units='run', estimator=None).set_ylabel(name)
        sns.lineplot(x='Lead Length', y=col, data=df, ci=None, color='black')
        plt.tight_layout()
        plt.savefig(os.path.join(PATH, 'all_runs_{}.png'.format(path_name)))

    # Histograms for lead length == 1
    lead_length_1 = df.loc[df['Lead Length'] == 1]
    plt.close()
    sns.histplot(x=col, data=lead_length_1).set_xlabel(name)
    plt.tight_layout()
    plt.savefig(os.path.join(PATH, 'lead_1_hist_{}.png'.format(path_name)))

    if type == 'chi_squared':
        plt.close()
        sns.histplot(x='Chi-square', data=df).set_xlabel('$\chi^{2}$ Statistic')
        plt.tight_layout()
        plt.savefig(os.path.join(PATH, 'chi_square_hist.png'))

    # Make descriptive statistics
    stats = df.groupby('Lead Length')[col].describe().reset_index()
    stats['type'] = path_name

    return stats


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
    # Rework chi_squared to get the right format
    chi_square_combined = chi_square.loc[chi_square.Group == 'Combined']

    PATH = os.path.join(PATH, 'plots')
    os.makedirs(PATH, exist_ok=True)

    stats = []
    # Create AUROC evaluation plots
    temp = evaluate_results(evals, PATH, 'auroc')
    stats.append(temp)
    temp = evaluate_results(chi_square_combined, PATH, 'chi_squared')
    stats.append(temp)

    stats = pd.concat(stats).reset_index()
    stats.to_csv(os.path.join(PATH, 'descriptive_statistics.csv'), index=False)


if __name__ == '__main__':
    run_evaluations()
