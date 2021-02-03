""" This script generates descriptive statistics and plots about the results of the Monte Carlo Simulation."""
import os
from datetime import date

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from tests_performance.test_on_Monte_Carlo_simulation import run_simulation


def plot_differences(dfs, PATH, type):
    """
    This function plots all of the data generating process results for a given simulation run on the same plots.
    :param dfs: A dataframe with markers for the DGP that the data belongs to
    :param PATH: A filepath where to save plots
    :param type: str, 'auroc' or 'chi_squared'.
    :return: None, but save plots in the filepath specified.
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

    # Build a plot with the average line of whatever statistic we're interested in seeing.
    # remove NaN AUROC to determine if we should make a lineplot of average leads
    dfs = dfs.loc[dfs[col].notnull()]

    if dfs['Lead Length'].max() > 2:
        plt.close()
        sns.lineplot(x='Lead Length', y=col, data=dfs, hue='DGP', legend='auto', palette='deep').set_ylabel("Average {}".format(name))
        plt.tight_layout()
        plt.savefig(os.path.join(PATH, 'average_runs_{}.png'.format(path_name)))

    # Histograms for lead length == 1
    lead_length_1 = dfs.loc[dfs['Lead Length'] == 1]
    plt.close()
    sns.histplot(x=col, data=lead_length_1, hue='DGP', element='step', palette='deep').set_xlabel(name)
    plt.tight_layout()
    plt.savefig(os.path.join(PATH, 'lead_1_hist_{}.png'.format(path_name)))

    if type == 'chi_squared':
        plt.close()
        sns.histplot(x='Chi-square', data=dfs, hue='DGP', palette='deep').set_xlabel('$\chi^{2}$ Statistic')
        plt.tight_layout()
        plt.savefig(os.path.join(PATH, 'chi_square_hist.png'))


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

    # Set visual palette
    sns.set_style('white')
    sns.set_palette('deep')

    chi_squ_all = []
    eval_all = []

    # Go through each dgp in a path and pull those out, too
    for i in tqdm(range(3)):
        newpath = os.path.join(PATH, '{}_dgp'.format(i + 1))
        evals = pd.read_csv(os.path.join(newpath, 'evaluations_{}.csv'.format(MODEL)))
        chi_square = pd.read_csv(os.path.join(newpath, 'chi_squared_{}.csv'.format(MODEL)))
        # Rework chi_squared to get the right format
        chi_square_combined = chi_square.loc[chi_square.Group == 'Combined']

        newpath = os.path.join(newpath, 'plots')
        os.makedirs(newpath, exist_ok=True)

        stats = []
        # Create AUROC evaluation plots
        temp = evaluate_results(evals, newpath, 'auroc')
        stats.append(temp)
        temp = evaluate_results(chi_square_combined, newpath, 'chi_squared')
        stats.append(temp)

        stats = pd.concat(stats).reset_index()
        stats.to_csv(os.path.join(newpath, 'descriptive_statistics.csv'), index=False)

        # Save data to list
        evals['DGP'] = i + 1
        eval_all.append(evals)
        chi_square['DGP'] = i + 1
        chi_squ_all.append(chi_square)

    # Create plots with information from all dgps
    eval_all = pd.concat(eval_all)
    chi_squ_all = pd.concat(chi_squ_all)
    plot_differences(eval_all, PATH, type='auroc')
    plot_differences(chi_squ_all, PATH, type='chi_squared')


if __name__ == '__main__':
    PATH = r'X:\Human Capital Group\Sponsored Projects\4854 DoN FIFE Extensions\Code\FIFE_Testing\exit_2021-02-02\1000_simulations\1000_persons\10_periods'
    run_evaluations(PATH=PATH)
