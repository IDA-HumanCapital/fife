"""This file creates Cumulative Incidence Function forecasts by combining forecasts from the LGBSurvivalModeler and the LGBExitModeler"""

from fife.processors import PanelDataProcessor
from fife.lgb_modelers import LGBExitModeler, LGBSurvivalModeler, LGBStateModeler
from tests_performance.Data_Fabrication import fabricate_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_forecast(df, modeler="LGBSurvivalModeler", exit_col='exit_type'):
    # Need to be able to incorporate appropriate inputs to the modelers.
    # Maybe use something similar to the relevant function from the testing code.
    dp = PanelDataProcessor(data=df)
    dp.build_processed_data()
    if modeler == "LGBSurvivalModeler":
        m = LGBSurvivalModeler(data=dp.data)
    elif modeler == "LGBExitModeler":
        m = LGBExitModeler(data=dp.data, exit_col=exit_col)
    else:
        raise NameError('Invalid modeler')
    m.build_model(parallelize=False)
    f = m.forecast()
    f = f.reset_index()
    return f


def get_forecasts(df, exit_col='exit_type'):
    f0 = get_forecast(df, modeler="LGBSurvivalModeler", exit_col=exit_col)
    f1 = get_forecast(df, modeler="LGBExitModeler", exit_col=exit_col)
    f = f0.merge(f1, how="outer", on="ID")
    return f


def rename_cols(f):
    # Unused
    temp = [i for i in f.columns]
    for i in range(len(f.columns)):
        if 'State' in f.columns[i]:
            temp[i] = ''.join(f.columns[i].split('-')[0] + '-State')
        elif 'Survival' in f.columns[i]:
            temp[i] = ''.join(f.columns[i].split('-')[0] + '-Survival')
    f.columns = temp
    return f


def calc_CIF(f):
    temp = [i for i in f.columns if "-period Surv" in i]
    num = sorted([int(x.split('-')[0]) for x in temp])
    for i in num:
        a = f[''.join(str(i) + '-period State Probabilities')]
        b = 1 - f[''.join(str(i) + '-period Survival Probability')]
        c = 0  # might need to fix for cases when (i-1) not in num; right now I just raise an error
        e = 0  # might need to fix this by making it the previous existing marginal e.g. for cases when (i-1) not in num; right now I just raise an error
        if i > 1:
            if (i - 1) in num:
                c = 1 - f[''.join(str(i - 1) + '-period Survival Probability')]
                e = f[''.join(str(i - 1) + '-CIF')]
            else:
                raise NameError("Warning: (i-1) not in num")
        f[''.join(str(i) + '-joint')] = a * (b - c)  # time period specific probabilities
        f[''.join(str(i) + '-CIF')] = f[''.join(str(i) + '-joint')] + e  # cumulative
    # f = rename_cols(f)  # no need to rename any more
    return f


def get_features_and_collapse(d0, f, grouping_vars=None):
    if grouping_vars is None:
        grouping_vars = ["X1", "Future exit_type"]  # need to specify this for other data sets
    # now merge in with the original data to get features
    temp = d0.groupby("ID").tail(1)  # may need to sort by period first to get the last row
    out = f.merge(temp, how="left", on="ID")
    df = out.groupby(grouping_vars).mean()
    df = df.reset_index()
    return df


def wide_to_long(df, grouping_vars=None):
    if grouping_vars is None:
        grouping_vars = ["X1", "Future exit_type"]  # need to specify this for other data sets
    temp = [i for i in df.columns if "-CIF" in i]
    num = sorted([int(x.split('-')[0]) for x in temp])
    df2 = df.melt(id_vars=grouping_vars, value_vars=[str(n) + '-CIF' for n in num], value_name="CIF",
                  var_name="Period")
    df2["Period"] = df2["Period"].apply(lambda x: int(x.split('-')[0]))
    return df2


def CIF(d0, grouping_vars=None, exit_col='exit_type'):
    if grouping_vars is None:
        grouping_vars = ["X1", "Future " + exit_col]  # need to specify this for other data sets
    f = get_forecasts(d0, exit_col=exit_col)
    f = calc_CIF(f)
    df = get_features_and_collapse(d0, f, grouping_vars=grouping_vars)
    df2 = wide_to_long(df, grouping_vars=grouping_vars)
    return df2


def plot_CIF(df2, linestyles=None):
    # Note that this plotting function is for the specific case generated from our simulations only.
    # Different data will necessitate a modified plotting function.
    if linestyles is None:
        linestyles = {'X': 'solid', 'Y': 'dashed', 'Z': 'dotted'}
    xlims = [(min(df2["Period"])-1), (max(df2["Period"])+1)]
    xticks = np.sort(df2["Period"].unique())
    grouped = df2.groupby(["X1"])
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 3), sharey=True)
    for (key, ax) in zip(grouped.groups.keys(), axes.flatten()):
        print(key)
        print(grouped.get_group(key))
        print(grouped.get_group(key).groupby("Future exit_type"))
        for label, a in grouped.get_group(key).groupby("Future exit_type"):
            linestyle = linestyles[label]
            a.plot(x='Period', y='CIF', ax=ax, label=label, linestyle=linestyle)
            ax.set_title("$CIF$, $X_{1}=" + key + "$, by exit type")
            ax.set_xlim(xlims)
            ax.set_xticks(xticks)
    axes[0].set_ylabel(r'$P(1 \leq T_{i} \leq h, D=d|...)$')
    # plt.ylabel(r'$P(0 \le T_{i} \le h, D=d)$')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = fabricate_data(N_PERSONS=1000, N_PERIODS=20, SEED=1234, exit_prob=.3)  # example data
    cif = CIF(df)
    plot_CIF(cif)


