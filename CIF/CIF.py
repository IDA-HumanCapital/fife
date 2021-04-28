"""This file creates Cumulative Incidence Function forecasts by combining forecasts from the LGBSurvivalModeler and the LGBExitModeler"""

from fife.processors import PanelDataProcessor
from fife.lgb_modelers import LGBExitModeler, LGBSurvivalModeler, LGBStateModeler
from tests_performance.Data_Fabrication import fabricate_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_forecast(df, modeler="LGBSurvivalModeler", exit_col='exit_type', PDPkwargs=None, Survivalkwargs=None, Exitkwargs=None):
    # Need to be able to incorporate appropriate inputs to the modelers.
    # Maybe use something similar to the relevant function from the testing code.
    if PDPkwargs is None:
        dp = PanelDataProcessor(data=df)
    else:
        if type(PDPkwargs) is not dict:
            raise TypeError("PDPkwargs must be a dictionary")
        dp = PanelDataProcessor(data=df, **PDPkwargs)
    dp.build_processed_data()
    if modeler == "LGBSurvivalModeler":
        if Survivalkwargs is None:
            m = LGBSurvivalModeler(data=dp.data)
        else:
            if type(Survivalkwargs) is not dict:
                raise TypeError("Survivalkwargs must be a dictionary")
            m = LGBSurvivalModeler(data=dp.data, **Survivalkwargs)
    elif modeler == "LGBExitModeler":
        if Exitkwargs is None:
            m = LGBExitModeler(data=dp.data, exit_col=exit_col)
        else:
            if type(Exitkwargs) is not dict:
                raise TypeError("Exitkwargs must be a dictionary")
            if "exit_col" in Exitkwargs.keys():
                exit_col = Exitkwargs["exit_col"]
                Exitkwargs.pop("exit_col")
            m = LGBExitModeler(data=dp.data, exit_col=exit_col, **Exitkwargs)
    else:
        raise NameError('Invalid modeler')
    m.build_model(parallelize=False)
    f = m.forecast()
    f = f.reset_index()
    return f


def get_forecasts(df, ID="ID", exit_col='exit_type', PDPkwargs=None, Survivalkwargs=None, Exitkwargs=None):
    f0 = get_forecast(df, modeler="LGBSurvivalModeler", exit_col=exit_col, PDPkwargs=PDPkwargs, Survivalkwargs=Survivalkwargs)
    f1 = get_forecast(df, modeler="LGBExitModeler", exit_col=exit_col, PDPkwargs=PDPkwargs, Exitkwargs=Exitkwargs)
    f = f0.merge(f1, how="outer", on=ID) # maybe change to get ID automatically by having get_forecast return the id from the data processor
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


def get_features_and_collapse(d0, f, grouping_vars=None, exit_col="exit_type", ID="ID"):
    grouping_vars = grouping_vars_subfcn(grouping_vars=grouping_vars, exit_col=exit_col)
    # if grouping_vars is None:
    #     grouping_vars = ["X1", "Future exit_type"]  # need to specify this for other data sets
    # now merge in with the original data to get features
    temp = d0.groupby(ID).tail(1)  # may need to sort by period first to get the last row
    out = f.merge(temp, how="left", on=ID)
    df = out.groupby(grouping_vars).mean()
    df = df.reset_index()
    return df


def wide_to_long(df, grouping_vars=None, exit_col="exit_type"):
    grouping_vars = grouping_vars_subfcn(grouping_vars=grouping_vars, exit_col=exit_col)
    # if grouping_vars is None:
    #     grouping_vars = ["X1", "Future exit_type"]  # need to specify this for other data sets
    temp = [i for i in df.columns if "-CIF" in i]
    num = sorted([int(x.split('-')[0]) for x in temp])
    df2 = df.melt(id_vars=grouping_vars, value_vars=[str(n) + '-CIF' for n in num], value_name="CIF",
                  var_name="Period")
    df2["Period"] = df2["Period"].apply(lambda x: int(x.split('-')[0]))
    return df2


def CIF(d0, ID="ID", grouping_vars=None, exit_col="exit_type", PDPkwargs=None, Survivalkwargs=None, Exitkwargs=None):
    grouping_vars = grouping_vars_subfcn(grouping_vars=grouping_vars, exit_col=exit_col)
    f = get_forecasts(d0, ID=ID, exit_col=exit_col, PDPkwargs=PDPkwargs, Survivalkwargs=Survivalkwargs, Exitkwargs=Exitkwargs)
    f = calc_CIF(f)
    df = get_features_and_collapse(d0, f, grouping_vars=grouping_vars, exit_col=exit_col, ID=ID)
    df2 = wide_to_long(df, grouping_vars=grouping_vars, exit_col=exit_col)
    return df2


def grouping_vars_subfcn(grouping_vars=None, exit_col="exit_type", include_forecast=True):
    if grouping_vars is None:
        grouping_vars = []
    else:
        if type(grouping_vars) is not list:
            raise TypeError("grouping_vars must be a list")
    if include_forecast:
        if "Future " + exit_col not in grouping_vars:
            grouping_vars = grouping_vars + ["Future " + exit_col]
    return grouping_vars


def plot_CIF(df2, linestyles=None, grouping_vars=None):
    # Note that this plotting function is for the specific case generated from our simulations only.
    # Different data will necessitate a modified plotting function.
    grouping_vars = grouping_vars_subfcn(grouping_vars=grouping_vars, include_forecast=False)
    if linestyles is None:
        linestyles = {'X': 'solid', 'Y': 'dashed', 'Z': 'dotted'}
    xlims = [(min(df2["Period"])-1), (max(df2["Period"])+1)]
    xticks = np.sort(df2["Period"].unique())
    if len(grouping_vars) == 0:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 3), sharey=True)
        for label, a in df2.groupby("Future exit_type"):
            linestyle = linestyles[label]
            a.plot(x='Period', y='CIF', ax=axes, label=label, linestyle=linestyle)
            axes.set_title("$CIF$ by exit type")
            axes.set_xlim(xlims)
            axes.set_xticks(xticks)
        axes.set_ylabel(r'$P(0 < T_{i} \leq h, D=d|...)$')
    elif len(grouping_vars) > 0:
        grouped = df2.groupby(grouping_vars)
        key_label = " ".join(k for k in grouping_vars)
        ncols = len(grouped.groups.keys())
        fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(4*ncols, 3), sharey=True)
        for (key, ax) in zip(grouped.groups.keys(), axes.flatten()):
            for label, a in grouped.get_group(key).groupby("Future exit_type"):
                linestyle = linestyles[label]
                a.plot(x='Period', y='CIF', ax=ax, label=label, linestyle=linestyle)
                ax.set_title("$CIF$, " + key_label + "=" + str(key) + ", by exit type")
                ax.set_xlim(xlims)
                ax.set_xticks(xticks)
        axes[0].set_ylabel(r'$P(0 < T_{i} \leq h, D=d|...)$')
    plt.tight_layout()
    plt.show()





if __name__ == "__main__":
    df = fabricate_data(N_PERSONS=1000, N_PERIODS=20, SEED=1234, exit_prob=.4)  # example data
    cif = CIF(df)
    plot_CIF(cif)
    cif = CIF(df, grouping_vars=["X1"])
    plot_CIF(cif, grouping_vars=["X1"])

