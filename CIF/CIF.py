"""This file creates Cumulative Incidence Function forecasts by combining forecasts from the LGBSurvivalModeler and the LGBExitModeler"""

from fife.processors import PanelDataProcessor
from fife.lgb_modelers import LGBExitModeler, LGBSurvivalModeler, LGBStateModeler
# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fife.utils import make_results_reproducible


def process_data(df, PDPkwargs=None, seed=None):
    if seed is not None:
        make_results_reproducible(seed)
    if PDPkwargs is None:
        dp = PanelDataProcessor(data=df)
    else:
        if type(PDPkwargs) is not dict:
            raise TypeError("PDPkwargs must be a dictionary")
        dp = PanelDataProcessor(data=df, **PDPkwargs)
    dp.build_processed_data()
    ID = dp.config["INDIVIDUAL_IDENTIFIER"]
    return dp.data, ID


def get_model(df, modeler="LGBSurvivalModeler", exit_col=None, process_data_first=True, PDPkwargs=None, Survivalkwargs=None, Exitkwargs=None, seed=None):
    if seed is not None:
        make_results_reproducible(seed)
    if process_data_first:
        df, ID = process_data(df, PDPkwargs=PDPkwargs)
    if modeler == "LGBSurvivalModeler":
        if exit_col is not None:
            if exit_col in df.columns:
                df = df.drop(columns=exit_col)
        if Survivalkwargs is None:
            m = LGBSurvivalModeler(data=df)
        else:
            if type(Survivalkwargs) is not dict:
                raise TypeError("Survivalkwargs must be a dictionary")
            m = LGBSurvivalModeler(data=df, **Survivalkwargs)
    elif modeler == "LGBExitModeler":
        if Exitkwargs is None:
            if exit_col is None:
                raise TypeError("exit_col must be specified")
            m = LGBExitModeler(data=df, exit_col=exit_col)
        else:
            if type(Exitkwargs) is not dict:
                raise TypeError("Exitkwargs must be a dictionary")
            if "exit_col" in Exitkwargs.keys():
                exit_col = Exitkwargs["exit_col"]
                Exitkwargs.pop("exit_col")
            if exit_col is None:
                raise TypeError("exit_col must be specified")
            m = LGBExitModeler(data=df, exit_col=exit_col, **Exitkwargs)
    else:
        raise NameError('Invalid modeler')
    m.build_model(parallelize=False)
    return m


def get_forecast(df, modeler="LGBSurvivalModeler", exit_col=None, process_data_first=True, PDPkwargs=None, Survivalkwargs=None, Exitkwargs=None, build_model=True, seed=None, return_model=False):
    if build_model:
        m = get_model(df, modeler=modeler, exit_col=exit_col, process_data_first=process_data_first, PDPkwargs=PDPkwargs, Survivalkwargs=Survivalkwargs, Exitkwargs=Exitkwargs, seed=seed)
    else:
        m = df
    f = m.forecast()
    f = f.reset_index()
    if return_model:
        out = (f, m)
    else:
        out = f
    return out


def get_forecasts(df, ID=None, exit_col=None, PDPkwargs=None, Survivalkwargs=None, Exitkwargs=None, seed=None):
    df, PDPID = process_data(df, PDPkwargs=PDPkwargs, seed=seed)
    returnID = False
    if ID is None:
        ID = PDPID
        returnID = True
    f0 = get_forecast(df, modeler="LGBSurvivalModeler", exit_col=exit_col, process_data_first=False, Survivalkwargs=Survivalkwargs)
    f1 = get_forecast(df, modeler="LGBExitModeler", exit_col=exit_col, process_data_first=False, Exitkwargs=Exitkwargs)
    f = f0.merge(f1, how="outer", on=ID)
    if returnID:
        out = (f, ID)
    else:
        out = f
    return out


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
    temp2 = [i for i in f.columns if "-period Stat" in i]
    if (len(temp) == 0):
        raise ValueError("Survival Probability Forecasts are missing.")
    if (len(temp2) == 0):
        raise ValueError("State Probability Forecasts are missing.")
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


def get_features_and_collapse(d0, f, grouping_vars=None, exit_col="exit_type", ID=None, subset=None, individual_level=False):
    if ID is None:
        raise TypeError("ID was not specified, so the merge cannot be completed.")
    grouping_vars = grouping_vars_subfcn(grouping_vars=grouping_vars, exit_col=exit_col)
    # need to merge with data frame d0[d0p[m.predict_col] == True], so pass the predict_obs column as an input to here
    if subset is None:
        temp = d0.groupby(ID).tail(1)  # may need to sort by period first to get the last row
    else:
        temp = d0[subset]
    out = f.merge(temp, how="left", on=ID)
    if individual_level:
        df = out
    else:
        df = out.groupby(grouping_vars).mean()
        df = df.reset_index()
    return df


def wide_to_long(df, grouping_vars=None, exit_col="exit_type", ID=None, individual_level=False):
    grouping_vars = grouping_vars_subfcn(grouping_vars=grouping_vars, exit_col=exit_col)
    temp = [i for i in df.columns if "-CIF" in i]
    num = sorted([int(x.split('-')[0]) for x in temp])
    if individual_level:
        if ID is None:
            raise ValueError("You must specify the ID column with ID=")
        else:
            if ID not in grouping_vars:
                grouping_vars = [ID] + grouping_vars
    df2 = df.melt(id_vars=grouping_vars, value_vars=[str(n) + '-CIF' for n in num], value_name="CIF", var_name="Period")
    df2["Period"] = df2["Period"].apply(lambda x: int(x.split('-')[0]))
    return df2


def CIF(d0, f0=None, f1=None, ID=None, grouping_vars=None, exit_col=None, PDPkwargs=None, Survivalkwargs=None, Exitkwargs=None, seed=None, subset=None,individual_level=False):
    if (f0 is None) | (f1 is None):
        if (f0 is None) & (f1 is None):
            f = get_forecasts(d0, ID=ID, exit_col=exit_col, PDPkwargs=PDPkwargs, Survivalkwargs=Survivalkwargs, Exitkwargs=Exitkwargs, seed=seed)
            if ID is None:
                ID = f[1]
                f = f[0]
        else:
            if f0 is not None:
                f = f0
            else:
                f = f1
            temp = [i for i in f.columns if "-period Surv" in i]
            temp2 = [i for i in f.columns if "-period Stat" in i]
            if (len(temp) == 0) | (len(temp2) == 0):
                raise ValueError("It appears you only passed one forecast.  You must pass both survival probability and exit type probability forecasts using the arguments f0= and f1=")
    else:
        if ID is None:
            raise TypeError("You specified both forecasts, but did not specify the ID for merging them.  Use the arg ID= to specify the ID.")
        f = f0.merge(f1, how="outer", on=ID)
    grouping_vars = grouping_vars_subfcn(grouping_vars=grouping_vars, exit_col=exit_col)
    f = calc_CIF(f)
    df = get_features_and_collapse(d0, f, grouping_vars=grouping_vars, exit_col=exit_col, ID=ID, subset=subset, individual_level=individual_level)
    df2 = wide_to_long(df, grouping_vars=grouping_vars, exit_col=exit_col, ID=ID, individual_level=individual_level)
    return df2


def grouping_vars_subfcn(grouping_vars=None, exit_col=None, include_forecast=True):
    if exit_col is None:
        raise TypeError("You must specify exit_col")
    if grouping_vars is None:
        grouping_vars = []
    else:
        if type(grouping_vars) is not list:
            raise TypeError("grouping_vars must be a list")
    if include_forecast:
        if "Future " + exit_col not in grouping_vars:
            grouping_vars = grouping_vars + ["Future " + exit_col]
    return grouping_vars


def plot_CIF(df2, linestyles=None, grouping_vars=None, exit_col=None):
    if exit_col is None:
        raise TypeError("You must specify exit_col")
    # Note that this plotting function is for the specific case generated from our simulations only.
    # Different data will necessitate a modified plotting function.
    grouping_vars = grouping_vars_subfcn(grouping_vars=grouping_vars, include_forecast=False, exit_col=exit_col)
    if linestyles is None:
        linestyles = {temp[0]: "solid" for temp in df2.groupby("Future " + exit_col)}
    xlims = [(min(df2["Period"])-1), (max(df2["Period"])+1)]
    xticks = np.sort(df2["Period"].unique())
    if len(grouping_vars) == 0:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 3), sharey=True)
        for label, a in df2.groupby("Future " + exit_col):
            linestyle = linestyles[label]
            a.plot(x='Period', y='CIF', ax=axes, label=label, linestyle=linestyle)
            axes.set_title("$CIF$ by exit type")
            axes.set_xlim(xlims)
            axes.set_xticks(xticks)
        # axes.set_ylabel(r'$P(0 < T_{i} \leq h, D=d|...)$')
        axes.set_ylabel('Probability')
    elif len(grouping_vars) > 0:
        grouped = df2.groupby(grouping_vars)
        key_label = " ".join(k for k in grouping_vars)
        ncols = len(grouped.groups.keys())
        fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(4*ncols, 3), sharey=True)
        for (key, ax) in zip(grouped.groups.keys(), axes.flatten()):
            for label, a in grouped.get_group(key).groupby("Future " + exit_col):
                linestyle = linestyles[label]
                a.plot(x='Period', y='CIF', ax=ax, label=label, linestyle=linestyle)
                ax.set_title("$CIF$, " + key_label + "=" + str(key) + ", by exit type")
                ax.set_xlim(xlims)
                ax.set_xticks(xticks)
        # axes[0].set_ylabel(r'$P(0 < T_{i} \leq h, D=d|...)$')
        axes[0].set_ylabel('Probability')
    plt.tight_layout()
    plt.show()





