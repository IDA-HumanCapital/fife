import numpy as np
import CIF.CIF as cif
from tests_performance.Data_Fabrication import fabricate_data

# make some data
df0 = fabricate_data(N_PERSONS=1000, N_PERIODS=20, SEED=1234, exit_prob=.4)

#####
# example 1 - aggregate CIF
df = df0.copy()
dfcif = cif.CIF(df, exit_col="exit_type")
cif.plot_CIF(dfcif, exit_col="exit_type", linestyles={'X': 'solid', 'Y': 'dashed', 'Z': 'dotted'})

#####
# example 2 - examine CIF aggregated by groups
df = df0.copy()
dfcif = cif.CIF(df, grouping_vars=["X1"], exit_col="exit_type")
cif.plot_CIF(dfcif, grouping_vars=["X1"], exit_col="exit_type")

#####
# example 3 - more than one exit column
# first add a 2nd exit column for demonstration
df = df0.copy()
df["exit_type2"] = "No_exit"
for i in df["ID"].unique():
    if "No_exit" not in df[df["ID"] == i]["exit_type"]:
        temp = np.random.choice(["I", "V"], p=[.3, .7])
        df.loc[df["ID"] == i, "exit_type2"] = temp

seed = 1234  # Only used in this example to show that the computationally efficient method and the easy method produce the same results.  Without setting the seed, the training and validation set will differ, resulting in slightly different results.

# now the procedure - computationally efficient method
grouping_vars = ["X1"]
dfp, ID = cif.process_data(df, seed=seed)
dfps = dfp.drop(columns=["exit_type", "exit_type2"])
dfp1 = dfp.drop(columns="exit_type2")
dfp2 = dfp.drop(columns="exit_type")

fs, ms = cif.get_forecast(dfps, modeler="LGBSurvivalModeler", process_data_first=False, return_model=True)
fe1 = cif.get_forecast(dfp1, modeler="LGBExitModeler", exit_col='exit_type', process_data_first=False)
fe2 = cif.get_forecast(dfp2, modeler="LGBExitModeler", exit_col='exit_type2', process_data_first=False)
# now create the CIF; notice that we pass in the original dataframe df here because we have already produced the forecasts.  Passing in the processed data will work too, but will produce additional rows with NaN which must be removed.
dfcif1A = cif.CIF(df, f0=fs, f1=fe1, ID=ID, exit_col="exit_type", grouping_vars=grouping_vars)
dfcif2A = cif.CIF(df, f0=fs, f1=fe2, ID=ID, exit_col="exit_type2", grouping_vars=grouping_vars)

# now the procedure - computationally inefficient method but a lot easier
df1 = df.drop(columns="exit_type2")
df2 = df.drop(columns="exit_type")
# Observe that here we must pass in the relevant data frames, df1 and df2, in order to correctly produce forecasts.
dfcif1B = cif.CIF(df1, grouping_vars=grouping_vars, exit_col="exit_type", seed=seed)
dfcif2B = cif.CIF(df2, grouping_vars=grouping_vars, exit_col="exit_type2", seed=seed)

# verify that the methods agree
max(abs(dfcif1A["CIF"] - dfcif1B["CIF"]))
max(abs(dfcif2A["CIF"] - dfcif2B["CIF"]))
dfcif1A.ne(dfcif1B).sum()
dfcif2A.ne(dfcif2B).sum()

# plot them
cif.plot_CIF(dfcif1B, grouping_vars=grouping_vars, exit_col="exit_type",
         linestyles={'X': 'solid', 'Y': 'dashed', 'Z': 'dotted'})
cif.plot_CIF(dfcif2B, grouping_vars=grouping_vars, exit_col="exit_type2", linestyles={'I': 'solid', 'V': 'dashed'})

#####
# example 4 - model diagnostics:
# In addition to examining the CIF, you may wish to examine model diagnostics on the estimated survival and exit models.  You can build the models prior to forecasts with get_model():
df = df0.copy()
seed = 1234
dfp, ID = cif.process_data(df, seed=seed)
dfps = dfp.drop(columns=["exit_type"])
ms = cif.get_model(dfps, modeler="LGBSurvivalModeler", process_data_first=False)
# You may now perform model diagnostics using the modeler objects ms, me1, and me2.
ms.evaluate()
# Once you already have the models, you don't need to build them again.  You can just pass the modeler to get_forecast in place of the data frame along with the argument build_model=False:
fs = cif.get_forecast(ms, build_model=False)

#####
# example 5 - passing additional instructions to  PanelDataProcessor, LGBSurvivalModeler, and LGBExitModeler:
# Do do this, simply put the instructions in a dictionary of the form {<arg1>: <value1>, <arg2>: <value2>, <kwarg1>: <kwvalue1>, ...} and pass this dictionary to CIF(), get_forecasts(), get_forecast(), or process_data() as appropriate, using the arguments PDPkwargs=..., Survivalkwargs=..., Exitkwargs=....
# This allows any and all possible allowable options to be passed directly to the processor and modelers.
df = df0.copy()
seed = 1234
grouping_vars = None
dfp0, ID = cif.process_data(df, seed=seed)
dfp1, ID = cif.process_data(df, PDPkwargs={"config": {"TEST_INTERVALS": 4}}, seed=seed)
# etc.
# or
dfcif = cif.CIF(df, grouping_vars=grouping_vars, exit_col="exit_type", seed=seed, PDPkwargs={"config": {"TEST_INTERVALS": 4}})
# etc.

# These will still only produce forecasts for the censored observations at the last period in the data set.
# You might want to calculate the CIF for individuals in the test set to illustrate the forecasted vs the empirical exit probabilities as an additional illustration of model performance.
# The following illustrates this.
df = df0.copy()
seed = 1234
grouping_vars = ["X1"]
dfp1, ID = cif.process_data(df, PDPkwargs={"config": {"TEST_INTERVALS": 4}}, seed=seed)
# First, find the test set flagged by the data processor:
test_period_start = min(dfp1[dfp1["_test"] == True]["period"])
test_set_predict_obs = (dfp1["_test"] == True) & (dfp1["period"] == test_period_start)
# Then define a new column that flags the first period of each ID in the test set
dfp1["_predict_obs_test_set"] = test_set_predict_obs
# Now build the needed models:
ms = cif.get_model(dfp1, modeler="LGBSurvivalModeler", exit_col='exit_type', process_data_first=False)
me = cif.get_model(dfp1, modeler="LGBExitModeler", exit_col='exit_type', process_data_first=False)
# Now we have to tell the models which columns flags the observations we want to forecast on.  Note, you could skip this next step by instead passing Survivalkwargs={"predict_col": "_predict_obs_test_set"} and Exitkwargs={"predict_col": "_predict_obs_test_set"} to the respective get_model() calls.
ms.predict_col = "_predict_obs_test_set"
me.predict_col = "_predict_obs_test_set"
# Next, pass the models to get_forecast() using the flag build_model=False, since we have already built and modified the model
fs = cif.get_forecast(ms, build_model=False)
fe = cif.get_forecast(me, build_model=False)
# Finally, we'll pass the forecasts to CIF() along with the original data and the argument subset=test_set_predict_obs
dfcif = cif.CIF(df, f0=fs, f1=fe, ID=ID, grouping_vars=grouping_vars, exit_col="exit_type", subset=test_set_predict_obs)

# Now we can plot the forecasted CIF.
cif.plot_CIF(dfcif, grouping_vars=grouping_vars, exit_col="exit_type")
# Note that I do not cover how to calculate the empirical CIF on the test set here.  I should come back and add that in.


#####
# example 6 - getting the individual level CIF forecasts
# By default, CIF() calculates the mean CIF over groups specified by the grouping_vars argument.  To get individual level forecasts, you can use the individual_level=True argument:
df = df0.copy()
dfcif = cif.CIF(df, grouping_vars=["X1"], exit_col="exit_type", individual_level=True)

# This works along with the topics in the other examples as well.  For example, you may need the models separately for evaluation, or you may wish to forecast on the test set for comparison, or you may need to produce forecasts separately due to the presence of multiple outcome columns.
seed = 1234
grouping_vars = ["X1"]
dfp, ID = cif.process_data(df, seed=seed)
dfps = dfp.drop(columns=["exit_type"])
ms = cif.get_model(dfps, modeler="LGBSurvivalModeler", process_data_first=False)
me = cif.get_model(dfp, modeler="LGBExitModeler", process_data_first=False, exit_col="exit_type")
fs = cif.get_forecast(ms, build_model=False)
fe = cif.get_forecast(me, build_model=False)
dfcif = cif.CIF(df, f0=fs, f1=fe, ID=ID, grouping_vars=grouping_vars, exit_col="exit_type", individual_level=True)

dfcif[dfcif["ID"] == dfcif["ID"][0]]




#####
# example 7 - A complete example
seed = 1234
grouping_vars = ["X1"]
df = df0.copy()

# Pass the relevant arguments to the PDP.  Note that these arguments do not need to be passed to the modeler.  If they are passed to the modeler, they should be identical to those passed to the PDP.
# It is not clear to me if the modelers will overwrite what is specified by the PDP, so I would advise NOT passing these arguments to the modelers.
# Note that setting n_intervals is NOT required, and I recommend not using it.  That is, do not use it at all; do not pass it to the PDP or the Modelers.
dfp, ID = cif.process_data(df, PDPkwargs={"config": {"TEST_INTERVALS": 2}}, seed=seed)
# Find the rows corresponding to the start of the test set and make a new column that flags those rows
test_period_start = min(dfp[dfp["_test"] == True]["period"])
test_set_predict_obs = (dfp["_test"] == True) & (dfp["period"] == test_period_start)
dfp["_predict_obs_test_set"] = test_set_predict_obs

# remove the exit type column(s) for the survival modeler
dfps = dfp.drop(columns=["exit_type"])
# similarly, if there are multiple exit_cols, make a df for each by removing the unused columns from dfp

# Build the models on the relevant dfs
ms = cif.get_model(dfps, modeler="LGBSurvivalModeler", exit_col='exit_type', process_data_first=False)
me = cif.get_model(dfp, modeler="LGBExitModeler", exit_col='exit_type', process_data_first=False)

# First, let's get the evaluation information:
eval_s = ms.evaluate()
eval_e = me.evaluate()

# Next, let's get the forecasts for the censored individuals
fs = cif.get_forecast(ms, build_model=False)
fe = cif.get_forecast(me, build_model=False)

# Next, let's get the forecasts for the test set
# To do this, we need to change the modelers' predict_col attribute to the column we created earlier
ms.predict_col = "_predict_obs_test_set"
me.predict_col = "_predict_obs_test_set"
# Now get the forecasts
fs_test = cif.get_forecast(ms, build_model=False)
fe_test = cif.get_forecast(me, build_model=False)

# Now we will use CIF() to merge the relevant forecasts and return some different CIFs

# CIFs for censored individuals
dfcif_indiv = cif.CIF(df, f0=fs, f1=fe, ID=ID, exit_col="exit_type", grouping_vars=grouping_vars, individual_level=True)

# CIFs for censored individuals, grouped by grouping_vars
dfcif_grouped = cif.CIF(df, f0=fs, f1=fe, ID=ID, exit_col="exit_type", grouping_vars=grouping_vars)

# CIFs for individuals in the test set.  Notice that we also use the subset= argument to ensure that we are recording the covariate information from the relevant time periods in the output.
# You might want to calculate the CIF for individuals in the test set to illustrate the forecasted vs the empirical exit probabilities as an additional illustration of model performance.
dfcif_test_indiv = cif.CIF(df, f0=fs_test, f1=fe_test, ID=ID, exit_col="exit_type", grouping_vars=grouping_vars, individual_level=True, subset=test_set_predict_obs)
dfcif_test_grouped = cif.CIF(df, f0=fs_test, f1=fe_test, ID=ID, exit_col="exit_type", grouping_vars=grouping_vars, subset=test_set_predict_obs)

