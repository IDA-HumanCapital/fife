FIFE can be used to forecast time-to-event under a competing risks framework. Competing risks analysis is useful when you desire to predict not only when a terminal event occurs, but which terminal event occurs. Forecasts from the survival modeler and the exit modeler can be combined to produce cause-specific hazard probabilities and cumulative incidence function forecasts. An alternative approach is to model the subdistribution of risk, which is not currently implemented in FIFE (Schmid and Berger, 2020).

Suppose there are :math:`M` different terminal events that occur at times :math:`T_{i1}, T_{i2}, ..., T_{iM}` for individual :math:`i` where each time shares a common time scale and starting period. The period of analysis terminates at :math:`T_{iM}=T_{M}`, representing a right-censoring time that is shared for all individuals. Thus :math:`T_M=\underset{i}{max}\;T_i`.  For each individual we only observe the first occurring event :math:`T_i=min(T_{i1}, T_{i2}, ..., T_{iM})` and which event occurred :math:`d_i=\underset{d\in\{1,2,...,M\}}{argmin}(T_{i1}, T_{i2}, ..., T_{iM})`. After time :math:`T_i` individual :math:`i` is no longer observed in the data. 

#### Dataset for training and prediction

The dataset format is similar to the format as used for the single-risk case for `LGBSurvivalModeler` except that a column containing the type of terminal event for the individual must be provided as an argument to `exit_col` in `LGBExitModeler`. Currently, exit modeling is only supported for LightGBM. 

The REIGN dataset as an example for predicting not just when a leader's reign will end but how their rein will end (see https://oefresearch.org/datasets/reign for details). The reign can end by the government type remaining the same (e.g., remains as presidential democracy, USA 2016), government type changing (e.g., changes from personal dictatorship to warlordism, Yemen 2015), or a complete country dissolution (e.g., Yugoslavia 1991). 

.. code-block:: python

	from fife.lgb_modelers import LGBSurvivalModeler
	from fife.lgb_modelers import LGBExitModeler
	from fife.processors import PanelDataProcessor
	from fife.utils import make_results_reproducible, plot_shap_values
	from pandas import concat, date_range, read_csv, to_datetime, Index, DataFrame
	from numpy import array, repeat, nan, ones
	
	# import warnings
	#warnings.filterwarnings('ignore')
	
	SEED = 9999
	make_results_reproducible(SEED)
	
	data = read_csv("https://www.dl.dropboxusercontent.com/s/3tdswu2jfgwp4xw/REIGN_2020_7.csv?dl=0")

	# Define individual and time units
	data["country-leader"] = data["country"] + ": " + data["leader"]
	data["year-month"] = data["year"].astype(int).astype(str) + data["month"].astype(int).astype(str).str.zfill(2)
	data["year-month"] = to_datetime(data["year-month"], format="%Y%m")
	data = concat([data[["country-leader", "year-month"]],
        	       data.drop(["ccode", "country-leader", "leader", "year-month"],
                	         axis=1)], axis=1)
	
	data = data.drop_duplicates(["country-leader", "year-month"], keep="first")
	data.head()
	```
	Out[1]: 
	country-leader year-month country    year  month  ...  pt_suc  pt_attempt    precip  couprisk  pctile_risk
	0    USA: Truman 1950-01-01     USA  1950.0    1.0  ...     0.0         0.0 -0.069058       NaN          NaN
	1    USA: Truman 1950-02-01     USA  1950.0    2.0  ...     0.0         0.0 -0.113721       NaN          NaN
	2    USA: Truman 1950-03-01     USA  1950.0    3.0  ...     0.0         0.0 -0.108042       NaN          NaN
	3    USA: Truman 1950-04-01     USA  1950.0    4.0  ...     0.0         0.0 -0.041600       NaN          NaN
	4    USA: Truman 1950-05-01     USA  1950.0    5.0  ...     0.0         0.0 -0.129703       NaN          NaN
	
	[5 rows x 38 columns]
	​```
	
	# Process data for model input
	data_processor = PanelDataProcessor(data=data, config={'NUMERIC_SUFFIXES':	['year','month','age','tenure_months','lastelection','loss','irregular','prev_conflict','precip','couprisk','pctile_risk']})
	data_processor.build_processed_data()
	
	mdata = data_processor.data.copy()

The REIGN data does not come with columns indicating the type of terminal event, thus such a column needs to be engineered. The terminal event type column must be a categorical data type. Only the value from the last observation in the terminal event type column is used, any preceding values within an individual are ignored. FIFE's `build_processed_data()` identifies unique individuals by their spells (or broken panels). That is if an individual leaves the dataset and reenters, they are treated as two different individuals. Therefore, terminal event types need to be provided at the last time period for each individual-spell. This can be accomplished using the `_period` column output with `build_processed_data()` as shown below.

.. code-block:: python

	# Assign unique ID based off broken panels
	gaps = mdata.groupby("country-leader")["_period"].shift() < mdata["_period"] - 1
	spells = gaps.groupby(mdata["country-leader"]).cumsum()
	mdata['country-leader-spell'] = mdata['country-leader'] + array(spells, dtype = 'str')
	
	## Create exit status for dissolution, change of government type, and same government type
	mdata['outcome'] = 'No exit'
	
	for i in mdata['country-leader-spell'].unique():
	    # Boolean index for the unique country-leader-spell observations
	    dex1 = mdata['country-leader-spell'] == i
	    # Boolean index for the last observation for the country-leader-spell observations
	    dex2 = dex1 & (mdata['year-month'] == mdata.loc[dex1,'year-month'].max())
	    # Boolean index for the last observation for the country-leader-spell observations unless it is censored
	    dex3 = dex2 & (mdata['year-month'] != mdata['year-month'].max())
	    # Only assign exit status if not censored
	    if dex3.sum() > 0:
	        futuredex = (mdata['year-month'] >= mdata.loc[dex3,'year-month'].values[0]) & (mdata['country'] == mdata.loc[dex3,"country"].values[0]) & ~dex1
	        if futuredex.sum() == 0:
	            mdata.loc[dex2,'outcome'] = "Dissolution" # could alternatively use dex1
	        else:
	            oldgovernment = mdata.loc[dex3,'government'].values[0]
	            futuremdata = mdata[futuredex]
	            newgovernment = futuremdata.loc[futuremdata['year-month'] == futuremdata['year-month'].min(),'government'].values[0]
	            if oldgovernment == newgovernment:
	                mdata.loc[dex2,'outcome'] = "Same government type" # could alternatively use dex1
	            else:
	                mdata.loc[dex2,'outcome'] = "Change of government type" # could alternatively use dex1
	
	# The outcome variable must be a category type
	mdata['outcome'] = mdata['outcome'].astype('category')

We can do additional feature engineering such as including features that count the number of new leaders and changes of government types in the last 10 and 20 years.

.. code-block:: python

	# Feature engineering (optional)
	# Include number of new leaders in last 10 and 20 years
	# Include number of new governments types in last 10 and 20 years
	mdata['new_leader_10'] = nan
	mdata['new_leader_20'] = nan
	mdata['new_govt_10'] = nan
	mdata['new_govt_20'] = nan 
	
	for i in mdata['country'].unique():
	    dex4 = mdata['country'] == i
	    for j in mdata.loc[dex4,'year'].unique():
	        if mdata.loc[dex4,'year'].min() <= j - 10:
	           mdata.loc[dex4 & (mdata['year'] == j), 'new_leader_10'] = mdata.loc[dex4 & (mdata['year'] >= j - 10) & (mdata['year'] < j), 'country-leader-spell'].nunique()
	           mdata.loc[dex4 & (mdata['year'] == j), 'new_govt_10'] = mdata.loc[dex4 & (mdata['year'] >= j - 10) & (mdata['year'] < j), 'government'].nunique()
	           if mdata.loc[dex4,'year'].min() <= j - 20:
	               mdata.loc[dex4 & (mdata['year'] == j), 'new_leader_20'] = mdata.loc[dex4 & (mdata['year'] >= j - 20) & (mdata['year'] < j), 'country-leader-spell'].nunique()
	               mdata.loc[dex4 & (mdata['year'] == j), 'new_govt_20'] = mdata.loc[dex4 & (mdata['year'] >= j - 20) & (mdata['year'] < j), 'government'].nunique()
	            
	# We no longer need country-leader-spell 
	mdata = mdata.drop(['country-leader-spell'], axis = 1)

#### Training and forecasting

##### Survival probabilities

Now that the data is prepared, we can train and forecast with the survival modeler, `LGBSurvivalModeler`, to obtain the survival probabilities. The survival probabilities are defined to be 

.. math::

   Pr(T_i\geq t|X_{i}).

Forecasts are produced for censored individuals with `.forecast()` up to the furthest observed time period after the initial period, :math:`t\in\{T_M+1,T_M+2,...,2\cdot T_M\}`. The features used are from the last observed row, :math:`X_{i{T_M}}`. Thus all forecasted probabilities are conditional on the same observed features and are coherent up to numerical accuracy. If a terminal event type column is in the data, it should not be passed to `LGBSurvivalModeler()`.

.. code-block:: python

	# Obtain the survival probability
	survival_modeler = LGBSurvivalModeler(data=mdata.drop('outcome', axis = 1))
	survival_modeler.build_model(parallelize=False)
	survival_modeler_forecasts = survival_modeler.forecast()
	
	# Set custom column names
	custom_forecast_headers = date_range(data["year-month"].max(),
	                                      periods=len(survival_modeler_forecasts .columns),
	                                      freq="M").strftime("%b %Y")
	survival_modeler_forecasts.columns = custom_forecast_headers
	survival_modeler_forecasts
	```
	                           Jul 2020  Aug 2020  Sep 2020  Oct 2020  ...  May 2040  Jun 2040  Jul 2040  Aug 2040
	country-leader                                                     ...                                        
	Afghanistan: Ashraf Ghani  0.986044  0.965914  0.953822  0.942614  ...  0.000322  0.000320  0.000273  0.000271
	Albania: Rama              0.992290  0.984483  0.977156  0.968703  ...  0.008364  0.008308  0.008252  0.008197
	Algeria: Tebboune          0.991351  0.977605  0.965622  0.941865  ...  0.007509  0.007316  0.006988  0.006939
	Andorra: Espot Zamora      0.990101  0.981595  0.975345  0.966500  ...  0.000727  0.000723  0.000718  0.000713
	Angola: Lourenco           0.985230  0.972513  0.959758  0.947597  ...  0.025139  0.024493  0.023396  0.023232
	                            ...       ...       ...       ...  ...       ...       ...       ...       ...
	Venezuela: Nicolas Maduro  0.993150  0.986412  0.982183  0.975254  ...  0.000331  0.000329  0.000321  0.000313
	Vietnam: Phu Trong         0.991983  0.982679  0.968412  0.959021  ...  0.002620  0.002586  0.002209  0.002123
	Yemen: Houthi              0.992832  0.983635  0.950930  0.939746  ...  0.056119  0.055744  0.055367  0.054997
	Zambia: Lungu              0.993150  0.986317  0.981646  0.973240  ...  0.008273  0.006682  0.005709  0.005411
	Zimbabwe: Mnangagwa        0.993150  0.986421  0.981954  0.974727  ...  0.058133  0.057746  0.057357  0.056975
	
	[194 rows x 242 columns]
	```

##### Type of exit probabilities

The same data can be used to train and forecast with the exit modeler, `LGBExitModeler`, where the terminal event type column name is passed as an argument to `exit_col`. The exit modeler produces forecasts for type of exit, conditional on :math:`t` being the last period observed. This is defined formally as

.. math::
   Pr(D_i=d_i|T_i= t,X_{i}).

`LGBExitModeler` follows similar rules as `LGBSurvivalModeler` for selecting observations for training and forecasting. Except for training, where only observations with an observed exit in the desired time period are used.

.. code-block:: python

	# Obtain the probability of type of exit conditional on exit that period
	exit_modeler = LGBExitModeler(data=mdata, exit_col = 'outcome')
	exit_modeler.build_model(parallelize=False)
	exit_modeler_forecasts = exit_modeler.forecast()
	
	# Set MultiIndex and column names
	exit_modeler_forecasts = exit_modeler_forecasts.set_index('Future outcome', append = True, drop = True)
	exit_modeler_forecasts = exit_modeler_forecasts.reindex(['Same government type','Change of government type','Dissolution','No exit'], axis=0,level=1)
	exit_modeler_forecasts.columns = custom_forecast_headers
	exit_modeler_forecasts
	```
	                                                         Jul 2020      Aug 2020  ...      Jul 2040      Aug 2040
	country-leader            Future outcome                                         ...                            
	Afghanistan: Ashraf Ghani Same government type       8.578190e-01  7.704733e-01  ...  7.762426e-01  7.535799e-01
	                          Change of government type  1.419008e-01  2.289023e-01  ...  2.237574e-01  2.464201e-01
	                          Dissolution                2.801606e-04  6.244381e-04  ...  9.282013e-16  9.760585e-16
	                          No exit                    1.004800e-15  1.110349e-15  ...  9.282013e-16  9.760585e-16
	Albania: Rama             Same government type       9.911523e-01  9.692952e-01  ...  7.521312e-01  7.186383e-01
	                                                          ...           ...  ...           ...           ...
	Zambia: Lungu             No exit                    5.816850e-16  5.342558e-16  ...  9.282013e-16  9.760585e-16
	Zimbabwe: Mnangagwa       Same government type       9.176592e-01  8.384861e-01  ...  6.396734e-01  7.186383e-01
	                          Change of government type  8.204236e-02  1.608740e-01  ...  3.603266e-01  2.813617e-01
	                          Dissolution                2.984380e-04  6.399155e-04  ...  1.069257e-15  1.018499e-15
	                          No exit                    1.070355e-15  1.137919e-15  ...  1.069257e-15  1.018499e-15
	```

##### Cumulative Incidence Function

It is often of interest to obtain the Cumulative Incidence Function (CIF) defined as :math:`Pr(T_i\leq t,D_i=d|X_{i})`.  The CIF is obtained using the forecasts from `LGBSurvivalModeler` and `LGBExitModeler` using the relationship

.. math::
	Pr(T_i\leq t,D_i=d|X_{i})&=\sum_{l=T_M+1}^{t}Pr(D_i=d|T_i=l,X_{i})Pr(T_i= l|X_{i})\textrm{, where}\\
	Pr(T_i=l|X_{i})&=Pr(T_i\geq l-1|X_{i})-Pr(T_i\geq l|X_{i}).

Note, :math:`Pr(T_i\geq 0 |X_{i})=1`.

.. code-block:: python

	# Obtain Pr(T=l)
	tmp = concat([DataFrame(ones(survival_modeler_forecasts.shape[0]),index = survival_modeler_forecasts.index),survival_modeler_forecasts.iloc[:,:-1]], axis = 1)
	Pr_T_eq_l_forecasts = tmp.values - survival_modeler_forecasts
	
	# Obtain the cumulative incidence function 
	cumulative_incidence_forecasts = exit_modeler_forecasts.copy()
	cumulative_incidence_forecasts = cumulative_incidence_forecasts * DataFrame(repeat(Pr_T_eq_l_forecasts.values, repeats = mdata['outcome'].nunique(), axis = 0)).values
	cumulative_incidence_forecasts = cumulative_incidence_forecasts.cumsum(axis = 1)
	cumulative_incidence_forecasts
	```
	                                                         Jul 2020      Aug 2020  ...      Jul 2040      Aug 2040
	country-leader            Future outcome                                         ...                            
	Afghanistan: Ashraf Ghani Same government type       1.197189e-02  2.748160e-02  ...  8.427979e-01  8.427992e-01
	                          Change of government type  1.980397e-03  6.588223e-03  ...  1.549343e-01  1.549348e-01
	                          Dissolution                3.909978e-06  1.647998e-05  ...  1.994732e-03  1.994732e-03
	                          No exit                    1.402320e-17  3.637464e-17  ...  9.295344e-16  9.295362e-16
	Albania: Rama             Same government type       7.641719e-03  1.520875e-02  ...  9.458562e-01  9.458959e-01
	                                                          ...           ...  ...           ...           ...
	Zambia: Lungu             No exit                    3.984779e-18  7.635237e-18  ...  6.894410e-16  6.897318e-16
	Zimbabwe: Mnangagwa       Same government type       6.286339e-03  1.192818e-02  ...  7.680164e-01  7.682909e-01
	                          Change of government type  5.620235e-04  1.644481e-03  ...  1.741905e-01  1.742979e-01
	                          Dissolution                2.044422e-06  6.350160e-06  ...  4.357975e-04  4.357975e-04
	                          No exit                    7.332364e-18  1.498897e-17  ...  1.108881e-15  1.109270e-15
	
	[776 rows x 242 columns]
	```
	
	# Plot cumulative incidence function
	plotdf = cumulative_incidence_forecasts.loc["USA: Trump",:].transpose()
	plt.figure()
	plt.ylim(bottom=0,top=1)
	plotdf.plot()

##### Cause-specific hazard probability

You can also obtain the cause-specific probability defined as :math:`Pr(T_i=t,D_i=d|T_i\geq t,X_{i})`. The relationship is

.. math::

	Pr(T_i=t,D_i=d|T_i\geq t,X_{i})=\frac{Pr(D_i=d|T_i=t,X_{i})\cdot Pr(T_i=t|X_{i})}{Pr(T_i\geq t|X_{i})}.


.. code-block:: python

	# Obtain the cause-specific hazard probability
	cause_hazard_forecasts = exit_modeler_forecasts.copy()
	cause_hazard_forecasts = cause_hazard_forecasts * DataFrame(repeat(Pr_T_eq_l_forecasts.values, repeats = mdata['outcome'].nunique(), axis = 0)).values
	cause_hazard_forecasts = cause_hazard_forecasts / DataFrame(repeat(survival_modeler_forecasts.values, repeats = mdata['outcome'].nunique(), axis = 0)).values
	cause_hazard_forecasts
	```
	                                                         Jul 2020      Aug 2020  ...      Jul 2040      Aug 2040
	country-leader            Future outcome                                         ...                            
	Afghanistan: Ashraf Ghani Same government type       1.214134e-02  1.605703e-02  ...  1.323774e-01  5.052472e-03
	                          Change of government type  2.008427e-03  4.770433e-03  ...  3.815871e-02  1.652154e-03
	                          Dissolution                3.965319e-06  1.301359e-05  ...  1.582918e-16  6.544107e-18
	                          No exit                    1.422168e-17  2.314020e-17  ...  1.582918e-16  6.544107e-18
	Albania: Rama             Same government type       7.701094e-03  7.686298e-03  ...  5.117019e-03  4.839527e-03
	                                                          ...           ...  ...           ...           ...
	Zambia: Lungu             No exit                    4.012265e-18  3.701101e-18  ...  1.582918e-16  5.374086e-17
	Zimbabwe: Mnangagwa       Same government type       6.329700e-03  5.719506e-03  ...  4.332986e-03  4.818201e-03
	                          Change of government type  5.659002e-04  1.097358e-03  ...  2.440761e-03  1.886425e-03
	                          Dissolution                2.058524e-06  4.365010e-06  ...  7.242876e-18  6.828658e-18
	                          No exit                    7.382940e-18  7.762008e-18  ...  7.242876e-18  6.828658e-18
	
	[776 rows x 242 columns]
	```

#### References

Schmid, M., & Berger, M. (2020) Competing risks analysis for discrete time‐to‐event data. Wiley Interdisciplinary Reviews: Computational Statistics, e1529.