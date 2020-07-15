The Finite-Interval Forecasting Engine (FIFE) provides machine learning and other models for discrete-time survival analysis and finite time series forecasting.

Suppose you have a dataset that looks like this:

| ID | period | feature_1 | feature_2 | feature_3 | ... |
|----|--------|-----------|-----------|-----------|-----|
| 0  | 2016   | 7.2       | A         | 2AX       | ... |
| 0  | 2017   | 6.4       | A         | 2AX       | ... |
| 0  | 2018   | 6.6       | A         | 1FX       | ... |
| 0  | 2019   | 7.1       | A         | 1FX       | ... |
| 1  | 2016   | 5.3       | B         | 1RM       | ... |
| 1  | 2017   | 5.4       | B         | 1RM       | ... |
| 2  | 2017   | 6.7       | A         | 1FX       | ... |
| 2  | 2018   | 6.9       | A         | 1RM       | ... |
| 2  | 2019   | 6.9       | A         | 1FX       | ... |
| 3  | 2017   | 4.3       | B         | 2AX       | ... |
| 3  | 2018   | 4.1       | B         | 2AX       | ... |
| 4  | 2019   | 7.4       | B         | 1RM       | ... |
| ...| ...    | ...       | ...       |...        | ... |

The entities with IDs 0, 2, and 4 are observed in the dataset in 2019.

* What are each of their probabilities of being observed in 2020? 2021? 2022?
* How reliable can we expect those probabilities to be?
* How do the values of the features guide our predictions?

FIFE answers these and other questions for any "unbalanced panel dataset" - a dataset where entities are observed periodically, but may depart the dataset after varying numbers of periods.

FIFE applies machine learning methods - your choice of a feedforward neural network or gradient-boosted tree models - but requires no methodological knowledge to use. You need only point FIFE to your choice of data and configuration parameters and let FIFE run. FIFE will automatically produce a host of spreadsheets and figures for you.

FIFE modules offer further customization for Python programmers. See [Advanced Usage](#advanced-usage) for examples.

### Quick Start

##### Installation

`pip install fife`

<u>Details</u>

* Install an [Anaconda distribution](https://www.anaconda.com/distribution/) of Python version 3.7.6 or later
* From pip (with no firewall):
	* Open Anaconda Prompt
	* Execute `pip install fife`
	* If that results in an error for installing the SHAP dependency, try `conda install -c conda-forge shap` before `pip install fife`
	* If that results in an error for installing the TensorFlow dependency, try `conda install -c anaconda tensorflow` before `pip install fife`

<u>Alternatives</u>

* From pypi.org (https://pypi.org/project/fife/):
	* Download the `.whl` or `.tar.gz` file
	* Open Anaconda Prompt
	* Change the current directory in Anaconda Prompt to the location where the `.whl` or `.tar.gz` file is saved.<br/>
		Example: `cd C:\Users\insert-user-name\Downloads`
	* Pip install the name of the `.whl` or `.tar.gz` file.<br/>
		Example: `pip install fife-1.1.0-py3-none-any.whl`
* From GitHub (https://github.com/IDA-HumanCapital/fife):
	*	Clone the FIFE repository
	* Open Anaconda Prompt
	* Change the current directory in Anaconda Prompt to the directory of the cloned FIFE repository
	* Execute `python setup.py sdist bdist_wheel`
	* Execute `pip install dist/fife-1.1.0-py3-none-any.whl`

##### Configuration

* Copy your data the directory where you'd like to store your results
* Ensure there are no other data files in the directory
* Ensure that the individual and time identifiers are the leftmost and second-leftmost columns in your data, respectively
* Don't have a data file ready but want to see what FIFE can do?
	* Change the directory of your Anaconda prompt to your chosen directory
	* Execute `python -c "import fife.utils; fife.utils.create_example_data().to_csv('Input_Data.csv', index=False)"`

##### Execution

* Change the directory of your Anaconda prompt to the directory containing your data
* Execute `fife`
* Observe the contents of the FIFE_results folder

### Inputs

FIFE takes as input:
-	An unbalanced panel dataset
-	(Optional) A set of configuration parameters described in the table at the bottom of this document

Configuration parameters may be provided in the JSON file format. If you execute FIFE from a directory with a single file with extension _.json_, FIFE will use configuration parameters from that file. If you execute FIFE from a directory with multiple such files, you will need to specify which file contains your configuration parameters by appending a space and the file name to your command. For example, you will need to execute `fife example_config.json`.

Input data may be provided in any of the following file formats with the corresponding file extension:

| File Format                  | Extension  |
| -----------------------------|------------|
| Feather                      | .feather   |
| Comma-separated values (CSV) | .csv       |
| CSV with gzip compression    | .csv.gz    |
| Pickle 	               | .p or .pkl |
| HDF5 		               | .h5 	    |
| JSON 		               | .json      |

If you execute FIFE from a directory with a single file with any of these extensions except .json, FIFE will use the data from that file. If you execute FIFE from a directory with multiple such files, or if you wish to use a data file in a different directory than the directory where you want to store your results, you will need use the `DATA_FILE_PATH` configuration parameter to specify the path to your data file.

Input data may not contain any of the following feature names:

-	`_duration`
-	`_event_observed`
-	`_maximum_lead`
-	`_period`
-	`_predict_obs`
-	`_test`
-   `_validation`

### Intermediate Files

FIFE produces as intermediate files:
-	*Intermediate/Data/Categorical_Maps.p*: Crosswalks from whole numbers to original values for each categorical feature
-	*Intermediate/Data/Numeric_Ranges.p*: Maximum and minimum values in the training set for each numeric feature
-	*Intermediate/Data/Processed_Data.p*: The provided data after removing uninformative features, sorting, assigning train and test sets, normalizing numeric features, and factorizing categorical features
-	If the `FIXED_EFFECT_FEATURES` configuration parameter contains a list of one or more column names:
	-	*Intermediate/Models/IFE_Model.p*: An interacted fixed effects model for all time horizons
-	One of the following:
	-	*Intermediate/Models/[horizon]-lead_GBT_Model.json*: A gradient-boosted tree model for each time horizon
	-	*Intermediate/Models/FFNN_Model.h5*: A feedforward neural network model for all time horizons

Intermediate files are not intended to be read by humans.

### Outputs

FIFE produces as output:
-	*Output/Tables/Survival_Curves.csv*: Predicted survival curves for each individual observed in the final period of the dataset
-	*Output/Tables/Aggregate_Survival_Bounds.csv*: Expected number of individuals retained for each time horizon with corresponding uncertainty bounds
-	*Output/Tables/Metrics.csv*: Model performance metrics
-	*Output/Tables/Counts_by_Quantile.csv*: Number of individuals retained for each time horizon based on different quantiles of the predicted survival probability
-	If the `FIXED_EFFECT_FEATURES` configuration parameter contains a list of one or more column names:
	-	*Output/Tables/IFE_Metrics.csv*: Performance metrics for an interacted fixed effects model
	-	*Output/Tables/IFE_Counts_by_Quantile.csv*: Number of individuals retained for each time horizon based on different quantiles of the predicted survival probability from an interacted fixed effects model
-	*Output/Tables/Retention_Rates.csv* and *Output/Figures/Retention_Rates.png*: Actual and predicted share of individuals who survived a given number of periods for each period in the dataset
-   In *Output/Figures/*, the following plots of relationships between features and the probabilities of exit for each time horizon conditional on survival to the final period of that time horizon:
	-   *Dependence_[horizon]_lead.png* and *Dependence_RMST.png*
	-   *Importance_[horizon]_lead.png* and *Importance_RMST.png*
	-   *Summary_[horizon]_lead.png* and *Summary_RMST.png*
-	*Output/Logs/Log.txt*: A log of script execution

All files produced by FIFE will overwrite files of the same name in the directory designated by the `RESULTS_PATH` configuration parameter.

##### Survival_Curves.csv
- First column: The identifier of each individual observed in the final time period of the dataset.
- Second column: Each individual's probability of surviving one more period (i.e., appearing in the data once more if the data were to be extended one period beyond the final observed period).
- Third column: Each individual's probability of surviving two more periods.
- N-th column: Each individual's probability of surviving N more periods.

##### Aggregate_Survival_Bounds.csv
- First column: Time horizons over which the survival probabilities of all individuals observed in the final period of the dataset will be aggregated.
- Second column: Expected number of individuals retained for each time horizon.
- Third and fourth columns: Lower and upper two-sided [Chernoff bounds](https://en.wikipedia.org/wiki/Chernoff_bound) to quantify the aggregation uncertainty around the expected number of retained individuals.

##### Metrics.csv
- First column: Durations of forecast time horizons, given in terms of the number of periods observed in the data. Each of the metrics in the subsequent columns (except the overall ["C-index"](https://www.statisticshowto.datasciencecentral.com/c-statistic/)) are calculated over all test set observations for which the given horizon could be observed. For example, if the periods in the data are Jan, Feb, Mar, and Apr, the first column will have values 1, 2, and 3 (more generally, if there are N periods, the values in the first column will be 1 through N-1). Metrics in the row with a 1 in the first column will be based on observations in Jan, Feb, and Mar (or periods 1 through N-1). Metrics in the row with a 2 in the first column will be based on observations in Jan and Feb (or periods 1 through N-2). Metrics in the row with a 3 in the first column will be based on observations in Jan (or periods 1 through N-3).

- Second column: Area Under the Receiver Operating Characteristic (AUROC). The AUROC is, over all pairs of observations where exactly one of the observations survived in the given horizon, the share of those pairs for which the model predicts the survivor was more likely to survive. A totally uninformed model would be expected to produce an AUROC of 0.5 and a perfect model would produce an AUROC of 1.0. A blank value indicates that either all or no test set observations survived the given horizon, so an AUROC could not be calculated.

- Third column: Mean of each individual's predicted probability of surviving through the given horizon, which gives the predicted share of individuals that will survive the horizon.

- Fourth column: Actual share of those individuals that survived the horizon. All else equal, a better model will produce smaller absolute differences between predicted shares and actual shares.

- Fifth column: Number of observations with a predicted survival probability greater than or equal to 0.5 that survived.

- Sixth column: Number of observations with a predicted survival probability greater than or equal to 0.5 that did not survive.

- Seventh column: Number of observations with a predicted survival probability less than 0.5 that survived.

- Eighth column: Number of observations with a predicted survival probability less than 0.5 that did not survive. All else equal, a better model will have larger values in the fifth and eighth columns and smaller values in the sixth and seventh columns.

- Ninth column: Concordance index (or "C-index") over the entire test set, and is therefore a single value. The concordance index is, over all pairs of observations where one observation is known to have survived longer, the share of those pairs for which the model predicts a greater restricted mean survival time (RMST) for the observation that survived longer. An individual's RMST is the number of periods the model expects the individual to survive over the greatest horizon for which the model produced a prediction. Like AUROC, a totally uninformed model would be expected to produce a concordance index of 0.5 and a perfect model would produce a concordance index of 1.0.

- *IFE_Metrics.csv* follows the same format as *Metrics.csv*.

##### Counts_by_Quantile.csv
- First column: Time horizon.
- Second column: The quantile based on predicted probability of survival. For example, if the number of quantiles is given to be 5, quantile 1 represents the 20% (perhaps roughly) of observations predicted least likely to survive the given time horizon among those for which the given horizon could be observed. The number of quantiles is given by the `QUANTILES` configuration parameter.
- Third column: The number of observations for which the individuals in the given quantile survived the given horizon.
- Fourth column: The number of observations in the given quantile. Ties in predicted probability among observations may cause the number of observations to vary across quantile for a given time horizon.
- *IFE_Counts_by_Quantile.csv* follows the same format as *Counts_by_Quantile.csv*.

##### Retention_Rates.csv
- First column: The number of time periods since the earliest period in the data.
- Second column: The share of individuals observed a given number of periods prior who survived to the corresponding period. The number of periods prior is given by the `RETENTION_INTERVAL` configuration parameter.
- Third column: The model prediction of the corresponding actual retention rate in the second column for periods in the data on which the model was trained.
- Fourth column: The predicted retention rate for periods in the data on which the model was not trained. These periods include periods in the data excluded from training by the `TEST_PERIODS` configuration parameter and periods beyond those in the data.

##### Figures

*Retention_Rates.png* plots the contents of *Retention_Rates.csv*.

For gradient-boosted tree modelers there will be other plots in *Output/Figures/* which depict aggregations and subsets of Shapley Additive Explanations (SHAP) values for individuals in the final period of the dataset. A [SHAP value](https://github.com/slundberg/shap/blob/master/README.md) is a measure of how much a specific feature value contributes (positively or negatively) to a specific predicted value. For files with names ending in `_lead`, the predicted value is the probability of surviving the number of additional periods specified in the file name. For files with names ending in `RMST`, the predicted value is the RMST.

Plots with names beginning with `Importance_`: Mean absolute SHAP values for up to 20 features with the highest such means. The greater its mean absolute SHAP value, the more a feature contributes to the survival probability (positively or negatively) in the given period (or RMST), on average. Thus the features at the top of the plot are the most important features.

Plots with names beginning with `Summary_`: SHAP values associated with each of the most important features against the values of those features. A pattern in such a plot indicates an association between the plotted feature and the predicted outcome. For example, points shifting from blue to red from left to right for a given feature indicates that the feature is positively correlated with the predicted outcome.

Plots with names beginning with `Dependence_`: SHAP values associated with the most important feature against the values of that feature and the values of an automatically selected other feature. Different patterns for differently-colored points indicate that the association between the most important feature and the predicted outcome depends on the value of the other feature. For example, if blue points form an upward-sloping line but red points form a downward-sloping line then a low value of the other feature is associated with a negative correlation between the most important feature and the predicted outcome, but a high value of the other feature is associated with a positive correlation.

### Configuration Parameters

You may customize many parameters such as the maximum share of a feature that may be missing and the share of individuals to reserve for model evaluation. You can find an example configuration file, readable with any text editor, in the [FIFE GitHub repository](https://github.com/IDA-HumanCapital/fife). FIFE accepts the following parameters:

##### Input/output
DATA_FILE_PATH; default: `"Input_Data.csv"`; type: String
	A relative or absolute path to the input data file.
NOTES_FOR_LOG; default: `"No config notes specified"`; type: String
	Custom text that will be printed in the log produced during execution.
RESULTS_PATH; default: `"FIFE_results"`; type: String
	The relative or absolute path on which Intermediate and Output folders will be stored. The path will be created if it does not exist.

##### Reproducibility
SEED; default: 9999; type: Integer
	The initializing value for all random number generators. Strongly recommend not changing; changes will not produce generalizable changes in performance.

##### Identifiers
INDIVIDUAL_IDENTIFIER; default: `""` (empty string); type: String
	The name of the feature that identifies individuals that persist over multiple time periods in the data. If an empty string, defaults to the leftmost column in the data.
TIME_IDENTIFIER; default: `""` (empty string); type: String
	The name of the feature that identifies time periods in the data. If an empty string, defaults to the second-leftmost column in the data.

##### Feature types
CATEGORICAL_SUFFIXES; default: `[]` (empty list); type: List of strings
	Optional list of suffixes denoting that columns ending with such a suffix should be treated as categorical. Useful for flagging categorical columns that have a numeric data type and more than MAX_NUM_CAT unique values. Column names with a categorical suffix and a numeric suffix will be identified as categorical.
DATETIME_AS_DATE; default: `true`; type: Boolean
	How datetime features will be represented. If True, datetime features will be converted to integers in YYYYMMDD format. Otherwise, datetime features will be converted to nanoseconds.
MAX_NULL_SHARE; default: 0.999; type: Decimal
	The maximum share of observations that may have a null value for a feature to be kept for training. Larger values may increase run time, risk of memory error, and/or model performance
MAX_UNIQUE_NUMERIC_CATS; default: 1024; type: Integer
	The maximum number of unique values for a feature of a numeric type to be considered categorical. Larger values may increase or decrease performance and/or increase run time.
NUMERIC_SUFFIXES; default: `[]` (empty list); type: List of strings
	Optional list of suffixes denoting that columns ending with such a suffix should be treated as numeric. Useful for flagging columns that have a numeric data type and fewer than MAX_NUM_CAT unique values. Column names with a categorical suffix and a numeric suffix will be identified as categorical.

##### Training set
MIN_SURVIVORS_IN_TRAIN; default: 64; type: Integer
	The minimum number of training set observations surviving a given time horizon for the model to be trained to make predictions for that time horizon.
TEST_PERIODS; default: 0; type: Integer
	The number of most recent periods excluded from training and used to evaluate model performance in periods after those in the training set. Larger values may decrease model performance and run time and/or increase evaluation precision.
VALIDATION_SHARE; default: 0.25; type: Decimal
	The share of individuals excluded from training and used to decide when to stop training and to evaluate the model in the same periods as in the training set. Larger values may decrease model performance and run time and/or increase evaluation precision.

##### Modeler types
TREE_MODELS; default: `true`; type: Boolean
	Whether FIFE will train gradient-boosted trees, as opposed to a neural network.
FIXED_EFFECT_FEATURES; default: `[]` (empty list); type: List of strings
	Optional list of column names of features to be used to train an interacted fixed effects (IFE) model in addition to tree models or a neural network. An IFE model will not be trained if list is empty.

##### General hyperparameters
MAX_EPOCHS; default: 256; type: Integer
	The maximum number of passes through the training set. Larger values may increase run time and/or model performance.
PATIENCE; default: 4; type: Integer
	The number of passes through the training dataset without improvement in validation set performance before training is stopped early. Larger values may increase run time and/or model performance.

##### Neural network hyperparameters
BATCH_SIZE; default: 512; type: Integer
	The number of observations per batch of data fed to the machine learning model for training. Larger values may decrease model performance and/or run time.
DENSE_LAYERS; default: 2; type: Integer
	The number of dense layers in the neural network. Larger values may increase model performance and/or run time.
DROPOUT_SHARE; default: 0.25; type: Decimal
	The probability of a densely connected node of the neural network being set to zero weight during training. Larger values may increase or decrease model performance.
EMBED_EXPONENT; default: 0; type: Decimal
	The ratio of the natural logarithm of the number of embedded values to the natural logarithm of the number of unique categories for each categorical feature. Larger values may increase run time, risk of memory error, and/or model performance.
EMBED_L2_REG; default: 2.0; type: Decimal
	The L2 regularization coefficient for each embedding layer. Larger values may increase or decrease model performance.
NODES_PER_DENSE_LAYER; default: 512; type: Integer
	The number of nodes per dense layer in the neural network. Larger values may increase model performance and/or run time.
NON_CAT_MISSING_VALUE; default: -1; type: Decimal
	The value used to replace missing values of numeric features. Outside the [-0.5, 0.5] domain of normalized values as recommended by Chollet (2018) "Deep Learning with Python" p. 102
PROPORTIONAL_HAZARDS; default: `false`; type: Boolean
	Whether the capability will restrict the neural network to a proportional hazards model.

##### Metrics
QUANTILES; default: 5; type: Integer
	The number of similarly-sized bins for which survival and total counts will be reported for each time horizon. Larger values may increase run time and/or evaluation precision.
RETENTION_INTERVAL; default: 1; type: Integer
	The number of periods over which retention rates are computed.
SHAP_PLOT_ALPHA; default: 0.5; type: Decimal
	The transparency of points in SHAP plots. Larger values may increase visibility of non-overlapped points and/or decrease visibility of overlapped points.
SHAP_SAMPLE_SIZE; default: 128; type: Integer
	The number of observations randomly sampled for SHAP value calculation and plotting. Larger values may increase SHAP plot representativeness and/or run time.

### Advanced Usage

The [Quick Start](#quick-start) guide describes the default FIFE pipeline. You can view the source code for that pipeline as follows:

```python
import fife.__main__
import inspect
print(inspect.getsource(fife.__main__))
```

You can use FIFE modules to customize your own pipeline, whether starting from scratch or by copying and modifying `fife.__main__`. We'll want a module to process our panel data and another module to use the processed data to train a model. For this example we'll build gradient-boosted tree models using [LightGBM](https://lightgbm.readthedocs.io/en/latest/).

```python
from fife.processors import PanelDataProcessor
from fife.lgb_modelers import GradientBoostedTreesModeler
```

We'll need to supply a data file.

```python
from fife.utils import create_example_data
data = create_example_data()
```

Let's ensure the reproducibility of our results.

```python
from fife.utils import make_results_reproducible
make_results_reproducible()
```

The Panel Data Processor will make our dataset ready for modeling, to include sorting by individual and time, dropping degenerate and duplicated features, and computing survival durations and censorship status. We can specify processing parameters in the `config` argument. Perhaps we'd rather have an 80/20 validation split than the default 75/25. See the [Configuration Parameters](#configuration-parameters) section for more details.

```python
data_processor = PanelDataProcessor(config={'VALIDATION_SHARE': 0.2},
                                    data=data)
data_processor.build_processed_data()
```

We can access the processed data as `data_processor.data` for input to a modeler. Let's use a gradient-boosted trees modeler. We can specify modeling parameters in the config argument. Perhaps we prefer a patience value of 8 rather than the default value of 4.

```python
gbt_modeler = GradientBoostedTreesModeler(config={'PATIENCE': 8},
                                          data=data_processor.data)
gbt_modeler.build_model()
```

In the case of gradient-boosted trees, our discrete-time survival "model" is actually a list of models, one for each time horizon. The first model produces a probability of survival through the first future period, the second model produces a probability of survival through the second future period conditional on survival through the first future period, and so on.

We can access the list of models as an attribute of our modeler. For example, we can see how many trees are in each model. Our `PATIENCE` parameter value of 8 configured each model to train until model performance on the validation set does not improve for eight consecutive trees (or no more splits can improve fit). That means the number of trees can vary across models.

```python
[i.num_trees() for i in gbt_modeler.model]
```

We can also observe the relative importance of each feature, in this case in terms of the number of times the feature was used to split a tree branch.

```python
dict(zip(gbt_modeler.model[0].feature_name(),
         gbt_modeler.model[0].feature_importance()))
```

In general we are most interested in forecasts for individuals that are still in the population in the final period of the data. The `forecast` method gives us exactly those forecasts. The survival probabilities in the forecasts are cumulative - they are not conditional on survival through any prior future period. See the description of [*Survival_Curves.csv*](#survival_curves.csv) for more details.

```python
gbt_modeler.forecast()
```

The `evaluate` method offers a suite of performance metrics specific to each time horizon as well as the concordance index over the restricted mean survival time. See the description of [*Metrics.csv*](#metrics.csv) above for more details. We can pass a Boolean mask to `evaluate` to obtain metrics only on the validation set. If we specified one or more test periods in our processor configuration we could have used `gbt_modeler.data['_test']` instead.

```python
gbt_modeler.evaluate(gbt_modeler.data['_validation'])
```

In each time period, what share of the observations two periods past would we expect to still be around? See the description of [_Retention_Rates.csv_](#retention_rates.csv) for more details.
```python
gbt_modeler.tabulate_retention_rates(2)
```

Other modelers define different ways of using data to create forecasts, but they all support the methods `build_model`, `forecast`, `evaluate`, `tabulate_retention_rates`, and more.

### Acknowledgement

FIFE uses the `nnet_survival` module of Gensheimer, M.F., and Narasimhan, B., "A scalable discrete-time survival model for neural networks," *PeerJ* 7 (2019): e6257. The `nnet_survival` version packaged with FIFE is GitHub commit d5a8f26 on Nov 18, 2018 posted to https://github.com/MGensheimer/nnet-survival/blob/master/nnet_survival.py. `nnet_survival` is licensed under the MIT License. The FIFE development team modified lines 12 and 13 of nnet_survival for compatibility with TensorFlow 2.0.

### Project Information

The Institute for Defense Analyses (IDA) developed FIFE on behalf of the U.S. Department of Defense, Office of the Under Secretary of Defense for Personnel and Readiness. Among other applications, FIFE is used to produce IDA's Retention Prediction Model.

FIFE has also been known as the Persistence Prediction Capability (PPC).

### Contact Information

- IDA development team: humancapital@ida.org
- IDA Principal Investigator: Dr. Julie Lockwood, jlockwood@ida.org, 703-578-2858

### Alternative Installation

If firewall, air gap, or other issues prevent you from installing FIFE, please contact us at humancapital@ida.org. We know your pain. We offer in-person assistance in the Washington, D.C. metropolitan area. If you are unable to install Python, we can send you an executable version of FIFE.

### License

Copyright (c) 2018 - 2020, Institute for Defense Analyses (IDA).
All rights reserved.

FIFE is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.

FIFE is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

### Citation
Please cite FIFE as:

Institute for Defense Analyses. **FIFE: Finite-Interval Forecasting Engine [software].** https://github.com/IDA-HumanCapital/fife, 2020. Version 1.x.x.

BibTex:
```
@misc{FIFE,
  author={Institute for Defense Analyses},
  title={{FIFE}: {Finite-Interval Forecasting Engine [software]}},
  howpublished={https://github.com/IDA-HumanCapital/fife},
  note={Version 1.x.x},
  year={2020}
}
```

This document was most recently updated 12 June 2020.
