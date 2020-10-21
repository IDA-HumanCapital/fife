The Finite-Interval Forecasting Engine (FIFE) Python package provides machine learning and other models for discrete-time survival analysis and multivariate time series forecasting.

Suppose you have a dataset that looks like this:

| ID   | period | feature_1 | feature_2 | feature_3 | ...  |
| ---- | ------ | --------- | --------- | --------- | ---- |
| 0    | 2016   | 7.2       | A         | 2AX       | ...  |
| 0    | 2017   | 6.4       | A         | 2AX       | ...  |
| 0    | 2018   | 6.6       | A         | 1FX       | ...  |
| 0    | 2019   | 7.1       | A         | 1FX       | ...  |
| 1    | 2016   | 5.3       | B         | 1RM       | ...  |
| 1    | 2017   | 5.4       | B         | 1RM       | ...  |
| 2    | 2017   | 6.7       | A         | 1FX       | ...  |
| 2    | 2018   | 6.9       | A         | 1RM       | ...  |
| 2    | 2019   | 6.9       | A         | 1FX       | ...  |
| 3    | 2017   | 4.3       | B         | 2AX       | ...  |
| 3    | 2018   | 4.1       | B         | 2AX       | ...  |
| 4    | 2019   | 7.4       | B         | 1RM       | ...  |
| ...  | ...    | ...       | ...       | ...       | ...  |

The entities with IDs 0, 2, and 4 are observed in the dataset in 2019.

* What are each of their probabilities of being observed in 2020? 2021? 2022?
* Given that they will be observed, what will be the value of feature_1? feature_3?
* Suppose entities can exit the dataset under a variety of circumstances. If entities 0, 2, or 4 exit in a given year, what will their circumstances be?
* How reliable can we expect these forecasts to be?
* How do the values of the features inform these forecasts?

FIFE can estimate answers to these questions for any unbalanced panel dataset.

FIFE unifies survival analysis (including competing risks) and multivariate time series analysis. Tools for the former neglect future states of survival; tools for the latter neglect the possibility of discontinuation. Traditional forecasting approaches for each, such as proportional hazards and vector autoregression (VAR), respectively, impose restrictive functional forms that limit forecasting performance. FIFE supports *the* state-of-the-art approaches for maximizing forecasting performance: gradient-boosted trees (using LightGBM) and neural networks (using Keras).

FIFE is simple to use:

```python
from fife.processors import PanelDataProcessor
from fife.lgb_modelers import LGBSurvivalModeler
import pandas as pd

data_processor = PanelDataProcessor(data=pd.read_csv(path_to_your_data))
data_processor.build_processed_data()

modeler = LGBSurvivalModeler(data=data_processor.data)
modeler.build_model()

forecasts = modeler.forecast()
```

Want to forecast future states, too? Just replace `LGBSurvivalModeler` with `LGBStateModeler` and specify the column you'd like to forecast with the `state_col` argument.

Want to forecast circumstances of exit ("competing risks" or its analogue for continuous outcomes)? Try `LGBExitModeler` with the `exit_col` argument instead.

See [Usage](#id1) and [this notebook using real data](https://nbviewer.jupyter.org/github/IDA-HumanCapital/fife/blob/master/examples/country_leadership.ipynb) for more examples.

FIFE is even simpler to use [from the command line](cli_link.html) (but currently only supports survival modeling - state and exit modeling coming soon!). Just point FIFE to your data and let FIFE run - no statistical or programming knowledge required. FIFE will automatically produce a host of spreadsheets and figures for you.

### Installation

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
		Example: `pip install fife-1.3.4-py3-none-any.whl`
* From GitHub (https://github.com/IDA-HumanCapital/fife):
	*	Clone the FIFE repository
	* Open Anaconda Prompt
	* Change the current directory in Anaconda Prompt to the directory of the cloned FIFE repository
	* Execute `python setup.py sdist bdist_wheel`
	* Execute `pip install dist/fife-1.3.4-py3-none-any.whl`

### Usage

The [Command-line Interface](cli_link.html) guide describes a pre-packaged FIFE pipeline. You can view the source code for that pipeline as follows:

```python
import fife.__main__
import inspect
print(inspect.getsource(fife.__main__))
```

You can use FIFE modules to customize your own pipeline, whether starting from scratch or by copying and modifying `fife.__main__`. Let's start from scratch for the following example.

We'll want a module to process our panel data and another module to use the processed data to train a model. For this example we'll build gradient-boosted tree models using [LightGBM](https://lightgbm.readthedocs.io/en/latest/).

```python
from fife.processors import PanelDataProcessor
from fife.lgb_modelers import LGBSurvivalModeler
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

The Panel Data Processor will make our dataset ready for modeling, to include sorting by individual and time, dropping degenerate and duplicated features, and computing survival durations and censorship status. We can specify processing parameters in the `config` argument. Perhaps we'd rather have an 80/20 validation split than the default 75/25. See the [Configuration Parameters](#id2) section for more details.

```python
data_processor = PanelDataProcessor(config={'VALIDATION_SHARE': 0.2},
                                    data=data)
data_processor.build_processed_data()
```

We can access the processed data as `data_processor.data` for input to a modeler. Let's use a gradient-boosted trees modeler. We can specify modeling parameters in the config argument. Perhaps we prefer a patience value of 8 rather than the default value of 4.

```python
survival_modeler = LGBSurvivalModeler(config={'PATIENCE': 8},
                                      data=data_processor.data)
survival_modeler.build_model()
```

In the case of gradient-boosted trees, our discrete-time survival "model" is actually a list of models, one for each time horizon. The first model produces a probability of survival through the first future period, the second model produces a probability of survival through the second future period conditional on survival through the first future period, and so on.

We can access the list of models as an attribute of our modeler. For example, we can see how many trees are in each model. Our `PATIENCE` parameter value of 8 configured each model to train until model performance on the validation set does not improve for eight consecutive trees (or no more splits can improve fit). That means the number of trees can vary across models.

```python
[i.num_trees() for i in survival_modeler.model]
```

We can also observe the relative importance of each feature, in this case in terms of the number of times the feature was used to split a tree branch.

```python
dict(zip(survival_modeler.model[0].feature_name(),
         survival_modeler.model[0].feature_importance()))
```

In general we are most interested in forecasts for individuals that are still in the population in the final period of the data. The `forecast` method gives us exactly those forecasts. The survival probabilities in the forecasts are cumulative - they are not conditional on survival through any prior future period. See the description of [Survival_Curves.csv](cli_link.html#survival-curves-csv) for more details.

```python
survival_modeler.forecast()
```

Because our forecasts are for the final period of data, we can't measure their performance in future periods. However, we can train a new model where we pretend an earlier period is the final period, then measure performance through the actual final period. In other words, we can reserve some of the most recent periods for testing.

```python
test_intervals = 4
data_processor = PanelDataProcessor(config={'VALIDATION_SHARE': 0.2,
                                            'TEST_INTERVALS': test_intervals},
                                    data=data)
data_processor.build_processed_data()
survival_modeler = LGBSurvivalModeler(config={'PATIENCE': 8},
                                      data=data_processor.data)
survival_modeler.build_model()
```

The `evaluate` method offers a suite of performance metrics specific to each time horizon as well as the concordance index over the restricted mean survival time. See the description of [Metrics.csv](cli_link.html#metrics-csv) above for more details. We can pass a Boolean mask to `evaluate` to obtain metrics only for the period we pretended was the most recent period. Be careful to not set `TEST_INTERVALS` to be too large; if it exceeds half of the number of intervals, it will not provide the training dataset with enough intervals.

```python
evaluation_subset = survival_modeler.data["_period"] == (
            survival_modeler.data["_period"].max() - test_intervals
        )
survival_modeler.evaluate(evaluation_subset)
```

The model we train depends on hyperparameters such as the maximum number of leaves per tree and the minimum number of observations per leaf. The `hyperoptimize` method searches for better hyperparameter values than the LightGBM defaults. We need only specify the number of hyperparameter sets to trial. `hyperoptimize` will return the set that performs best on the validation set for each time horizon.

```python
params = survival_modeler.hyperoptimize(16)
```

Now we can train and evaluate a new model with our curated hyperparameters.

```python
survival_modeler.build_model(params=params)
survival_modeler.evaluate(evaluation_subset)
```

`evaluate` offers just one of many ways to examine a model. For example, we can answer "In each time period, what share of the observations two periods past would we expect to still be around?" See the description of [Retention_Rates.csv](cli_link.html#retention-rates-csv) for more details.

```python
survival_modeler.tabulate_retention_rates(2)
```

Other modelers define different ways of using data to create forecasts and metrics, but they all support the methods `build_model`, `forecast`, `evaluate` and more.

Suppose we want to model not survival, but the future value of a feature conditional on survival. We can do with only two modifications: 1) replace our survival modeler with a state modeler, and 2) specify the feature to forecast. Let's pick `feature_4`.

```python
from fife.lgb_modelers import LGBStateModeler
state_modeler = LGBStateModeler(state_col="feature_4",
                                config={'PATIENCE': 8},
                                data=data_processor.data)
state_modeler.build_model()
state_modeler.forecast()
```

Because `feature_4` is categorical, our modeler trains a multiclass classification model for each time horizon and forecasts a probability of each category value in each future period for each individual in observed in the most recent period of data.  If we had chosen a numeric feature like `feature_1` or `feature_3`, our modeler would train a regression model for each time horizon and forecast the value of the feature in each future period.

Suppose we want to forecast the circumstances under which an exit would occur if it did occur. Typically, information on exit circumstances is an a separate dataset (otherwise, you might have an absurdly good predictor of survival!). In that case, you'd want to merge that dataset with your panel data to label each observation directly preceding an exit with the circumstances of that exit. For the purpose of this example, however, we'll treat `feature_3` as the circumstances of exit. FIFE will exclude the column representing circumstances of exit from the set of features.

We need only change from a state modeler to an exit modeler, and specify an `exit_col` instead of a `state_col`.

```python
from fife.lgb_modelers import LGBExitModeler
exit_modeler = LGBExitModeler(exit_col="feature_3",
                              config={'PATIENCE': 8},
                              data=data_processor.data)
exit_modeler.build_model()
exit_modeler.forecast()
```

Just like the state modeler, the exit modeler can handle categorical or numeric outcomes. Because `feature_3` is numeric, our modeler trains a regression model for each time horizon. For each future period, conditional on exit (i.e., ceasing to be observed) in that period, our modeler forecasts the value of `feature_3` in the period directly prior.

### Configuration Parameters

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
	How datetime features will be represented for the gradient-boosted trees modeler. If True, datetime features will be converted to integers in YYYYMMDD format for gradient-boosted trees modeler. Otherwise, datetime features will be converted to nanoseconds.
MAX_NULL_SHARE; default: 0.999; type: Decimal
	The maximum share of observations that may have a null value for a feature to be kept for training. Larger values may increase run time, risk of memory error, and/or model performance
MAX_UNIQUE_NUMERIC_CATS; default: 1024; type: Integer
	The maximum number of unique values for a feature of a numeric type to be considered categorical. Larger values may increase or decrease performance and/or increase run time.
NUMERIC_SUFFIXES; default: `[]` (empty list); type: List of strings
	Optional list of suffixes denoting that columns ending with such a suffix should be treated as numeric. Useful for flagging columns that have a numeric data type and fewer than MAX_NUM_CAT unique values. Column names with a categorical suffix and a numeric suffix will be identified as categorical.

##### Training set

MIN_SURVIVORS_IN_TRAIN; default: 64; type: Integer
	The minimum number of training set observations surviving a given time horizon for the model to be trained to make predictions for that time horizon.
TEST_INTERVALS; default: -1; type: Integer
	The number of most recent periods to treat as absent from the data during training for the purpose of model evaluation. Larger values may decrease model performance and run time and/or increase evaluation time frame.
TEST_PERIODS; default: 0; type: Integer
	One plus the value represented by TEST_INTERVALS. Deprecated and overriden by TEST_INTERVALS.
VALIDATION_SHARE; default: 0.25; type: Decimal
	The share of observations used for evaluation instead of training for hyperoptimization or early stopping. Larger values may increase or decrease model performance and/or run time.

##### Modeler types

TREE_MODELS; default: `true`; type: Boolean
	Whether FIFE will train gradient-boosted trees, as opposed to a neural network.

##### Hyperoptimization

HYPER_TRIALS; default: 0; type: Integer
	The number of hyperparameter sets to trial. If zero, validation early stopping will be used to decide the number of epochs. Larger values may increase run time and/or model performance.
MAX_EPOCHS; default: 256; type: Integer
	If HYPER_TRIALS is zero, the maximum number of passes through the training set. Larger values may increase run time and/or model performance.
PATIENCE; default: 4; type: Integer
	If HYPER_TRIALS is zero, the number of passes through the training dataset without improvement in validation set performance before training is stopped early. Larger values may increase run time and/or model performance.

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
	Whether FIFE will restrict the neural network to a proportional hazards model.

##### Metrics

BY_FEATURE; default: default: `""` (empty string); type: String
	The name of the feature for which FIFE will also produce separate Metrics_{value}.csv files for each group defined by each unique value of that feature.
QUANTILES; default: 5; type: Integer
	The number of similarly-sized bins for which survival and total counts will be reported for each time horizon. Larger values may increase run time and/or evaluation precision.
RETENTION_INTERVAL; default: 1; type: Integer
	The number of periods over which retention rates are computed.
SHAP_PLOT_ALPHA; default: 0.5; type: Decimal
	The transparency of points in SHAP plots. Larger values may increase visibility of non-overlapped points and/or decrease visibility of overlapped points.
SHAP_SAMPLE_SIZE; default: 128; type: Integer
	The number of observations randomly sampled for SHAP value calculation and plotting. Larger values may increase SHAP plot representativeness and/or run time.

### Acknowledgement

FIFE uses the `nnet_survival` module of Gensheimer, M.F., and Narasimhan, B., "A scalable discrete-time survival model for neural networks," *PeerJ* 7 (2019): e6257. The `nnet_survival` version packaged with FIFE is GitHub commit d5a8f26 on Nov 18, 2018 posted to https://github.com/MGensheimer/nnet-survival/blob/master/nnet_survival.py. `nnet_survival` is licensed under the MIT License. The FIFE development team modified lines 12 and 13 of nnet_survival for compatibility with TensorFlow 2.0 and added lines 110 through 114 to allow users to save a model with a PropHazards layer.

### Project Information

The Institute for Defense Analyses (IDA) developed FIFE on behalf of the U.S. Department of Defense, Office of the Under Secretary of Defense for Personnel and Readiness. Among other applications, FIFE is used to produce IDA's Retention Prediction Model.

FIFE has also been known as the Persistence Prediction Capability (PPC).

### Contact Information

- IDA development team: humancapital@ida.org
- IDA Principal Investigator: Dr. Julie Lockwood, jlockwood@ida.org, 703-578-2858

### Contributing

To contribute to FIFE please contact us at humancapital@ida.org and/or open a pull request at https://github.com/IDA-HumanCapital/fife.

### Testing

You can find our unit tests, powered by pytest, at https://github.com/IDA-HumanCapital/fife/tree/master/tests. For Windows users with conventional Anaconda or Miniconda installation paths, build_package.bat executes those unit tests and some functional tests in a fresh environment. Other users can execute the functional tests individually.

### Future Work

We intend to develop functionality for:

- Model stacking and/or blending
- GPU training
- Distributed computing with Apache Spark

We invite requests at humancapital@ida.org and issues at https://github.com/IDA-HumanCapital/fife/issues.

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
```bib
@misc{FIFE,
  author={Institute for Defense Analyses},
  title={{FIFE}: {Finite-Interval Forecasting Engine [software]}},
  howpublished={https://github.com/IDA-HumanCapital/fife},
  note={Version 1.x.x},
  year={2020}
}
```

This document was most recently updated 25 August 2020.
