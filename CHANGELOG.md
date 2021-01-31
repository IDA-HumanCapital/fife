## 1.5.0 - 2021-01-31

### Added

- [Hyperparameter search documentation](https://fife.readthedocs.io/en/latest/hyperparameter_search_link.html)

<u>Command-line Interface</u>

- Can now specify `TIME_ID_AS_FEATURE` as false to exclude the time identifier from the set of features

### Fixed

<u>StateModeler and ExitModeler</u>

- Observations with NaN outcome values now excluded from R-squared calculation

### Changed

- build_packages.bat and requirements.txt updated to Python 3.8

<u>StateModeler and ExitModeler</u>

- Prediction DataFrames for categorical outcomes now include future state in the index

## 1.4.2 - 2021-01-21

### Added

<u>StateModeler and ExitModeler</u>

- Outcome categories now accessible through class_values attribute

### Fixed

<u>ExitModeler</u>

- If the outcome is categorical, only labels associated with an exit (i.e., that appear in the last observation of a spell) are used for training

## 1.4.1 - 2020-12-30

### Added

<u>Modelers</u>

- Can now specify observation weights through the argument `weight_col`. The specified column will not be used as a feature, but will be used to weight observations during training and evaluation.

### Fixed

- Area under the receiver operating characteristic curve (AUROC) now computed for multiclass if no class is entirely positive. Classes with no positive values are excluded.
- ExitModeler outcome labeling
- Two hyperparameter prior distribution lower bounds now 2 ** -5 instead of 2e-5.
- LGBModelers now handle datetime categories

### Changed

- Multiclass AUROC now weighted by class share (`average="weighted"` in call to [sklearn.metrics.roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)).

## 1.4.0 - 2020-12-11

### Added

<u>Modelers</u>

- Can now specify `allow_gaps=True` to remove the restriction that individuals be observed in every future period over the given time horizon. For example, for a time horizon of 2, the default behavior of the StateModeler is to train and evaluate only on observations where the same individual was observed in the next 2 periods. `allow_gaps=True` will instead only require that the same individual be observed 2 periods into the future, thereby allowing a gap where the individual is not observed 1 period into the future.

<u>PanelDataProcessor</u>

- Now produces "_spell" column, which reports the number of gaps in observing the given individual up to the observed time period.

## 1.3.4 - 2020-12-09

### Added

<u>Command-line Interface</u>

- Can now use `BY_FEATURE` to produce separate Metrics.csv files for each value of a selected feature

### Fixed

- Number of classes now correctly specified for multiclass outcomes during hyperoptimization

## 1.3.3 - 2020-09-24

### Changed

- SHAP is now an optional dependency; install fife with `pip install fife[shap]` to ensure you can produce SHAP plots

### Removed

- Dask optional dependencies except `cloudpickle` and `toolz`

## 1.3.2 - 2020-09-22

### Removed

- Bokeh dependency

## 1.3.1 - 2020-09-08

### Added

- A guided [example notebook](https://github.com/IDA-HumanCapital/fife/blob/master/examples/country_leadership.ipynb); prettier view [here](https://nbviewer.jupyter.org/github/IDA-HumanCapital/fife/blob/master/examples/country_leadership.ipynb)

### Changed

- modeler.evaluate method now defaults to evaluating on the earliest period of test set observations instead of all observations

## 1.3.0 - 2020-08-25

### Added

- LGBStateModeler, which forecasts the value of a feature conditional on survival ("multivariate time series forecasting")

- LGBExitModeler, which forecasts the circumstances of exit conditional on exit ("competing risks")

### Deprecated

- GradientBoostedTreesModeler, now called "LGBSurvivalModeler"

- Standalone functions in the processors module, their responsibility having moved to the modeler method transform_features()

## 1.2.0 - 2020-08-17

### Added

<u>GradientBoostedTreesModeler</u>

- modeler.build_model() and modeler.train() now parallelize training over time horizons

<u>PanelDataProcessor</u>

- processor.build_processed_data() and processor.process_all_columns() now parallelize processing over columns

<u>Command-line Interface</u>

- Command-line execution now produces calibration and forecast error outputs

<u>Utils</u>

- Option within create_example_data() to specify number of persons and time periods in dataset

### Fixed

- Null category added to columns of pandas Categorical type in PanelDataProcessor
- Command-line execution now trains modeler for specific number of test intervals if specified

## 1.1.0 - 2020-07-20

### Added

<u>GradientBoostedTreesModeler and FeedforwardNeuralNetworkModeler</u>

- Support for hyperoptimization with modeler.hyperoptimize()
- Options within modeler.build_model() and modeler.train() for:
  - hyperparameters (such as those returned by hyperoptimization)
  - toggling off validation early stopping (using params argument in the case of build_model())
  - training on subset
- Defaults for all configuration parameters
- Default option to represent datetime features represented YYYYMMDD integers
- Option to represent datetime features as nanoseconds

<u>PanelDataProcessor</u>

- "\_period" and "\_maximum_lead" columns, which replace computation of "factorized time ids" in various methods
- Defaults for all configuration parameters
- Categorical feature conversion to pandas Categorical type

<u>Command-line Interface</u>

- Option to execute from command line without configuration file
- Option to specify individual parameter values
- Default configuration for processors and modelers
- Command-line execution now uses data file in current directory if there is only one file with a matching extension

### Removed

<u>PanelDataProcessor</u>

- Numeric feature normalization
- Homebrewed categorical feature integer mapping
- Raw subsetting

<u>Command-line Interface</u>

- Interacted fixed effects modeling
- Metrics-related output when no test set specified
- Forecast-related output when test set specified

### Fixed

- Validation and test sets no longer overlap
- modeler.evaluate() now reports correct metrics for subsets in which maximum observable period varies (e.g., train and test set combined)
- First period of test set now considered observed for computing training set outcomes
- ProportionalHazards models can now be saved to files
- Code now formatted using _Black_
- Command-line interface now evaluates on earliest period of test set instead of validation set

