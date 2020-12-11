FIFE's command-line interface allows you to forecast survival without writing any code. Forecasting circumstances of exit and future states of survival from the command line is coming soon!

### Quick Start

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

Configuration parameters may be provided in the JSON file format and/or individually with the parameter name prepended by `--`. Individually provided parameter values will override values in a JSON file. For example, `fife config.json --TEST_INTERVALS 4 --HYPER_TRIALS 16 ` will assign a value of 4 to TEST_INTERVALS and a value of 16 to HYPER_TRIALS, and otherwise use the parameter values provided in the config.json file in the current directory.

Input data may be provided in any of the following file formats with the corresponding file extension:

| File Format                  | Extension  |
| ---------------------------- | ---------- |
| Feather                      | .feather   |
| Comma-separated values (CSV) | .csv       |
| CSV with gzip compression    | .csv.gz    |
| Pickled Pandas DataFrame     | .p or .pkl |
| HDF5                         | .h5        |
| JSON                         | .json      |

If you execute FIFE from a directory with a single file with any of these extensions except .json, FIFE will use the data from that file. If you execute FIFE from a directory with multiple such files, or if you wish to use a data file in a different directory than the directory where you want to store your results, you will need use the `DATA_FILE_PATH` configuration parameter to specify the path to your data file.

Input data may not contain any of the following feature names:

-	`_duration`
-	`_event_observed`
-	`_maximum_lead`
-	`_period`
-	`_predict_obs`
-   `_spell`
-	`_test`
-	`_validation`

### Intermediate Files

FIFE produces as intermediate files:

-	*Intermediate/Data/Categorical_Maps.p*: Crosswalks from whole numbers to original values for each categorical feature
-	*Intermediate/Data/Numeric_Ranges.p*: Maximum and minimum values in the training set for each numeric feature
-	*Intermediate/Data/Processed_Data.p*: The provided data after removing uninformative features, sorting, assigning train and test sets, normalizing numeric features, and factorizing categorical features
-	One of the following:
  -	*Intermediate/Models/[horizon]-lead_GBT_Model.json*: A gradient-boosted tree model for each time horizon
  -	*Intermediate/Models/FFNN_Model.h5*: A feedforward neural network model for all time horizons

Intermediate files are not intended to be read by humans.

### Outputs

If `TEST_INTERVALS` is zero, or if `TEST_PERIODS` is one or less and `TEST_INTERVALS` is unspecified, FIFE produces:

-	*Output/Tables/Survival_Curves.csv*: Predicted survival curves for each individual observed in the final period of the dataset
-	*Output/Tables/Aggregate_Survival_Bounds.csv*: Expected number of individuals retained for each time horizon with corresponding uncertainty bounds
-	In *Output/Figures/*, the following plots of relationships between features and the probabilities of exit for each time horizon conditional on survival to the final period of that time horizon:
  -   *Dependence_[horizon]_lead.png* and *Dependence_RMST.png*
  -   *Importance_[horizon]_lead.png* and *Importance_RMST.png*
  -   *Summary_[horizon]_lead.png* and *Summary_RMST.png*
-	*Output/Tables/Retention_Rates.csv* and *Output/Figures/Retention_Rates.png*: Actual and predicted share of individuals who survived a given number of periods for each period in the dataset

Otherwise, FIFE produces:

-	*Output/Tables/Metrics.csv*: Model performance metrics on the test set
-	*Output/Tables/Counts_by_Quantile.csv*: Number of individuals retained for each time horizon based on different quantiles of the predicted survival probability
-	*Output/Tables/Forecast_Errors.csv*: The difference between predicted probability and actual binary outcome by time horizon for each individual in the test set
-	*Output/Tables/Calibration_Errors.csv*: Actual and predicted shares survived by octile of predicted survival probability, aggregated over all time horizons and individuals in the test set

In either case, FIFE produces *Output/Logs/Log.txt*, a log of script execution.

All files produced by FIFE will overwrite files of the same name in the directory designated by the `RESULTS_PATH` configuration parameter.

If 'BY_FEATURE' is specified, then, in addition to Metrics.csv, a separate Metrics_{value}.csv file will be produced for each group defined by each unique value of that feature.

##### Survival_Curves.csv

- First column: The identifier of each individual observed in the final time period of the dataset.
- Second column: Each individual's probability of surviving one more period (i.e., appearing in the data once more if the data were to be extended one period beyond the final observed period).
- Third column: Each individual's probability of surviving two more periods.
- N-th column: Each individual's probability of surviving N more periods.

##### Aggregate_Survival_Bounds.csv

- First column: Time horizons over which the survival probabilities of all individuals observed in the final period of the dataset will be aggregated.
- Second column: Expected number of individuals retained for each time horizon.
- Third and fourth columns: Lower and upper two-sided [Chernoff bounds](https://en.wikipedia.org/wiki/Chernoff_bound) to quantify the aggregation uncertainty around the expected number of retained individuals.

##### Retention_Rates.csv

- First column: The number of time periods since the earliest period in the data.
- Second column: The share of individuals observed a given number of periods prior who survived to the corresponding period. The number of periods prior is given by the `RETENTION_INTERVAL` configuration parameter.
- Third column: The model prediction of the corresponding actual retention rate in the second column for periods in the data on which the model was trained.
- Fourth column: The predicted retention rate for periods in the data on which the model was not trained. These periods include periods in the data excluded from training by the `TEST_PERIODS` or `TEST_INTERVALS` configuration parameter and periods beyond those in the data.

##### Figures

*Retention_Rates.png* plots the contents of *Retention_Rates.csv*.

For gradient-boosted tree modelers there will be other plots in *Output/Figures/* which depict aggregations and subsets of Shapley Additive Explanations (SHAP) values for individuals in the final period of the dataset. A [SHAP value](https://github.com/slundberg/shap/blob/master/README.md) is a measure of how much a specific feature value contributes (positively or negatively) to a specific predicted value. For files with names ending in `_lead`, the predicted value is the probability of surviving the number of additional periods specified in the file name. For files with names ending in `RMST`, the predicted value is the RMST.

Plots with names beginning with `Importance_`: Mean absolute SHAP values for up to 20 features with the highest such means. The greater its mean absolute SHAP value, the more a feature contributes to the survival probability (positively or negatively) in the given period (or RMST), on average. Thus the features at the top of the plot are the most important features.

Plots with names beginning with `Summary_`: SHAP values associated with each of the most important features against the values of those features. A pattern in such a plot indicates an association between the plotted feature and the predicted outcome. For example, points shifting from blue to red from left to right for a given feature indicates that the feature is positively correlated with the predicted outcome.

Plots with names beginning with `Dependence_`: SHAP values associated with the most important feature against the values of that feature and the values of an automatically selected other feature. Different patterns for differently-colored points indicate that the association between the most important feature and the predicted outcome depends on the value of the other feature. For example, if blue points form an upward-sloping line but red points form a downward-sloping line then a low value of the other feature is associated with a negative correlation between the most important feature and the predicted outcome, but a high value of the other feature is associated with a positive correlation.

##### Metrics.csv

- First column: Durations of forecast time horizons, given in terms of the number of periods observed in the data. Each of the metrics in the subsequent columns are calculated over the observations from the earliest period in the test set.

- Second column: Area Under the Receiver Operating Characteristic (AUROC). The AUROC is, over all pairs of observations where exactly one of the observations survived in the given horizon, the share of those pairs for which the model predicts the survivor was more likely to survive. A totally uninformed model would be expected to produce an AUROC of 0.5 and a perfect model would produce an AUROC of 1.0. A blank value indicates that either all or no observations survived the given horizon, so an AUROC could not be calculated.

- Third column: Mean of each individual's predicted probability of surviving through the given horizon, which gives the predicted share of individuals that will survive the horizon.

- Fourth column: Actual share of those individuals that survived the horizon. All else equal, a better model will produce smaller absolute differences between predicted shares and actual shares.

- Fifth column: Number of observations with a predicted survival probability greater than or equal to 0.5 that survived.

- Sixth column: Number of observations with a predicted survival probability greater than or equal to 0.5 that did not survive.

- Seventh column: Number of observations with a predicted survival probability less than 0.5 that survived.

- Eighth column: Number of observations with a predicted survival probability less than 0.5 that did not survive. All else equal, a better model will have larger values in the fifth and eighth columns and smaller values in the sixth and seventh columns.

- Ninth column: Concordance index (or "C-index"), a single value to evaluate the model over all time horizons in the test set. The concordance index is, over all pairs of observations where one observation is known to have survived longer, the share of those pairs for which the model predicts a greater restricted mean survival time (RMST) for the observation that survived longer. An individual's RMST is the number of periods the model expects the individual to survive over the greatest horizon for which the model produced a prediction. Like AUROC, a totally uninformed model would be expected to produce a concordance index of 0.5 and a perfect model would produce a concordance index of 1.0.

##### Counts_by_Quantile.csv

- First column: Time horizon.
- Second column: The quantile based on predicted probability of survival. For example, if the number of quantiles is given to be 5, quantile 1 represents the 20% (perhaps roughly) of observations predicted least likely to survive the given time horizon among those for which the given horizon could be observed. The number of quantiles is given by the `QUANTILES` configuration parameter.
- Third column: The number of observations for which the individuals in the given quantile survived the given horizon.
- Fourth column: The number of observations in the given quantile. Ties in predicted probability among observations may cause the number of observations to vary across quantile for a given time horizon.

##### Forecast_Errors.csv

- N-th column: Each test set individual's probability of surviving N more periods, minus one if they survived. Positive zero error indicates a probability of zero; negative zero error indicates a probability of one. The set of errors in this file is sufficient to compute any model performance metrics on the test set.

##### Calibration_Errors.csv

- First column: The octile of predicted probabilities in the test set. For example, quantile 1 represents the lowest 12.5% of predicted probabilities among all test set individuals and time horizons.
- Second column: The mean of the predictions in the given octile.
- Third column: The share of predictions in the given octile that corresponded to an outcome of survival.
- Fourth column: The difference between the second and third columns. Values closer to zero indicate better model performance.

### Configuration Parameters

You may customize many parameters, such as the maximum share of a feature that may be missing and the share of individuals to reserve for model evaluation. See the [Configuration Parameters](user_guide_link.html#configuration-parameters) section of the User Guide. You can also view a list of parameters and descriptions by executing `fife --h` or `fife --help` in an Anaconda prompt. You can find an example configuration file, readable with any text editor, in the [FIFE GitHub repository](https://github.com/IDA-HumanCapital/fife).

