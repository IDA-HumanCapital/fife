# Quick Start

The Finite-Interval Forecasting Engine (FIFE) conducts discrete-time survival analysis and multivariate time series forecasting on panel data. We will show you here how to build your own model pipeline in Python, but you can also use the pre-created functionality in [Command Line](command_line.md). You can also see a full example that uses the [Rulers, Elections, and Irregular Governance dataset (REIGN)](https://oefdatascience.github.io/REIGN.github.io/) in this [notebook](https://nbviewer.jupyter.org/github/IDA-HumanCapital/fife/blob/master/examples/country_leadership.ipynb).

### LGBSurvivalModeler

To get started, we'll want a module to process our panel data and another module to use the processed data to train a model. In this example, we'll build gradient-boosted tree models using [LightGMB](https://lightgbm.readthedocs.io/en/latest/).  To learn more about how this model is built, see [Introduction to Survival Analysis using Panel Data](intro_md).

.. code-block:: python

    from fife.processors import PanelDataProcessor
    from fife.lgb_modelers import LGBSurvivalModeler

We’ll need to supply a data file in the form of panel data. If you don’t have a file already available, you can use our example data:

.. code-block:: python

    from fife.utils import create_example_data
    data = create_example_data()

In order to make sure our results are reproducible, we can use the `make_results_reproducible` function: 

.. code-block:: python
    
    from fife.utils import make_results_reproducible
    make_results_reproducible()

The Panel Data Processor will make our dataset ready for modeling, to include sorting by individual and time, dropping degenerate and duplicated features, and computing survival durations and censorship status. We can specify processing parameters different than the defaults in the `config` argument. Perhaps we'd rather have an 80/20 validation split than the default 75/25. See the [Configuration Parameters](#id2) section for more details.

.. code-block:: python
    
    data_processor = PanelDataProcessor(config={'VALIDATION_SHARE': 0.2}, data=data)
    data_processor.build_processed_data()

Once we've called `data_processor.build_processed_data()`, the completed data will be available at `data_processor.data`. We can then use this in our models. For our first example, let's use a gradient-boosted trees modeler. Like the data building process, we can change modeling parameters in the config argument. Perhaps we prefer a patience value of 8 rather than the default value of 4.

.. code-block:: python

    survival_modeler = LGBSurvivalModeler(config={'PATIENCE': 8}, 
        data=data_processor.data)
    survival_modeler.build_model()

In the case of gradient-boosted trees, our discrete-time survival "model" is actually a list of models, one for each time horizon. The first model produces a probability of survival through the first future period, the second model produces a probability of survival through the second future period conditional on survival through the first future period, and so on.

We can access the list of models as an attribute of our modeler. For example, we can see how many trees are in each model. Our `PATIENCE` parameter value of 8 configured each model to train until model performance on the validation set does not improve for eight consecutive trees (or no more splits can improve fit). That means the number of trees can vary across models.

.. code-block:: python
    
    [i.num_trees() for i in survival_modeler.model]

We can also observe the relative importance of each feature, in this case in terms of the number of times the feature was used to split a tree branch.

.. code-block:: python

    dict(zip(survival_modeler.model[0].feature_name(),
            survival_modeler.model[0].feature_importance()))

In general, we are most interested in forecasts for individuals that are still in the population in the final period of the data. The `forecast` method gives us exactly those forecasts. The survival probabilities in the forecasts are cumulative - they are not conditional on survival through any prior future period. See the description of [Survival_Curves.csv](cli_link.html#survival-curves-csv) for more details.

.. code-block:: python
    
    survival_modeler.forecast()

Because our forecasts are for the final period of data, we can't measure their performance in future periods. However, we can train a new model where we pretend an earlier period is the final period, then measure performance through the actual final period. In other words, we can reserve some of the most recent periods for testing. Here is an example:

.. code-block:: python

    test_intervals = 4
    data_processor = PanelDataProcessor(config={'VALIDATION_SHARE': 0.2,
                                            'TEST_INTERVALS': test_intervals},
                                    data=data)
    data_processor.build_processed_data()
    survival_modeler = LGBSurvivalModeler(config={'PATIENCE': 8},
                                      data=data_processor.data)
    survival_modeler.build_model()

The `evaluate` method offers a suite of performance metrics specific to each time horizon as well as the concordance index over the restricted mean survival time. See the description of [Metrics.csv](cli_link.html#metrics-csv) above for more details. We can pass a Boolean mask to `evaluate` to obtain metrics only for the period we pretended was the most recent period. Be careful to not set `TEST_INTERVALS` to be too large; if it exceeds half of the number of intervals, it will not provide the training dataset with enough intervals.

.. code-block:: python
    
    evaluation_subset = survival_modeler.data["_period"] == (
            survival_modeler.data["_period"].max() - test_intervals
        )
    survival_modeler.evaluate(evaluation_subset)

The model we train depends on hyperparameters such as the maximum number of leaves per tree and the minimum number of observations per leaf. The `hyperoptimize` method searches for better hyperparameter values than the LightGBM defaults. We need only specify the number of hyperparameter sets to trial. `hyperoptimize` will return the set that performs best on the validation set for each time horizon.

.. code-block:: python

    params = survival_modeler.hyperoptimize(16)

Now we can train and evaluate a new model with our curated hyperparameters.

.. code-block:: python

    survival_modeler.build_model(params=params)
    survival_modeler.evaluate(evaluation_subset)

`evaluate` offers just one of many ways to examine a model. For example, we can answer "In each time period, what share of the observations two periods past would we expect to still be around?" See the description of [Retention_Rates.csv](cli_link.html#retention-rates-csv) for more details.

.. code-block: python

    survival_modeler.tabulate_retention_rates(2)

Other modelers define different ways of using data to create forecasts and metrics, but they all support the methods `build_model`, `forecast`, `evaluate` and more.

### LGBStateModeler

Suppose we want to model not survival, but the future value of a feature conditional on survival. We can do with only two modifications: 1) replace our survival modeler with a state modeler, and 2) specify the feature to forecast. Let's pick `feature_4`.

.. code-block:: python

    from fife.lgb_modelers import LGBStateModeler
    state_modeler = LGBStateModeler(state_col="feature_4",
                                config={'PATIENCE': 8},
                                data=data_processor.data)
    state_modeler.build_model()
    state_modeler.forecast()

Because `feature_4` is categorical, our modeler trains a multiclass classification model for each time horizon and forecasts a probability of each category value in each future period for each individual in observed in the most recent period of data.  If we had chosen a numeric feature like `feature_1` or `feature_3`, our modeler would train a regression model for each time horizon and forecast the value of the feature in each future period.

### LGBExitModeler

Suppose we want to forecast the circumstances under which an exit would occur if it did occur. Typically, information on exit circumstances is an a separate dataset (otherwise, you might have an absurdly good predictor of survival!). In that case, you'd want to merge that dataset with your panel data to label each observation directly preceding an exit with the circumstances of that exit. For the purpose of this example, however, we'll treat `feature_3` as the circumstances of exit. FIFE will exclude the column representing circumstances of exit from the set of features.

We need only change from a state modeler to an exit modeler, and specify an `exit_col` instead of a `state_col`.

.. code-block:: python
    
    from fife.lgb_modelers import LGBExitModeler
    exit_modeler = LGBExitModeler(exit_col="feature_3",
                              config={'PATIENCE': 8},
                              data=data_processor.data)
    exit_modeler.build_model()
    exit_modeler.forecast()

Just like the state modeler, the exit modeler can handle categorical or numeric outcomes. Because `feature_3` is numeric, our modeler trains a regression model for each time horizon. For each future period, conditional on exit (i.e., ceasing to be observed) in that period, our modeler forecasts the value of `feature_3` in the period directly prior.