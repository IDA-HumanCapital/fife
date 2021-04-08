Hyperparameters are parameters that govern model training. For example, before we train a neural network, we need to define the number and width of hidden layers. Before we train a tree-based model, we need to define a set of parameters that govern whether or not a node will split, including the minimum number of observations in a node and the maximum tree depth.

FIFE and other packages typically offer default hyperparameter values. These defaults represent reasonable context-agnostic guesses at the hyperparameter values that will maximize model performance. However, we cannot expect default values to maximize the performance of any particular model. One way to improve model performance is to search for hyperparameter values that yield better performance than the defaults.

#### Hyperparameter search in one line

FIFE modelers make hyperparameter search as easy as possible through the `hyperoptimize` method. Just call `modeler.hyperoptimize`, and FIFE will evaluate 64 different sets of hyperparameter values in addition to the default values, and return the set that performed best on held-out data. For LightGBM modelers, FIFE will search for hyperparameter values separately for each time horizon. We cannot do the same for neural network modelers, which train a single model for all time horizons. Ensure you have already assigned `modeler.n_intervals` before you call `modeler.hyperoptimize`; you can do so manually (e.g., `modeler.n_intervals = 8`) or with the same method that `build_model` uses by default: `modeler.n_intervals = modeler.set_n_intervals()`.

If you prefer to evaluate fewer or more sets than 64, specify the number with the `n_trials` argument. Beware that computational constraints are not the only reason to limit `n_trials`; by evaluating more sets of hyperparameter values you risk overfitting to the held-out data. Like the hyperparameters themselves, the optimal number of trials is context-specific and generally unknowable. It's acceptable and not unusual for the default hyperparameter values to perform best.

`modeler.hyperoptimize` returns a dictionary where each key-value pair maps a time horizon to a set of hyperparameters. You can pass that dictionary directly to `modeler.build_model` through the `params` argument. When you don't specify `params`, `modeler.build_model` performs validation-based early stopping to decide the number of trees (for tree-based models) or epochs (for neural networks) to train. However, the number of trees/epochs is a hyperparameter. If you pass a result of `modeler.hyperoptimize()` as `params`, thereby specifying the number of trees/epochs for each time horizon, the modeler will not perform early stopping.

#### Validation

We seek to maximize model performance in future periods, so we should measure model performance on data that follow the training data. Hyperparameter search presents no exception to this principle. For a given time horizon :math:`t`, we must reserve at least :math:`t` of the most recent periods of data so we can measure model performance. If we reserve exactly :math:`t` of the most recent periods, we can measure model performance on the observations from one period - the last period of the training data. If we reserve :math:`t+1` of the most recent periods, we can measure model performance on observations from the last period of the training data and the period directly after; however, we have one fewer period of observations for training. For LGB modelers, FIFE defaults to reserving :math:`t` or 25% of the most recent periods, whichever is larger, for measuring model performance. This "rolling validation" method causes the validation data to vary across time horizons. For neural network modelers, FIFE uses the validation set defined by `modeler.validation_col`; if you prefer to use this method for an LGB modeler, specify `rolling_validation=False` in your call to `hyperoptimize`.

You can specify a value other than 25% through the `VALIDATION_SHARE` configuration parameter when you instantiate your modeler. If there is only one period of observations for a given time horizon, FIFE will hold out a share of those observations at random according to the `VALIDATION_SHARE`.

Note that test data (labeled by `modeler.test_col`) will be ignored for hyperparameter search. For example, if you have 20 periods of data and the test set comprises the most recent four periods, you will have 16 periods for hyperparameter search. For a time horizon of one period, FIFE will reserve 25% of the most recent of those 16 periods for measuring performance and train on the remaining 12 periods.

#### Under the hood

FIFE uses [Optuna](https://optuna.org/) to perform hyperparameter search. For each hyperparameter FIFE specifies a prior distribution. Optuna then draws from each distribution, trains a model with the drawn hyperparameters, evaluates the model on the validation data, updates each distribution in a Bayesian fashion, then repeats the aforementioned steps for a specified number of iterations. FIFE uses Optuna's [MedianPruner](https://optuna.readthedocs.io/en/latest/reference/generated/optuna.pruners.MedianPruner.html) 
to stop the training of poorly performing models early.

We've implemented `hyperoptimize` for LightGBM and neural network modelers. The following tables list the prior distributions we specify for those two modeler types. All prior distributions are uniform. The hyperparameters in the tables are only those we include in our search. There are other hyperparameters for which we accept the default values. You can find LightGBM hyperparameter descriptions and default values in the [LightGBM documentation](https://lightgbm.readthedocs.io/en/latest/Parameters.html). You can find neural network hyperparameter descriptions and default values in the [Configuration Parameters](user_guide_link.html#configuration-parameters) section of the FIFE user guide.

| Type  | LightGBM Hyperparameter | Min     | Max  |
| ----- | ----------------------- | ------- | ---- |
| Int   |                         |         |      |
|       | num_iterations          | 8       | 128  |
|       | num_leaves              | 8       | 256  |
|       | max_depth               | 4       | 32   |
|       | min_data_in_leaf        | 4       | 512  |
|       | bagging_freq            | 0       | 1    |
|       | min_data_per_group      | 1       | 512  |
|       | max_cat_threshold       | 1       | 512  |
|       | max_cat_to_onehot       | 1       | 64   |
|       | max_bin                 | 32      | 1024 |
|       | min_data_in_bin         | 1       | 64   |
| Float |                         |         |      |
|       | learning_rate           | 0.03125 | 0.5  |
|       | min_sum_hessian_in_leaf | 0.03125 | 0.25 |
|       | bagging_fraction        | 0.5     | 1    |
|       | feature_fraction        | 0.5     | 1    |
|       | lambda_l1               | 0       | 64   |
|       | lambda_l2               | 0       | 64   |
|       | min_gain_to_split       | 0       | 0.25 |
|       | cat_l2                  | 0       | 64   |
|       | cat_smooth              | 0       | 2048 |


| Type  | Neural Network Hyperparameter                                | Min        | Max  |
| ----- | ------------------------------------------------------------ | ---------- | ---- |
| Int   |                                                              |            |      |
|       | Dense hidden layers                                          | 0          | 3    |
|       | Nodes per dense hidden layer ("width")                       | 16         | 1024 |
|       | Epochs before freezing embedding layers                      | 4          | 128* |
|       | Epochs after freezing embedding layers                       | 4          | 128* |
|       | Batch size**                                                 | min(32, N) | N    |
| Float |                                                              |            |      |
|       | Dropout share                                                | 0          | 0.5  |
|       | Ratio of log of embedding length to log of feature cardinality ("EMBED_EXPONENT") | 0          | 0.25 |
|       | L2 regularization on embedding layers                        | 0          | 16   |

*You can specify a different value through the max_epochs argument to `modeler.hyperoptimize`.

**N is the number of training observations for the given time horizon.

Proportional hazards and interacted fixed effects modelers are so simple that they have no hyperparameters.

#### Hyperparameters all the way down

The hyperparameter search process itself requires the specification of many parameters. What hyperparameters will we include in the search? What prior distributions will we specify? How many hyperparameter sets will we evaluate? What observations will we hold out for validation? Just as hyperparameters are parameters of the training process, these parameters of the hyperparameter search process are "hyperhyperparameters." The `hyperoptimize` method provides reasonable guesses at the hyperhyperparameters that will maximize model performance, but you may believe that other hyperhyperparameters may yield better performance for your particular model. You are welcome to wrap `hyperoptimize` in your own Optuna routine, or to reach under the modeler hood to get more control over the hyperhyperparameters. Be forewarned, though: [that way madness lies](http://shakespeare.mit.edu/lear/full.html). The hyperhyperparameters you specify must themselves have some parameters that govern them, and those parameters must have their own parameters, and so on. It's hyperparameters [all the way down](https://en.wikipedia.org/wiki/Turtles_all_the_way_down). You must stop somewhere; we suggest you stop one step deep.

