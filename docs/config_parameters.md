# Configuration Parameters

FIFE is highly customizable for an individual project. These are the different parameters that can be overwritten in a `config.json` file.

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