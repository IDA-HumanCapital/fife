## 1.1.0 - 2020-07-20

### Added

<u>Execution</u>

- Option to execute from command line without configuration file
- Option to specify individual parameter values
- Default configuration for processors and modelers
- Command-line execution now uses data file in current directory if there is only one file with a matching extension

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
- ProportionalHazards models are can now be saved to files
- Code now formatted using _Black_
- Command-line interface now evaluates on earliest period of test set instead of validation set

