## 1.1.0 - 2020-06-12

### Added

<u>Execution</u>

- Option to execute from command line without configuration file
- Default configuration for processors and modelers
- Command-line execution now uses data file in current directory if there is only one file with a matching extension

<u>GradientBoostedTreesModeler</u>

- Support for hyperoptimization with modeler.hyperoptimize()
- Options within modeler.train() for:
  - hyperparameters (such as those returned by hyperoptimization)
  - toggling off validation early stopping (useful if num_iterations already hyperoptimized)
  - training on subset
- Defaults for all configuration parameters

<u>PanelDataProcessor</u>

- "\_period" and "\_maximum_lead" columns, which replace computation of "factorized time ids" in various methods
- Defaults for all configuration parameters

### Fixed

- Validation and test sets no longer overlap
- modeler.evaluate() now reports correct metrics for subsets in which maximum observable period varies (e.g., train and test set combined)
- SHAP sample size now capped at number of predict set observations
- Code now formatted using _Black_

