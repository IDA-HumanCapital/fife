# Finite-Interval Forecasting Engine (FIFE)

The Finite-Interval Forecasting Engine (FIFE) Python package provides machine learning and other models for discrete-time survival analysis and multivariate series forecasting. Why should you use the FIFE?

* Advanced machine learning methods that improve forecasting retention over traditional survival analysis
* Functionality for forecasting survival to a type of exit 
* Functionality for forecasting the likelihood of being in a certain state in the future
* Command-line functionality for the binary retention model complete with graphs showing model metrics

### Installation

```console
pip install fife
```

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
  *	Open Anaconda Prompt
  *	Change the current directory in Anaconda Prompt to the directory of the cloned FIFE repository
  *	Execute `python setup.py sdist bdist_wheel`
  *	Execute `pip install dist/fife-1.3.4-py3-none-any.whl`

### Documentation

#### Python User Guide

* [Quick Start](quick_start.md)
* [Introduction to Survival Analysis using Machine Learning](introduction_survival_analysis.md)
* Survival Analysis with FIFE
* [Competing Risk and Future State Forecasts](competing_risks.md)
* [Configuration Parameters](config_parameters.md)
* [Frequently Asked Questions](faq.md) 
* Alternative Installation
* Testing
* Future Work
* License

#### Command Line Interface

* Quick Start
* Inputs
* Intermediate Files
* Outputs

#### Modules

* fife.base_modelers module
* fife.lgb_modelers module
* fife.nnet_survival module
* fife.pd_modelers module
* fife.processors module
* fife.tf_modelers module
* fife.utils module

### Acknowledgement 

FIFE uses the `nnet_survival` module of Gensheimer, M.F., and Narasimhan, B., "A scalable discrete-time survival model for neural networks," *PeerJ* 7 (2019): e6257. The `nnet_survival` version packaged with FIFE is GitHub commit d5a8f26 on Nov 18, 2018 posted to https://github.com/MGensheimer/nnet-survival/blob/master/nnet_survival.py. `nnet_survival` is licensed under the MIT License. The FIFE development team modified lines 12 and 13 of nnet_survival for compatibility with TensorFlow 2.0 and added lines 110 through 114 to allow users to save a model with a PropHazards layer.

### Project Information

The Institute for Defense Analyses (IDA) developed FIFE on behalf of the U.S. Department of Defense, Office of the Under Secretary of Defense for Personnel and Readiness. Among other applications, FIFE is used to produce IDA's Retention Prediction Model.

FIFE has also been known as the Persistence Prediction Capability (PPC).

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

### Contributing 

To contribute to FIFE please contact us at humancapital@ida.org and/or open a pull request at https://github.com/IDA-HumanCapital/fife.

### Contact Information

- IDA development team: humancapital@ida.org
- IDA Principal Investigator: Dr. Julie Lockwood, jlockwood@ida.org, 703-578-2858