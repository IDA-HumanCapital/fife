The Finite-Interval Forecasting Engine (FIFE) provides machine learning and other models for discrete-time survival analysis and multivariate time series forecasting.

Suppose you have a dataset that looks like this:

| ID   | period | feature_1 | feature_2 | feature_3 | ...  |
| ---- | ------ | --------- | --------- | --------- | ---- |
| 0    | 2019   | 7.2       | A         | 2AX       | ...  |
| 0    | 2020   | 6.4       | A         | 2AX       | ...  |
| 0    | 2021   | 6.6       | A         | 1FX       | ...  |
| 0    | 2022   | 7.1       | A         | 1FX       | ...  |
| 1    | 2019   | 5.3       | B         | 1RM       | ...  |
| 1    | 2020   | 5.4       | B         | 1RM       | ...  |
| 2    | 2020   | 6.7       | A         | 1FX       | ...  |
| 2    | 2021   | 6.9       | A         | 1RM       | ...  |
| 2    | 2022   | 6.9       | A         | 1FX       | ...  |
| 3    | 2020   | 4.3       | B         | 2AX       | ...  |
| 3    | 2021   | 4.1       | B         | 2AX       | ...  |
| 4    | 2022   | 7.4       | B         | 1RM       | ...  |
| ...  | ...    | ...       | ...       | ...       | ...  |

The entities with IDs 0, 2, and 4 are observed in the dataset in 2022.

* What are each of their probabilities of being observed in 2023? 2024? 2025?
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

Want to forecast circumstances of exit ("competing risks")? Try `LGBExitModeler` with the `exit_col` argument instead.

Here's a [guided example notebook](https://nbviewer.jupyter.org/github/IDA-HumanCapital/fife/blob/master/examples/country_leadership.ipynb) with real data, where we forecast when world leaders will lose power.

You can read the documentation for FIFE [here](https://fife.readthedocs.io/en/latest).
