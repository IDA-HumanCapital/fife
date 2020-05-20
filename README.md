The Finite-Interval Forecasting Engine (FIFE) provides machine learning and other models for discrete-time survival analysis and finite time series forecasting.

Suppose you have a dataset that looks like this:

| ID | period | feature_1 | feature_2 | feature_3 | ... |
|----|--------|-----------|-----------|-----------|-----|
| 0  | 2016   | 7.2       | A         | 2AX       | ... |
| 0  | 2017   | 6.4       | A         | 2AX       | ... |
| 0  | 2018   | 6.6       | A         | 1FX       | ... |
| 0  | 2019   | 7.1       | A         | 1FX       | ... |
| 1  | 2016   | 5.3       | B         | 1RM       | ... |
| 1  | 2017   | 5.4       | B         | 1RM       | ... |
| 2  | 2017   | 6.7       | A         | 1FX       | ... |
| 2  | 2018   | 6.9       | A         | 1RM       | ... |
| 2  | 2019   | 6.9       | A         | 1FX       | ... |
| 3  | 2017   | 4.3       | B         | 2AX       | ... |
| 3  | 2018   | 4.1       | B         | 2AX       | ... |
| 4  | 2019   | 7.4       | B         | 1RM       | ... |
| ...| ...    | ...       | ...       |...        | ... |

The entities with IDs 0, 2, and 4 are observed in the dataset in 2019.

* What are each of their probabilities of being observed in 2020? 2021? 2022?
* How reliable can we expect those probabilities to be?
* How do the values of the features guide our predictions?

FIFE answers these and other questions for any "unbalanced panel dataset" - a dataset where entities are observed periodically, but may depart the dataset after varying numbers of periods.

FIFE supports feedforward neural networks (using Keras) and gradient-boosted tree models (using LightGBM).

Read the documentation for FIFE at: https://fife.readthedocs.io/en/latest.
