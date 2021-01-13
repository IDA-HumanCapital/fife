# Introduction to Binary Survival Analysis using Machine Learning

FIFE unifies survival analysis (including competing risks) and multivariate time series analysis. Tools for the former neglect future states of survival; tools for the latter neglect the possibility of discontinuation. Traditional forecasting approaches for each, such as proportional hazards and vector autoregression (VAR), respectively, impose restrictive functional forms that limit forecasting performance. FIFE supports *the* state-of-the-art approaches for maximizing forecasting performance: gradient-boosted trees (using [LightGBM](https://lightgbm.readthedocs.io/en/latest/)) and neural networks (using [Keras](https://keras.io/)).

### Panel Data

FIFE uses data called **panel data**, which we define as periodic observations of subjects within a given time frame. Each observation is defined by a subject and a period. For example, each record of an active duty service member has a unique combination of social security number and year. Each observation has a vector of feature values. We may observe pay grade, occupation, and duty location. The period of observation is also a feature value. Not every service member is observed in every year, which makes the panel **unbalanced**. 

Unbalanced panels are not unique to personnel data, but are a natural product of any periodic record-keeping where the roster of subjects  changes over periods. An unbalanced panel could track equipment, medical patients, organizations, research and development programs, stockpiles, or natural phenomena. 

Perhaps your panel data looks like this:

.. table::

    ====    =======     ===========     =========   =========   ====
     ID     period      feature_1       feature_2   feature_3   ...
    ====    =======     ===========     =========   =========   ====
    0       2016        7.2             A           2AX         ...  
    0       2017        6.4             A           2AX         ...  
    0       2018        6.6             A           1FX         ...  
    0       2019        7.1             A           1FX         ...  
    1       2016        5.3             B           1RM         ...  
    1       2017        5.4             B           1RM         ...  
    2       2017        6.7             A           1FX         ...  
    2       2018        6.9             A           1RM         ...  
    2       2019        6.9             A           1FX         ...  
    3       2017        4.3             B           2AX         ...  
    3       2018        4.1             B           2AX         ...  
    4       2019        7.4             B           1RM         ...  
    ...     ...         ...             ...         ...         ...  
    ====    =======     ===========     =========   =========   ====

Using this data, we may want answers about which individuals will still be in the data in a future time period. We also may want to ask other questions:

* What are each of their probabilities of being observed in 2020? 2021? 2022?
* Given that they will be observed, what will be the value of feature_1? feature_3?
* Suppose entities can exit the dataset under a variety of circumstances. If entities 0, 2, or 4 exit in a given year, what will their circumstances be?

* How reliable can we expect these forecasts to be?
* How do the values of the features inform these forecasts?

FIFE can estimate answers to these questions for any unbalanced panel dataset.

### Survival Analysis

Survival analysis was first used to determine if someone lived or died. Today, however, we can use its theories to define exits from a binary state as either a _life_ or _death_. For instance, we can investigate if an individual leaves the military service by asking if this individual "dies." If another individual joins the service, we can say they were "born."

We can use these ideas to define retention as "living" and attrition as "dying." To investigate retention, FIFE seeks forecasts for observations in the final period of the dataset. For each observation, we define “retention” as the observation of the  same subject in a given number of consecutive future periods. For example, 2-year retention of a subject observed in 2019 means observing the same subject in 2020 and 2021. Note that we define  retention with reference to the time of observation. We ask not “will this individual be in the dataset in 2 years,” but “will this member be in this dataset 2 more years from this date?”

#### Censoring

One of the key concepts that survival analysis attempts to address is that of **censoring**. If an individual is present in the last date for which we have data, we don't know if that individual left the data, or if they did, at what point. This individual is *right censored* (left censoring can also happen if the individual entered before our data began, but this does not impact the FIFE in the same way as right censoring). If we were just to remove all censored individuals, we will create bias because we would only be using the people we've seen leave, therefore not allowing for a person to leave later than our final date. We must, then, assume that the censored individual could still attrite later and mark them as remaining.

#### Survival Function

The survival function is the true, unknown, curve that describes the amount of the population remaining at time :math:`t`. This function, :math:`S(t)`, is defined as:

.. math::

    S(t)=Pr(T>t)

Thus, this function notes the probability that an individual does not leave at time :math:`t`, or the probability of staying past time :math:`t`. This is a decreasing function. 

#### Hazard Function

Let :math:`y(t)` be 1 if a given individual exits an amount of time :math:`t` into the future, and 0 otherwise. Let :math:`h(t)` be the probability that an individual exits at time :math:`t`, given the individual has not exited before. :math:`h(t)` is a **hazard function**, which takes :math:`t` as input and outputs a probability that :math:`y(t)` equals 1. It can also be shown that: 

.. math:: 

    h(t) = \frac{-S'(t)}{S(t)}

Data that allows us to observe :math:`y` for different values of :math:`t` inform us about which hazard functions will better match true outcomes (thus, have better _performance_). We _fit_ a model by using data to estimate the function that matches the true data the best.

#### Traditional Methods for Fitting a Survival Curve

The hardest part about fitting a hazard function is accounting for censoring. A particularly famous and simple method of fitting a forecasting model  under censoring is the [**Kaplan-Meier estimator**](https://www.tandfonline.com/doi/abs/10.1080/01621459.1958.10501452). For each time horizon at  which at least one observation exited, the Kaplan-Meier estimator  computes the share of observations that exited among those observed to  exit or survive at the given time horizon. The Kaplan-Meier estimator is defined as: 

.. math::

    \hat{S}(t) = \prod_{t_{i}<t}\frac{n_i-d_i}{n_i}

where :math:`d_i` are the number of attritions at time :math:`t` and :math:`n_i` is the number of individuals who are likely to attrition just before :math:`t`. This survival curve estimator describes share retained ("alive") as a function of the time horizon. The Kaplan-Meier estimator is the most famous estimation to the true survival curve in traditional survival analysis.

##### Estimation using Feature Values

The problem is the Kaplan-Meier estimation to the survival curve, and the resulting hazard function, do not account for the vector of feature values, :math:`x`. Without accounting for :math:`x`, the hazard function is the same for all observations, so we cannot  estimate which subjects will retain longer, and we cannot target retention interventions. There are other ways to estimate hazards and survival curves using feature values. Parametric estimators account for :math:`x` by specifying a functional form for :math:`h`. For example, **proportional hazards regression** specifies that :math:`h` is the product of a baseline hazard :math:`h_{0}(t)` and :math:`e` raised to the inner product of :math:`x` and some constant vector :math:`\beta`. The principal drawback of specifying a functional form is that it limits how well a model can fit to data.

Another method to account for :math:`x` is to apply the Kaplan-Meier estimator separately for each unique value of :math:`x`. We can describe this method as :math:`h(t,x)=h_{x}(t)`. This method is only as useful as the number of observations with each unique value. For example, because there are thousands of active duty service members in each service in each year, this method would be  useful to estimate a separate hazard function for each service. We could also estimate a separate hazard function for each combination of service and entry year. However, we would be limiting :math:`x` to contain the values of only two features. We could not use this method to estimate a separate hazard function for each combination of  feature values among the hundreds of features in our data. Even a single feature with continuous support, such as the amount a member commits to their Thrift Savings Plan (TSP), makes this method infeasible.

### FIFE: Using Machine Learning

With FIFE, we can use machine learning to consider the interactions of feature values in predicting whether an individual will stay or leave. Traditional methods in the domain of survival analysis effectively consider all forecastable future time horizons by expressing the probability of retention as a continuous function of time since observation. In the case of panel data,  however, all forecastable time horizons fall in a discrete domain with a finite number of intervals. We can, therefore, consider each forecastable time horizon separately rather than treating time as continuous, unlocking all manner of forecasting methods not designed for survival analysis, including state-of-the-art machine learning methods.

Our method to account for :math:`x` is to estimate the hazard separately for each unique value of :math:`t`. We can describe this method as :math:`h(t,x)=h_{t}(x)`. This method is only as useful as the number of observations that exit at each time horizon. If the time horizon has continuous support, so that there are an infinite number of forecastable time horizons, this method is infeasible. However, panel data exhibit a finite number of forecastable time horizons. A panel data time horizon must be a whole  number of periods less than the number of periods in the dataset.  Therefore we can fit :math:`h_{t}(x)` separately for each such :math:`t`.

For each :math:`t`, we fit :math:`h_{t}(x)` to observations of the feature vector :math:`x` and the binary outcome :math:`y`. We can use any binary classification algorithm to fit :math:`h_{t}(x)`. We need not specify a functional form. Using FIFE, we can fit not only the hazard, but any function of :math:`x` for each time horizon :math:`t`. For example, we can forecast the future value of some feature  immediately before a given time horizon, conditional on exit at that  time horizon. Alternatively, we can forecast a future feature value conditional on survival. 

Like the Kaplan-Meier estimator, our method addresses censoring by  considering only observations for which we observe exit or survival at the given time horizon. In other words, we address censoring by subsetting the data for each :math:`t`. The panel nature of the data make this subsetting simple: if there are :math:`T` periods in the data, only observations from the earliest :math:`T−t` periods allow us to observe exit or survival :math:`t` periods ahead. For example, suppose we have annual observations from 2000 through 2019 available for model fitting. Let our time horizon be four years. We exclude the most recent four years, leaving observations from 2000 through 2015. Among those observations, we keep only those retained at least three years from the time of observation. Then we compute the binary outcome of survival or exit four years from the time of observation. This estimation actually looks at one minus the hazard, which leaves survival as a positive class. 

##### Gradient-Boosted Trees

Gradient-boosted trees are a state-of-the-art machine learning algorithm for binary classification and regression. A gradient-boosted trees model is a sequence of **decision trees**. Each decision tree repeatedly sorts observations based on a binary condition. For example, a decision tree for predicting one-year Army officer retention could first sort observations into those with fewer than six  years of service and those with six or more. Then it could sort the former subgroup into those who accessed through the Unites States Military Academy and those who accessed otherwise. The tree could sort the subgroup with six or more years of service into those who are in the Infantry Branch or Field Artillery Branch and those who are not. The tree could continue to sort each subgroup until reaching some stopping criterion. Any given observation sorts into one of the resulting subgroups (**leaves**), and is assigned the predicted value associated with that leaf.

.. image:: images/ex_dec_tree.png
    :width: 600px
    :align: center

*An example decision tree for predicting attrition of a first year Army Officer.*

In the model training process, we use training data to decide what  conditions to sort by and what values to predict for each leaf. We provide a cursory description of the model training process here to familiarize the reader and explain our modeling choices. Our description omits many caveats, intricacies, and techniques developed by the  machine learning community over years of research. We refer the advanced reader to the documentation for [LightGBM](https://lightgbm.readthedocs.io/en/latest/) for details on the particular software we use to train gradient-boosted tree models.

We begin model training with the entire training data. We sort the data into two subgroups using the feature and feature value that most informatively divides the training data into those who were retained and those who were not. We apply the same information criterion to split each of the resulting subgroups. We stop splitting after reaching any of our stopping criteria, such as a maximum number of splits.

Each leaf of a trained tree contains one or more training observations; the predicted value associated with a leaf is the mean outcome among those training observations. In our example, the mean outcome is the share of Army officers retained. Look at the leaf of the tree which represents Officers with less than six years of service not from the Military Academy and from a state that is not Virginia.[^1] If 90% of such Officers in the training data were retained, the tree will assign a retention probability of 90% to any Officer that falls into the same leaf. If we were just looking for binary classification, this would be assigned to "remain."

While we could stop training after one tree, gradient boosting allows us to improve model performance by training more trees to correct model errors. A prediction from a gradient-boosted tree model is a weighted sum of predicted values from the trained trees. Since our binary classification call,  **LGBSurvivalModeler**, uses boosted tree models, the cumulative product of the predictions form an estimated survival function. The survival probabilities at time horizon :math:`t` periods into the future are defined as: 

.. math::

    `Pr(T_{i\tau}\ge t|X_{i\tau})`, 

where :math:`X_{i\tau}` is a vector of feature values for individual :math:`i` at time :math:`\tau`, and :math:`T_{i\tau}` is the number of consecutive future periods the individual remains after time :math:`\tau`. 

##### Neural Networks

In using FIFE for forecasting defense personnel retention, we have found that gradient-boosted trees outperform neural networks for a variety of populations. However, we've limited our training data to subpopulations and to feature values at the time of observation. Neural networks offer unique ways to take advantage of data that surpasses these limits, so we have implemented this as an option in FIFE as well.

Specially structured data, such as text from performance reviews, audio or video samples of job performance, or social graphs, could also be predictive of retention, for which neural networks would make more sense. We have also not used past feature values for a given individual, a task uniquely suited to neural network models that exploit sequential structure, such as the  transformer. Training on the entire panel of data available for a given  individual at the time of observation could vault neural network performance over that of gradient-boosted trees. In our models, we use [Keras](https://keras.io/).

[^1]: In practice, we can expect many more conditions for a given leaf. We list only a few example conditions to simplify our explanation. 