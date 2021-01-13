# Frequently Asked Questions

### Should I specify all feature columns through the numeric or categorical `config` parameters? 

This mostly depends on what you wish to accomplish with your model. If, for example, you want to investigate the features that make the predictions using SHAP, you'll want to maximize your numeric features. In this case, you may want to make sure all columns are categorized to have the most numeric features. Otherwise, the model will run correctly with the default configuration.

### Should I treat a DateTime feature as numerical or categorical? 

Generally you should treat dates as numeric. The only time when you could use dates in the categorical sense and have the model be representative is in the unusual case where the categories of the observations for which you care to obtain predictions are all represented in the training set.

##### If I decide to treat DateTime as categorical, how can I ensure FIFE will run correctly? 

If you want to treat a date as a discrete, categorical variable, follow these steps: Datetime to string, string to categorical. This will allow LightGBM to properly serialize dates as categorical values.