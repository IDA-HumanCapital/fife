"""FIFE modeler based on LightGBM, which trains gradient-boosted trees."""

from typing import List, Union

from fife import survival_modeler
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap


class GradientBoostedTreesModeler(survival_modeler.SurvivalModeler):
    """Train a gradient-boosted tree model for each lead length using LightGBM.

    Attributes:
        config (dict): User-provided configuration parameters.
        data (pd.core.frame.DataFrame): User-provided panel data.
        categorical_features (list): Column names of categorical features.
        reserved_cols (list): Column names of non-features.
        numeric_features (list): Column names of numeric features.
        n_intervals (int): The largest number of one-period intervals any
            individual is observed to survive.
        models (list): A trained LightGBM model (lgb.basic.Booster) for each
            lead length.
    """

    def train(self) -> List[lgb.basic.Booster]:
        """Train a LightGBM model for each lead length."""
        models = []
        train_subset = (~self.data[self.validation_col]
                        & ~self.data[self.test_col]
                        & ~self.data[self.predict_col])
        validation_subset = (self.data[self.validation_col]
                             & ~self.data[self.test_col]
                             & ~self.data[self.predict_col])
        for time_horizon in range(self.n_intervals):
            train_data = self.data[(self.data[self.duration_col]
                                    + self.data[self.event_col]
                                    > time_horizon)
                                   & train_subset]
            train_data = lgb.Dataset(
                train_data[self.categorical_features + self.numeric_features],
                label=train_data[self.duration_col] > time_horizon)
            validation_data = self.data[(self.data[self.duration_col]
                                         + self.data[self.event_col]
                                         > time_horizon)
                                        & validation_subset]
            validation_data = train_data.create_valid(
                validation_data[self.categorical_features
                                + self.numeric_features],
                label=validation_data[self.duration_col] > time_horizon)
            model = lgb.train({'objective': 'binary'},
                              train_data,
                              num_boost_round=self.config['MAX_EPOCHS'],
                              early_stopping_rounds=self.config['PATIENCE'],
                              valid_sets=[validation_data],
                              valid_names=['validation_set'],
                              categorical_feature=self.categorical_features,
                              verbose_eval=True)
            models.append(model)
        return models

    def predict(self, subset: Union[None, pd.core.series.Series] = None,
                cumulative: bool = True) -> np.ndarray:
        """Use trained LightGBM models to predict observation survival rates.

        Args:
            subset: A Boolean Series that is True for observations for which
                predictions will be produced. If None, default to all
                observations.
                cumulative: If True, produce cumulative survival probabilies.
                If False, produce marginal survival probabilities (i.e., one
                minus the hazard rate).

        Returns:
            A numpy array of survival probabilities by observation and lead
            length.
        """
        subset = survival_modeler.default_subset_to_all(subset, self.data)
        predict_data = self.data[self.categorical_features
                                 + self.numeric_features][subset]
        predictions = np.array([lead_specific_model.predict(predict_data)
                                for lead_specific_model in self.model]).T
        if cumulative:
            predictions = np.cumprod(predictions, axis=1)
        return predictions

    def compute_shap_values(
            self, subset: Union[None, pd.core.series.Series] = None) -> dict:
        """Compute SHAP values by lead length, observation, and feature."""
        subset = survival_modeler.default_subset_to_all(subset, self.data)
        shap_subset = self.data[self.categorical_features
                                + self.numeric_features][subset]
        shap_values = {str(i + 1) + '_lead':
                       shap.TreeExplainer(
                           lead_specific_model).shap_values(shap_subset)[0]
                       for i, lead_specific_model in enumerate(self.model)
                       if lead_specific_model.num_trees() > 0}
        return shap_values
