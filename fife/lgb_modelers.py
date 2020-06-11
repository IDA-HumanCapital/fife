"""FIFE modeler based on LightGBM, which trains gradient-boosted trees."""

from typing import List, Union

from fife import survival_modeler
import lightgbm as lgb
import numpy as np
import optuna
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

    def hyperoptimize(self,
                      n_trials: int = 64,
                      rolling_validation: bool = True,
                      train_subset: Union[None, pd.core.series.Series] = None) -> dict:
        """Search for hyperparameters with greater out-of-sample performance.

        Args:
            n_trials: The number of hyperparameter sets to evaluate for each
                time horizon. Return None if non-positive.
            rolling_validation: Whether or not to evaluate performance on the
                most recent possible period instead of the validation set
                labeled by self.validation_col. Ignored for a given time horizon
                if there is only one possible period for training and evaluation.
            train_subset:  A Boolean Series that is True for observations on which
                to train. If None, default to all observations not flagged by
                self.validation_col, self.test_col, or self.predict_col.

        Returns:
            A dictionary containing the best-performing parameter dictionary for
            each time horizon.
        """
        def evaluate_params(trial: optuna.trial.Trial,
                            train_data: lgb.Dataset,
                            validation_data: lgb.Dataset) -> Union[None, dict]:
            """Compute out-of-sample performance for a parameter set."""
            params = {}
            params['num_iterations'] = trial.suggest_int('num_iterations', 8, 128)
            params['learning_rate'] = trial.suggest_uniform('learning_rate', 2e-5, 0.5)
            params['num_leaves'] = trial.suggest_int('num_leaves', 8, 256)
            params['max_depth'] = trial.suggest_int('max_depth', 4, 32)
            params['min_data_in_leaf'] = trial.suggest_int('min_data_in_leaf', 4, 512)
            params['min_sum_hessian_in_leaf'] = trial.suggest_uniform('min_sum_hessian_in_leaf', 2e-5, 0.25)
            params['bagging_freq'] = trial.suggest_int('bagging_freq', 0, 1)
            params['bagging_fraction'] = trial.suggest_uniform('bagging_fraction', 0.5, 1)
            params['feature_fraction'] = trial.suggest_uniform('feature_fraction', 0.5, 1)
            params['lambda_l1'] = trial.suggest_uniform('lambda_l1', 0, 64)
            params['lambda_l2'] = trial.suggest_uniform('lambda_l2', 0, 64)
            params['min_gain_to_split'] = trial.suggest_uniform('min_gain_to_split', 0, 0.25)
            params['min_data_per_group'] = trial.suggest_int('min_data_per_group', 1, 512)
            params['max_cat_threshold'] = trial.suggest_int('max_cat_threshold', 1, 512)
            params['cat_l2'] = trial.suggest_uniform('cat_l2', 0, 64)
            params['cat_smooth'] = trial.suggest_uniform('cat_smooth', 0, 2048)
            params['max_cat_to_onehot'] = trial.suggest_int('max_cat_to_onehot', 1, 64)
            params['max_bin'] = trial.suggest_int('max_bin', 32, 1024)
            params['min_data_in_bin'] = trial.suggest_int('min_data_in_bin', 1, 64)
            params['objective'] = 'binary'
            params['verbosity'] = -1
            booster = lgb.Booster(params=params,
                                  train_set=train_data)
            booster.add_valid(validation_data, 'validation_set')
            for step in range(params['num_iterations']):
                booster.update()
                validation_loss = booster.eval_valid()[0][2]
                trial.report(validation_loss, step)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            return validation_loss

        if n_trials <= 0:
            return None
        params = {}
        if train_subset is None:
            train_subset = (~self.data[self.validation_col]
                            & ~self.data[self.test_col]
                            & ~self.data[self.predict_col])
        for time_horizon in range(self.n_intervals):
            train_data = self.data[(self.data[self.duration_col]
                                    + self.data[self.event_col]
                                    > time_horizon)
                                   & train_subset]
            if rolling_validation:
                rolling_validation_n_periods = int(max(self.config.get('VALIDATION_SHARE', 0.25)
                                                       * train_data[self.config['TIME_IDENTIFIER']].nunique(), 1))
                rolling_validation_period_cutoff = pd.factorize(train_data[self.config['TIME_IDENTIFIER']],
                                                                sort=True)[1][-rolling_validation_n_periods]
                rolling_validation_subset = train_data[self.config['TIME_IDENTIFIER']] >= rolling_validation_period_cutoff
            else:
                rolling_validation_subset = [False] * train_data.shape[0]
            if (((train_data[self.duration_col][rolling_validation_subset] > time_horizon).nunique() > 1)
                and ((train_data[self.duration_col][~rolling_validation_subset] > time_horizon).nunique() > 1)):
                validation_data = train_data[rolling_validation_subset]
                train_data = train_data[~rolling_validation_subset]
            else:
                validation_subset = (self.data[self.validation_col]
                                     & ~self.data[self.test_col]
                                     & ~self.data[self.predict_col])
                validation_data = self.data[(self.data[self.duration_col]
                                             + self.data[self.event_col]
                                             > time_horizon)
                                            & validation_subset]
            train_data = lgb.Dataset(
                train_data[self.categorical_features + self.numeric_features],
                label=train_data[self.duration_col] > time_horizon)
            validation_data = train_data.create_valid(
                validation_data[self.categorical_features
                                + self.numeric_features],
                label=validation_data[self.duration_col] > time_horizon)
            study = optuna.create_study(pruner=optuna.pruners.MedianPruner(),
                                        sampler=optuna.samplers.TPESampler(seed=self.config.get('SEED', 9999)))
            study.optimize(lambda trial: evaluate_params(trial,
                                                         train_data,
                                                         validation_data),
                           n_trials=n_trials)
            params[time_horizon] = study.best_params
            params[time_horizon]['objective'] = 'binary'
            default_params = {'objective': 'binary',
                              'num_iterations': 100}
            default_booster = lgb.Booster(params=default_params,
                                          train_set=train_data)
            default_booster.add_valid(validation_data, 'validation_set')
            for _ in range(default_params['num_iterations']):
                default_booster.update()
            default_validation_loss = default_booster.eval_valid()[0][2]
            if default_validation_loss <= study.best_value:
                params = default_params
        return params

    def train(self,
              params: Union[None, dict] = None,
              validation_early_stopping: bool = True,
              train_subset: Union[None, pd.core.series.Series] = None) -> List[lgb.basic.Booster]:
        """Train a LightGBM model for each lead length."""
        models = []
        if params is None:
            params = {time_horizon: {'objective': 'binary',
                                     'num_iterations': self.config.get('MAX_EPOCHS', 256)}
                      for time_horizon in range(self.n_intervals)}
        if train_subset is None:
            train_subset = (~self.data[self.validation_col]
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
            if validation_early_stopping:
                validation_subset = (self.data[self.validation_col]
                                     & ~self.data[self.test_col]
                                     & ~self.data[self.predict_col])
                validation_data = self.data[(self.data[self.duration_col]
                                             + self.data[self.event_col]
                                             > time_horizon)
                                            & validation_subset]
                validation_data = train_data.create_valid(
                    validation_data[self.categorical_features
                                    + self.numeric_features],
                    label=validation_data[self.duration_col] > time_horizon)
                model = lgb.train(params[time_horizon],
                                  train_data,
                                  early_stopping_rounds=self.config.get('PATIENCE', 4),
                                  valid_sets=[validation_data],
                                  valid_names=['validation_set'],
                                  categorical_feature=self.categorical_features,
                                  verbose_eval=True)
            else:
                model = lgb.train(params[time_horizon],
                                  train_data,
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
