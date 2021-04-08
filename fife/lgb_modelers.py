"""FIFE modelers based on LightGBM, which trains gradient-boosted trees."""

import json
from typing import List, Union
from warnings import warn

import dask
from fife.base_modelers import (
    default_subset_to_all,
    Modeler,
    SurvivalModeler,
    StateModeler,
    ExitModeler,
)
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import shap


class LGBModeler(Modeler):
    """Train a gradient-boosted tree model for each lead length using LightGBM.

    Attributes:
        config (dict): User-provided configuration parameters.
        data (pd.core.frame.DataFrame): User-provided panel data.
        categorical_features (list): Column names of categorical features.
        duration_col (str): Name of the column representing the number of
            future periods observed for the given individual.
        event_col (str): Name of the column indicating whether the individual
            is observed to exit the dataset.
        predict_col (str): Name of the column indicating whether the
            observation will be used for prediction after training.
        test_col (str): Name of the column indicating whether the observation
            will be used for testing model performance after training.
        validation_col (str): Name of the column indicating whether the
            observation will be used for evaluating model performance during
            training.
        period_col (str): Name of the column representing the number of
            periods since the earliest period in the data.
        max_lead_col (str): Name of the column representing the number of
            observable future periods.
        spell_col (str): Name of the column representing the number of
            previous spells of consecutive observations of the same individual.
        weight_col (str): Name of the column representing observation weights.
        reserved_cols (list): Column names of non-features.
        numeric_features (list): Column names of numeric features.
        n_intervals (int): The largest number of periods ahead to forecast.
        model (list): A trained LightGBM model (lgb.basic.Booster) for each
            lead length.
        objective (str): The LightGBM model objective appropriate for the
            outcome type, which is "binary" for binary classification.
        num_class (int): The num_class LightGBM parameter, which is 1 for
            binary classification.
    """

    def hyperoptimize(
        self,
        n_trials: int = 64,
        rolling_validation: bool = True,
        subset: Union[None, pd.core.series.Series] = None,
    ) -> dict:
        """Search for hyperparameters with greater out-of-sample performance.

        Args:
            n_trials: The number of hyperparameter sets to evaluate for each
                time horizon. Return None if non-positive.
            rolling_validation: Whether or not to evaluate performance on the
                most recent possible periods instead of the validation set
                labeled by self.validation_col. Ignored for a given time horizon
                if there is only one possible period for training and evaluation.
            subset:  A Boolean Series that is True for observations on which
                to train and validate. If None, default to all observations not
                flagged by self.test_col or self.predict_col.

        Returns:
            A dictionary containing the best-performing parameter dictionary for
            each time horizon.
        """

        def evaluate_params(
            trial: optuna.trial.Trial,
            train_data: lgb.Dataset,
            validation_data: lgb.Dataset,
        ) -> Union[None, dict]:
            """Compute out-of-sample performance for a parameter set."""
            params = {}
            params["num_iterations"] = trial.suggest_int("num_iterations", 8, 128)
            params["learning_rate"] = trial.suggest_uniform(
                "learning_rate", 2 ** -5, 0.5
            )
            params["num_leaves"] = trial.suggest_int("num_leaves", 8, 256)
            params["max_depth"] = trial.suggest_int("max_depth", 4, 32)
            params["min_data_in_leaf"] = trial.suggest_int("min_data_in_leaf", 4, 512)
            params["min_sum_hessian_in_leaf"] = trial.suggest_uniform(
                "min_sum_hessian_in_leaf", 2 ** -5, 0.25
            )
            params["bagging_freq"] = trial.suggest_int("bagging_freq", 0, 1)
            params["bagging_fraction"] = trial.suggest_uniform(
                "bagging_fraction", 0.5, 1
            )
            params["feature_fraction"] = trial.suggest_uniform(
                "feature_fraction", 0.5, 1
            )
            params["lambda_l1"] = trial.suggest_uniform("lambda_l1", 0, 64)
            params["lambda_l2"] = trial.suggest_uniform("lambda_l2", 0, 64)
            params["min_gain_to_split"] = trial.suggest_uniform(
                "min_gain_to_split", 0, 0.25
            )
            params["min_data_per_group"] = trial.suggest_int(
                "min_data_per_group", 1, 512
            )
            params["max_cat_threshold"] = trial.suggest_int("max_cat_threshold", 1, 512)
            params["cat_l2"] = trial.suggest_uniform("cat_l2", 0, 64)
            params["cat_smooth"] = trial.suggest_uniform("cat_smooth", 0, 2048)
            params["max_cat_to_onehot"] = trial.suggest_int("max_cat_to_onehot", 1, 64)
            params["max_bin"] = trial.suggest_int("max_bin", 32, 1024)
            params["min_data_in_bin"] = trial.suggest_int("min_data_in_bin", 1, 64)
            params["objective"] = self.objective
            params["num_class"] = self.num_class
            params["verbosity"] = -1
            booster = lgb.Booster(params=params, train_set=train_data)
            booster.add_valid(validation_data, "validation_set")
            for step in range(params["num_iterations"]):
                booster.update()
                validation_loss = booster.eval_valid()[0][2]
                trial.report(validation_loss, step)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            return validation_loss

        default_params = {
            "objective": self.objective,
            "num_iterations": 100,
            "num_class": self.num_class,
        }
        if n_trials <= 0:
            return {
                time_horizon: default_params for time_horizon in range(self.n_intervals)
            }
        params = {}
        if subset is None:
            subset = ~self.data[self.test_col] & ~self.data[self.predict_col]
        for time_horizon in range(self.n_intervals):
            data = self.label_data(time_horizon)
            data = data[subset]
            data = self.subset_for_training_horizon(data, time_horizon)
            if rolling_validation:
                rolling_validation_n_periods = int(
                    max(
                        self.config.get("VALIDATION_SHARE", 0.25)
                        * data[self.config["TIME_IDENTIFIER"]].nunique(),
                        1,
                    )
                )
                rolling_validation_period_cutoff = pd.factorize(
                    data[self.config["TIME_IDENTIFIER"]], sort=True
                )[1][-rolling_validation_n_periods]
                rolling_validation_subset = (
                    data[self.config["TIME_IDENTIFIER"]]
                    >= rolling_validation_period_cutoff
                )
            else:
                rolling_validation_subset = [False] * data.shape[0]
            if (data[rolling_validation_subset]["_label"].nunique() > 1) and (
                data[~rolling_validation_subset]["_label"].nunique() > 1
            ):
                validation_subset = rolling_validation_subset
            else:
                validation_subset = data[self.validation_col]
            train_data = lgb.Dataset(
                data[~validation_subset][
                    self.categorical_features + self.numeric_features
                ],
                label=data[~validation_subset]["_label"],
                weight=data[~validation_subset][self.weight_col]
                if self.weight_col
                else None,
            )
            validation_data = train_data.create_valid(
                data[validation_subset][
                    self.categorical_features + self.numeric_features
                ],
                label=data[validation_subset]["_label"],
                weight=data[validation_subset][self.weight_col]
                if self.weight_col
                else None,
            )
            study = optuna.create_study(
                pruner=optuna.pruners.MedianPruner(),
                sampler=optuna.samplers.TPESampler(seed=self.config.get("SEED", 9999)),
            )
            study.optimize(
                lambda trial: evaluate_params(trial, train_data, validation_data),
                n_trials=n_trials,
            )
            params[time_horizon] = study.best_params
            params[time_horizon]["objective"] = self.objective
            default_booster = lgb.Booster(params=default_params, train_set=train_data)
            default_booster.add_valid(validation_data, "validation_set")
            for _ in range(default_params["num_iterations"]):
                default_booster.update()
            default_validation_loss = default_booster.eval_valid()[0][2]
            if default_validation_loss <= study.best_value:
                params[time_horizon] = default_params
        return params

    def build_model(
        self,
        n_intervals: Union[None, int] = None,
        params: dict = None,
        parallelize: bool = True,
    ) -> None:
        """Train and store a sequence of gradient-boosted tree models."""
        if n_intervals:
            self.n_intervals = n_intervals
        else:
            self.n_intervals = self.set_n_intervals()
        early_stopping = (params is None) or (
            any(["num_iterations" not in d for d in params.values()])
        )
        self.model = self.train(
            params=params,
            validation_early_stopping=early_stopping,
            parallelize=parallelize,
        )

    def train(
        self,
        params: Union[None, dict] = None,
        subset: Union[None, pd.core.series.Series] = None,
        validation_early_stopping: bool = True,
        parallelize: bool = True,
    ) -> List[lgb.basic.Booster]:
        """Train a LightGBM model for each lead length."""
        models = []
        if parallelize:
            for time_horizon in range(self.n_intervals):
                model = dask.delayed(self.train_single_model)(
                    time_horizon=time_horizon,
                    params=params,
                    subset=subset,
                    validation_early_stopping=validation_early_stopping,
                )
                models.append(model)
            models = dask.compute(*models)
        else:
            for time_horizon in range(self.n_intervals):
                model = self.train_single_model(
                    time_horizon=time_horizon,
                    params=params,
                    subset=subset,
                    validation_early_stopping=validation_early_stopping,
                )
                models.append(model)
        return models

    def train_single_model(
        self,
        time_horizon: int,
        params: Union[None, dict] = None,
        subset: Union[None, pd.core.series.Series] = None,
        validation_early_stopping: bool = True,
    ) -> lgb.basic.Booster:
        """Train a LightGBM model for a single lead length."""
        if params is None:
            params = {
                time_horizon: {
                    "objective": self.objective,
                    "num_iterations": self.config.get("MAX_EPOCHS", 256),
                }
            }
        params[time_horizon]["num_class"] = self.num_class
        if subset is None:
            subset = ~self.data[self.test_col] & ~self.data[self.predict_col]
        data = self.label_data(time_horizon)
        data = data[subset]
        data = self.subset_for_training_horizon(data, time_horizon)
        if validation_early_stopping:
            train_data = lgb.Dataset(
                data[~data[self.validation_col]][
                    self.categorical_features + self.numeric_features
                ],
                label=data[~data[self.validation_col]]["_label"],
                weight=data[~data[self.validation_col]][self.weight_col]
                if self.weight_col
                else None,
            )
            validation_data = train_data.create_valid(
                data[data[self.validation_col]][
                    self.categorical_features + self.numeric_features
                ],
                label=data[data[self.validation_col]]["_label"],
                weight=data[data[self.validation_col]][self.weight_col]
                if self.weight_col
                else None,
            )
            model = lgb.train(
                params[time_horizon],
                train_data,
                early_stopping_rounds=self.config.get("PATIENCE", 4),
                valid_sets=[validation_data],
                valid_names=["validation_set"],
                categorical_feature=self.categorical_features,
                verbose_eval=True,
            )
        else:
            data = lgb.Dataset(
                data[self.categorical_features + self.numeric_features],
                label=data["_label"],
                weight=data[self.weight_col] if self.weight_col else None,
            )
            model = lgb.train(
                params[time_horizon],
                data,
                categorical_feature=self.categorical_features,
                verbose_eval=True,
            )
        return model

    def predict(
        self, subset: Union[None, pd.core.series.Series] = None, cumulative: bool = True
    ) -> np.ndarray:
        """Use trained LightGBM models to predict the outcome for each observation and time horizon.

        Args:
            subset: A Boolean Series that is True for observations for which
                predictions will be produced. If None, default to all
                observations.
            cumulative: If True, produce cumulative survival probabilies.
                If False, produce marginal survival probabilities (i.e., one
                minus the hazard rate).

        Returns:
            A numpy array of predictions by observation and lead
            length.
        """
        subset = default_subset_to_all(subset, self.data)
        predict_data = self.data[self.categorical_features + self.numeric_features][
            subset
        ]
        predictions = np.array(
            [
                lead_specific_model.predict(predict_data)
                for lead_specific_model in self.model
            ]
        ).T
        if cumulative:
            predictions = np.cumprod(predictions, axis=1)
        return predictions

    def transform_features(self) -> pd.DataFrame:
        """Transform features to suit model training."""
        data = self.data.copy(deep=True)
        if self.config.get("DATETIME_AS_DATE", True):
            date_cols = list(data.select_dtypes("datetime").columns) + [
                col
                for col in data.select_dtypes("category")
                if np.issubdtype(data[col].cat.categories.dtype, np.datetime64)
            ]
            for col in date_cols:
                data[col] = (
                    data[col].dt.year * 10000
                    + data[col].dt.month * 100
                    + data[col].dt.day
                )
        else:
            data[data.select_dtypes("datetime").columns] = data[
                data.select_dtypes("datetime").columns
            ].apply(pd.to_numeric)
            for col in data.select_dtypes("category"):
                if np.issubdtype(data[col].cat.categories.dtype, np.datetime64):
                    data[col] = data[col].astype(int)
        return data

    def save_model(self, file_name: str = "GBT_Model", path: str = "") -> None:
        """Save the horizon-specific LightGBM models that comprise the model to disk."""
        for i, lead_specific_model in enumerate(self.model):
            with open(f"{path}{i + 1}-lead_{file_name}.json", "w") as file:
                json.dump(lead_specific_model.dump_model(), file, indent=4)

    def compute_shap_values(
        self, subset: Union[None, pd.core.series.Series] = None
    ) -> dict:
        """Compute SHAP values by lead length, observation, and feature."""
        subset = default_subset_to_all(subset, self.data)
        shap_subset = self.data[self.categorical_features + self.numeric_features][
            subset
        ]
        shap_values = {
            str(i + 1)
            + "_lead": shap.TreeExplainer(lead_specific_model).shap_values(shap_subset)[
                0
            ]
            for i, lead_specific_model in enumerate(self.model)
            if lead_specific_model.num_trees() > 0
        }
        return shap_values


class LGBSurvivalModeler(LGBModeler, SurvivalModeler):
    """Use LightGBM to forecast probabilities of being observed in future periods."""

    pass


class GradientBoostedTreesModeler(LGBSurvivalModeler):
    """Deprecated alias for LGBSurvivalModeler"""

    warn(
        "The name 'GradientBoostedTreesModeler' is deprecated. "
        "Please use 'LGBSurvivalModeler' instead.",
        DeprecationWarning,
    )


class LGBStateModeler(LGBModeler, StateModeler):
    """Use LightGBM to forecast the future value of a feature conditional on survival."""

    pass


class LGBExitModeler(LGBModeler, ExitModeler):
    """Use LightGBM to forecast the circumstance of exit conditional on exit."""

    pass
