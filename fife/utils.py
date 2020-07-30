"""I/O, logging, calculation, and plotting functions for FIFE."""

import argparse
import json
import os
import pickle
import random as rn
import sys
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap


def make_results_reproducible(seed: int = 9999) -> None:
    """Ensure executing from a fresh state produces identical results."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    rn.seed(seed)


def ensure_folder_existence(path: str = "") -> None:
    """Create a directory if it doesn't already exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def redirect_output_to_log(path: str = "") -> None:
    """Send future output to a text file instead of console."""
    ensure_folder_existence(f"{path}/Output/Logs/")
    sys.stdout = open(f"{path}/Output/Logs/Log.txt", "w")


def print_config(config: dict) -> None:
    """Neatly print given dictionary of config parameters."""
    title = "CONFIGURATION PARAMETERS"
    print(title)
    print("-" * len(title))
    print(json.dumps(config, indent=4) + "\n")


def import_data_file(file_path: str = "Input_Data") -> pd.core.frame.DataFrame:
    """Return the data stored in given file in given folder."""
    if file_path.endswith(".feather"):
        data = pd.read_feather(file_path)
    elif file_path.endswith(".csv"):
        data = pd.read_csv(file_path)
    elif file_path.endswith(".csv.gz"):
        data = pd.read_csv(file_path)
    elif file_path.endswith((".p", "pkl")):
        data = pd.read_pickle(file_path)
    elif file_path.endswith(".h5"):
        data = pd.read_hdf(file_path)
    elif file_path.endswith(".json"):
        data = pd.read_json(file_path)
    else:
        raise Exception(
            "Data file extension is invalid. See README for " + "valid file extensions."
        )
    return data


def compute_aggregation_uncertainty(
    individual_probabilities: pd.core.frame.DataFrame, percent_confidence: float = 0.95
) -> pd.core.frame.DataFrame:
    """Statistically bound number of events given each of their probabilities.

    Args:
        individual_probabilities: A DataFrame of probabilities where each row
            represents an individual and each column represents an event.
        percent_confidence: The percent confidence of the two-sided intervals
            defined by the computed bounds.

    Raises:
        ValueError: If percent_confidence is outside of the interval (0, 1).

    Returns:
        A DataFrame containing, for each column in individual_probabilities,
        the expected number of events and interval bounds on the number of
        events.
    """
    if (percent_confidence <= 0) | (percent_confidence >= 1):
        raise ValueError("Percent confidence outside (0, 1)")
    means = individual_probabilities.transpose().to_numpy().sum(axis=1)
    cutoff = np.log((1 - percent_confidence) / 2) / means
    upper_bounds = (
        1 + ((-cutoff) + np.sqrt(np.square(cutoff) - (8 * cutoff))) / 2
    ) * means
    upper_bounds = np.minimum(upper_bounds, individual_probabilities.shape[0])
    lower_bounds = np.maximum((1 - np.sqrt(-2 * cutoff)) * means, 0)
    lower_bounds = np.maximum(lower_bounds, 0)
    conf = str(int(percent_confidence * 100))
    times = list(individual_probabilities)
    output = pd.DataFrame(
        {
            "Event": times,
            "ExpectedEventCount": means,
            "Lower" + conf + "PercentBound": lower_bounds,
            "Upper" + conf + "PercentBound": upper_bounds,
        }
    )
    return output


def plot_binary_prediction_errors(
    errors: dict,
    width: float = 8,
    height: float = 1,
    alpha: float = 2 ** -8,
    color: str = "black",
    center_tick_color: str = "green",
    center_tick_height: float = 0.125,
    path: str = "",
) -> None:
    """Make a rug plot of binary prediction errors.

    Args:
        errors: A dictionary of numpy arrays, each of which contains error values
            for the outcome given by its key.
        width: Width of the rug plot.
        height: Height of the rug plot.
        alpha: The opacity of plotted ticks, from 2e-8 (nearly transparent) to
            1 (opaque).
        color: The color of plotted ticks.
        center_tick_color: The color of the ticks marking the center of the plot.
        center_tick_height: The height of the ticks marking the center of the plot.
        path: The path preceding the Output folder in which the plots will be
            saved.
    """
    sns.set(rc={"figure.figsize": (width, height)})
    for key, arr in errors.items():
        sns.rugplot(arr, height=height, alpha=alpha, color=color)
        plt.xlim(-1, 1)
        plt.axvline(ymin=0, ymax=center_tick_height, color=center_tick_color)
        plt.axvline(
            ymin=height - center_tick_height, ymax=height, color=center_tick_color
        )
        plt.xticks([])
        plt.yticks([])
        save_plot(f"Errors_{key}", path=path)


def plot_shap_values(
    shap_values: dict,
    raw_data: pd.core.frame.DataFrame,
    processed_data: Union[None, pd.core.frame.DataFrame] = None,
    no_summary_col: str = Union[None, str],
    alpha: float = 0.5,
    path: str = "",
) -> None:
    """Make plots of SHAP values.

    SHAP values quantify feature contributions to predictions.

    Args:
        shap_values: A dictionary of numpy arrays, each of which contains SHAP
            values for the outcome given by its key.
        raw_data: Feature values prior to processing into model input.
        processed_data: Feature values used as model input.
        no_summary_col: The name of a column to never use for summary plots.
        alpha: The opacity of plotted points, from 2e-8 (nearly transparent) to
            1 (opaque).
        path: The path preceding the Output folder in which the plots will be
            saved.
    """
    shap.initjs()
    if processed_data is None:
        processed_data = raw_data
    for col in processed_data.select_dtypes("category"):
        processed_data[col] = processed_data[col].cat.codes
    for key, arr in shap_values.items():
        shap.summary_plot(
            arr, plot_type="bar", feature_names=raw_data.columns, show=False
        )
        save_plot(f"Importance_{key}", path=path)
        shap.summary_plot(arr, raw_data, alpha=alpha, show=False)
        save_plot(f"Summary_{key}", path=path)
        if raw_data.columns[np.argmax(np.abs(arr).mean(axis=0))] == no_summary_col:
            shap.dependence_plot(
                f"rank(1)",
                arr,
                processed_data,
                display_features=raw_data,
                alpha=alpha,
                show=False,
            )
        else:
            shap.dependence_plot(
                f"rank(0)",
                arr,
                processed_data,
                display_features=raw_data,
                alpha=alpha,
                show=False,
            )
        save_plot(f"Dependence_{key}", path=path)


def save_maps(
    obj: Union[pd.core.series.Series, dict], file_name: str, path: str = ""
) -> None:
    """Save a map from values to other values to Intermediate folder."""
    ensure_folder_existence(f"{path}/Intermediate/Data")
    with open(f"{path}/Intermediate/Data/{file_name}.p", "wb") as file:
        pickle.dump(obj, file)


def save_intermediate_data(
    data: pd.core.frame.DataFrame,
    file_name: str,
    file_format: str = "pickle",
    path: str = "",
) -> None:
    """Save given DataFrame in Intermediate folder in given format."""
    ensure_folder_existence(f"{path}/Intermediate/Data")
    if file_format == "csv":
        data.to_csv(f"{path}/Intermediate/Data/{file_name}.csv")
    elif file_format == "feather":
        data.to_feather(f"{path}/Intermediate/Data/{file_name}.feather")
    elif file_format == "hdf":
        data.to_hdf(f"{path}/Intermediate/Data/{file_name}.h5", key=file_name)
    elif file_format == "pickle":
        data.to_pickle(f"{path}/Intermediate/Data/{file_name}.p")
    else:
        raise Exception(
            "File format invalid. Valid file formats "
            + 'are "csv", "feather", "hdf", and "pickle".'
        )


def save_output_table(
    data: pd.core.frame.DataFrame, file_name: str, index: bool = True, path: str = ""
) -> None:
    """Save given DataFrame in the Output folder as a csv file."""
    ensure_folder_existence(f"{path}/Output/Tables")
    data.to_csv(f"{path}/Output/Tables/{file_name}.csv", index=index)


def save_plot(file_name: str, path: str = "") -> None:
    """Save the most recently plotted plot in high resolution."""
    ensure_folder_existence(f"{path}/Output/Figures")
    plt.tight_layout()
    plt.savefig(f"{path}/Output/Figures/{file_name}.png", dpi=512)
    plt.close()


def create_example_data(n_persons: int = 8192, n_periods: int = 12
) -> pd.core.frame.DataFrame:
    """Fabricate an unbalanced panel dataset suitable as FIFE input."""
    seed = 9999
    np.random.seed(seed)
    values = []
    for i in np.arange(n_persons):
        period = np.random.randint(n_periods) + 1
        x_1 = np.random.uniform()
        x_2 = np.random.choice(["A", "B", "C"])
        x_3 = np.random.uniform() + 1.0
        x_4 = np.random.choice(["a", "b", "c", 1, 2, 3, np.nan])
        while period <= n_periods:
            values.append([i, period, x_1, x_2, x_3, x_4])
            if x_2 == "A":
                x_1 += np.random.uniform(0, 0.1)
            else:
                x_1 += np.random.uniform(0, 0.2)
            if x_1 > np.sqrt(x_3):
                break
            period += 1
    values = pd.DataFrame(
        values,
        columns=[
            "individual",
            "period",
            "feature_1",
            "feature_2",
            "feature_3",
            "feature_4",
        ],
    )
    values["feature_5"] = values["feature_2"]
    return values


class FIFEArgParser(argparse.ArgumentParser):
    """Argument parser for the FIFE command-line interface."""

    def __init__(self):
        super().__init__()
        self.add_argument(
            "CONFIG_PATH",
            type=str,
            nargs="?",
            default=None,
            help="A relative or absolute path to a configuration file.",
        )
        self.add_argument(
            "--DATA_FILE_PATH",
            type=str,
            help="A relative or absolute path to the input data file.",
        )
        self.add_argument(
            "--NOTES_FOR_LOG",
            type=str,
            help="Custom text that will be printed in the log produced during execution.",
        )
        self.add_argument(
            "--RESULTS_PATH",
            type=str,
            default="FIFE_results",
            help="The relative or absolute path on which Intermediate and Output folders will be stored.",
        )
        self.add_argument(
            "--SEED",
            type=int,
            default=9999,
            help="The initializing value for all random number generators.",
        )
        self.add_argument(
            "--INDIVIDUAL_IDENTIFIER",
            type=str,
            help="The name of the feature that identifies individuals that persist over multiple time periods in the data.",
        )
        self.add_argument(
            "--TIME_IDENTIFIER",
            type=str,
            help="The name of the feature that identifies time periods in the data.",
        )
        self.add_argument(
            "--CATEGORICAL_SUFFIXES",
            type=str,
            nargs="+",
            help="Optional list of suffixes denoting that columns ending with such a suffix should be treated as categorical.",
        )
        self.add_argument(
            "--DATETIME_AS_DATE",
            type=bool,
            help="How datetime features will be represented for the gradient-boosted trees modeler.",
        )
        self.add_argument(
            "--MAX_NULL_SHARE",
            type=float,
            help="The maximum share of observations that may have a null value for a feature to be kept for training.",
        )
        self.add_argument(
            "--MAX_UNIQUE_NUMERIC_CATS",
            type=int,
            help="The maximum number of unique values for a feature of a numeric type to be considered categorical.",
        )
        self.add_argument(
            "--NUMERIC_SUFFIXES",
            type=str,
            nargs="+",
            help="Optional list of suffixes denoting that columns ending with such a suffix should be treated as numeric.",
        )
        self.add_argument(
            "--MIN_SURVIVORS_IN_TRAIN",
            type=int,
            help="The minimum number of training set observations surviving a given time horizon for the model to be trained to make predictions for that time horizon.",
        )
        self.add_argument(
            "--TEST_INTERVALS",
            type=int,
            help="The number of most recent periods to treat as absent from the data during training for the purpose of model evaluation.",
        )
        self.add_argument(
            "--TEST_PERIODS",
            type=int,
            help="One plus the value represented by TEST_INTERVALS. Deprecated and overriden by TEST_INTERVALS.",
        )
        self.add_argument(
            "--VALIDATION_SHARE",
            type=float,
            help="The share of observations used for evaluation instead of training for hyperoptimization or early stopping.",
        )
        self.add_argument(
            "--TREE_MODELS",
            type=bool,
            default=True,
            help="Whether FIFE will train gradient-boosted trees, as opposed to a neural network.",
        )
        self.add_argument(
            "--HYPER_TRIALS",
            type=int,
            help="The number of hyperparameter sets to trial.",
        )
        self.add_argument(
            "--MAX_EPOCHS",
            type=int,
            help="If HYPER_TRIALS is zero, the maximum number of passes through the training set.",
        )
        self.add_argument(
            "--PATIENCE",
            type=int,
            help="If HYPER_TRIALS is zero, the number of passes through the training dataset without improvement in validation set performance before training is stopped early.",
        )
        self.add_argument(
            "--BATCH_SIZE",
            type=int,
            help="The number of observations per batch of data fed to the machine learning model for training. ",
        )
        self.add_argument(
            "--DENSE_LAYERS",
            type=int,
            help="The number of dense layers in the neural network.",
        )
        self.add_argument(
            "--DROPOUT_SHARE",
            type=float,
            help="The probability of a densely connected node of the neural network being set to zero weight during training.",
        )
        self.add_argument(
            "--EMBED_EXPONENT",
            type=float,
            help="	The ratio of the natural logarithm of the number of embedded values to the natural logarithm of the number of unique categories for each categorical feature.",
        )
        self.add_argument(
            "--EMBED_L2_REG",
            type=float,
            help="The L2 regularization coefficient for each embedding layer.",
        )
        self.add_argument(
            "--NODES_PER_DENSE_LAYER",
            type=int,
            help="The number of nodes per dense layer in the neural network.",
        )
        self.add_argument(
            "--NON_CAT_MISSING_VALUE",
            type=float,
            help="The value used to replace missing values of numeric features.",
        )
        self.add_argument(
            "--PROPORTIONAL_HAZARDS",
            type=bool,
            help="Whether FIFE will restrict the neural network to a proportional hazards model.",
        )
        self.add_argument(
            "--QUANTILES",
            type=int,
            default=5,
            help="The number of similarly-sized bins for which survival and total counts will be reported for each time horizon.",
        )
        self.add_argument(
            "--RETENTION_INTERVAL",
            type=int,
            default=1,
            help="The number of periods over which retention rates are computed.",
        )
        self.add_argument(
            "--SHAP_PLOT_ALPHA",
            type=float,
            help="The transparency of points in SHAP plots.",
        )
        self.add_argument(
            "--SHAP_SAMPLE_SIZE",
            type=int,
            default=128,
            help="The number of observations randomly sampled for SHAP value calculation and plotting.",
        )
