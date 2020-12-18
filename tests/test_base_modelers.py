"""Conduct unit testing for fife.base_modelers module."""

from fife import base_modelers
import numpy as np
import pandas as pd


def test_compute_metrics_for_binary_outcome(fabricate_forecasts):
    """Test that FIFE produces correct example AUROC and confusion matrices."""
    errors_list = []
    metrics = {}
    metrics["AUROC=1"] = base_modelers.compute_metrics_for_binary_outcome(
        fabricate_forecasts["AUROC=1"][0], fabricate_forecasts["AUROC=1"][1]
    )
    metrics["AUROC=0"] = base_modelers.compute_metrics_for_binary_outcome(
        fabricate_forecasts["AUROC=0"][0], fabricate_forecasts["AUROC=0"][1]
    )
    metrics["empty actual"] = base_modelers.compute_metrics_for_binary_outcome(
        fabricate_forecasts["empty actual"][0], fabricate_forecasts["empty actual"][1]
    )
    metrics[
        "AUROC=1, threshold_positive=1"
    ] = base_modelers.compute_metrics_for_binary_outcome(
        fabricate_forecasts["AUROC=1"][0],
        fabricate_forecasts["AUROC=1"][1],
        threshold_positive=1.0,
    )
    if not metrics["AUROC=1"]["AUROC"] == 1.0:
        errors_list.append(f"Condition 1 failed for AUROC=1.")
    if not metrics["AUROC=0"]["AUROC"] == 0.0:
        errors_list.append(f"Condition 2 failed for AUROC=0.")
    if not np.isnan(metrics["empty actual"]["AUROC"]):
        errors_list.append(f"Condition 3 failed for empty actual.")
    if not metrics["AUROC=1, threshold_positive=1"]["True Positives"] == 0:
        errors_list.append(f"Condition 4 failed for True Positives.")
    if not metrics["AUROC=1, threshold_positive=1"]["False Negatives"] == 4:
        errors_list.append(f"Condition 5 failed for False Negatives.")
    if not metrics["AUROC=1, threshold_positive=1"]["True Negatives"] == 5:
        errors_list.append(f"Condition 6 failed for True Negatives.")
    if not metrics["AUROC=1, threshold_positive=1"]["False Positives"] == 0:
        errors_list.append(f"Condition 7 failed for False Positives.")
    assert not errors_list, "Errors occurred: \n{}".format("\n".join(errors_list))


def test_compute_metrics_for_categorical_outcome():
    """Test that FIFE produces correct example AUROC and confusion matrices."""
    errors_list = []
    metrics = {}
    actuals = pd.DataFrame(
        [
            [False, True, False],
            [False, False, True],
            [False, False, True],
            [False, True, False],
        ]
    )
    predictions = np.array(
        [[0.1, 0.7, 0.2], [0.01, 0.6, 0.39], [0.01, 0.4, 0.59], [0.0, 0.6, 0.4]]
    )
    metrics[
        "one negative class"
    ] = base_modelers.compute_metrics_for_categorical_outcome(actuals, predictions)
    metrics[
        "positive weights for correct only"
    ] = base_modelers.compute_metrics_for_categorical_outcome(actuals, predictions, weights=np.array[1, 0, 1, 1])
    actuals = pd.DataFrame(
        [
            [False, True, False],
            [False, True, False],
            [False, True, False],
            [False, True, False],
        ]
    )
    metrics[
        "one positive class"
    ] = base_modelers.compute_metrics_for_categorical_outcome(actuals, predictions)
    if not isinstance(metrics["one negative class"]["AUROC"], float):
        errors_list.append(f"AUROC not a float with one of three classes entirely negative.")
    if metrics["positive weights for correct only"]["AUROC"] != 1:
        errors_list.append(f"AUROC not equal to 1 with only correct classifications having positive weight.")
    if not np.isnan(metrics["one positive class"]["AUROC"]):
        errors_list.append(
            f"Returns an AUROC of {metrics['one positive class']['AUROC']} instead of np.nan if all actual outcome values are identical."
        )
    assert not errors_list, "Errors occurred: \n{}".format("\n".join(errors_list))


SEED = 9999
np.random.seed(SEED)
