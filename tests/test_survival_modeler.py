"""Conduct unit testing for fife.survival_modeler module."""

from fife import survival_modeler
import numpy as np


def test_compute_metrics_for_binary_outcome(fabricate_forecasts):
    """Test that FIFE produces correct example AUROC and confusion matrices."""
    errors_list = []
    metrics = {}
    metrics["AUROC=1"] = survival_modeler.compute_metrics_for_binary_outcome(
        fabricate_forecasts["AUROC=1"][0], fabricate_forecasts["AUROC=1"][1]
    )
    metrics["AUROC=0"] = survival_modeler.compute_metrics_for_binary_outcome(
        fabricate_forecasts["AUROC=0"][0], fabricate_forecasts["AUROC=0"][1]
    )
    metrics["empty actual"] = survival_modeler.compute_metrics_for_binary_outcome(
        fabricate_forecasts["empty actual"][0], fabricate_forecasts["empty actual"][1]
    )
    metrics[
        "AUROC=1, threshold_positive=1"
    ] = survival_modeler.compute_metrics_for_binary_outcome(
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


SEED = 9999
np.random.seed(SEED)
