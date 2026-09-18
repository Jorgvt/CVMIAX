"""Evaluation suite comparing Classical Metrics (PSNR, SSIM) vs Deep IQA Models.

Computes benchmark correlations (SROCC, KROCC, PLCC, RMSE, MAE) overall and
broken down across distortion families.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

try:
    from .dataset import DISTORTION_NAMES, get_distortion_group
    from .metrics import (
        compute_psnr_batch,
        compute_ssim_batch,
        evaluate_iqa_predictions,
    )
except ImportError:
    from dataset import DISTORTION_NAMES, get_distortion_group
    from metrics import (
        compute_psnr_batch,
        compute_ssim_batch,
        evaluate_iqa_predictions,
    )



def evaluate_all_methods_on_test(
    test_refs: np.ndarray,
    test_dists: np.ndarray,
    test_mos: np.ndarray,
    test_dist_ids: np.ndarray,
    fr_model=None,
    nr_model=None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    """Run comprehensive quantitative evaluation across all models on test set.

    Returns:
        overall_df: DataFrame with benchmark metrics (SROCC, KROCC, PLCC, RMSE, MAE)
        distortion_df: DataFrame broken down by distortion group
        predictions_dict: Dictionary mapping model names to raw predicted metric arrays
    """
    print("Computing classical baselines (PSNR and SSIM)...")
    psnr_preds = compute_psnr_batch(test_refs, test_dists)
    ssim_preds = compute_ssim_batch(test_refs, test_dists)

    predictions = {
        "PSNR": psnr_preds,
        "SSIM": ssim_preds,
    }

    if fr_model is not None:
        print("Evaluating Full-Reference Deep CNN...")
        fr_preds = fr_model.predict(
            {"reference": test_refs, "distorted": test_dists}, verbose=0
        ).flatten()
        predictions["Deep FR-IQA"] = fr_preds

    if nr_model is not None:
        print("Evaluating No-Reference Deep CNN...")
        nr_preds = nr_model.predict(test_dists, verbose=0).flatten()
        predictions["Deep NR-IQA"] = nr_preds

    # 1. Overall Metrics
    results_list = []
    for model_name, preds in predictions.items():
        metrics = evaluate_iqa_predictions(
            test_mos, preds, model_name=model_name, apply_logistic_fit=True
        )
        results_list.append(metrics)

    overall_df = pd.DataFrame(results_list)

    # 2. Distortion Group Breakdown
    distortion_groups = [get_distortion_group(did) for did in test_dist_ids]
    unique_groups = sorted(list(set(distortion_groups)))

    dist_rows = []
    for group in unique_groups:
        mask = np.array([g == group for g in distortion_groups])
        if np.sum(mask) == 0:
            continue
        group_mos = test_mos[mask]

        for model_name, preds in predictions.items():
            group_preds = preds[mask]
            metrics = evaluate_iqa_predictions(
                group_mos,
                group_preds,
                model_name=model_name,
                apply_logistic_fit=True,
            )
            dist_rows.append(
                {
                    "Distortion Group": group,
                    "Model": model_name,
                    "SROCC": metrics["SROCC"],
                    "PLCC": metrics["PLCC"],
                    "RMSE": metrics["RMSE"],
                }
            )

    distortion_df = pd.DataFrame(dist_rows)

    return overall_df, distortion_df, predictions
