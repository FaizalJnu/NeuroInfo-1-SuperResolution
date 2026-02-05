"""
Low-Field to High-Field MRI Enhancement Challenge - Evaluation Metric

This metric evaluates predicted MRI slices against ground truth high-field images.
Each row represents a single slice: row_id format is "sample_XXX_slice_YYY"

The metric computes:
- SSIM (Structural Similarity Index): Measures perceptual similarity
- PSNR (Peak Signal-to-Noise Ratio): Measures reconstruction fidelity

Final score is a weighted combination: 0.5 * SSIM + 0.5 * (PSNR / 50)
Higher scores are better.
"""

import pandas as pd
import pandas.api.types
import numpy as np
import base64
import io


class ParticipantVisibleError(Exception):
    """Error messages shown to participants for debugging their submissions."""
    pass


def base64_to_slice(b64_string: str) -> np.ndarray:
    """
    Decode a base64 string back to a 2D numpy array.

    Args:
        b64_string: Base64 encoded string

    Returns:
        2D numpy array (float32)
    """
    try:
        buffer = io.BytesIO(base64.b64decode(b64_string))
        data = np.load(buffer)

        normalized = data['slice']
        min_val_arr = data['min_val']
        max_val_arr = data['max_val']

        # Handle both scalar and array storage formats
        min_val = float(min_val_arr.item()) if min_val_arr.ndim > 0 else float(min_val_arr)
        max_val = float(max_val_arr.item()) if max_val_arr.ndim > 0 else float(max_val_arr)

        if max_val - min_val > 0:
            original = normalized.astype(np.float32) / 255 * (max_val - min_val) + min_val
        else:
            original = np.zeros_like(normalized, dtype=np.float32)

        return original
    except Exception as e:
        raise ParticipantVisibleError(f"Failed to decode base64 slice: {str(e)}")


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Structural Similarity Index (SSIM) between two images.

    Args:
        img1, img2: 2D numpy arrays (must be same shape)

    Returns:
        SSIM value between 0 and 1
    """
    # Normalize to [0, 1]
    def normalize(x):
        x_min, x_max = x.min(), x.max()
        if x_max - x_min > 0:
            return (x - x_min) / (x_max - x_min)
        return np.zeros_like(x)

    img1_norm = normalize(img1)
    img2_norm = normalize(img2)

    # SSIM constants
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Compute means
    mu1 = img1_norm.mean()
    mu2 = img2_norm.mean()

    # Compute variances and covariance
    sigma1_sq = ((img1_norm - mu1) ** 2).mean()
    sigma2_sq = ((img2_norm - mu2) ** 2).mean()
    sigma12 = ((img1_norm - mu1) * (img2_norm - mu2)).mean()

    # SSIM formula
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)

    return float(numerator / denominator)


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        img1, img2: 2D numpy arrays (must be same shape)

    Returns:
        PSNR value in dB
    """
    # Normalize to [0, 1]
    def normalize(x):
        x_min, x_max = x.min(), x.max()
        if x_max - x_min > 0:
            return (x - x_min) / (x_max - x_min)
        return np.zeros_like(x)

    img1_norm = normalize(img1)
    img2_norm = normalize(img2)

    mse = ((img1_norm - img2_norm) ** 2).mean()

    if mse == 0:
        return 50.0  # Perfect match, cap at 50 dB

    psnr = 10 * np.log10(1.0 / mse)

    # Clamp to [0, 50] range
    return float(min(max(psnr, 0), 50))


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Evaluate MRI super-resolution predictions using SSIM and PSNR.

    Each row represents a single slice with format: row_id = "sample_XXX_slice_YYY"

    Final score = 0.5 * mean_SSIM + 0.5 * (mean_PSNR / 50)

    Higher scores are better. Maximum possible score is 1.0 (perfect prediction).

    Args:
        solution: DataFrame with columns [row_id, ground_truth, Usage]
        submission: DataFrame with columns [row_id, prediction]
        row_id_column_name: Name of the ID column (row_id)

    Returns:
        Combined score between 0 and 1
    """

    # ========================================
    # VALIDATION: Check submission structure
    # ========================================

    # Check if submission is empty
    if submission is None or len(submission) == 0:
        raise ParticipantVisibleError("Submission is empty. Please provide predictions for all test slices.")

    # Check if required columns exist
    if row_id_column_name not in submission.columns:
        raise ParticipantVisibleError(
            f"Submission must have a '{row_id_column_name}' column. "
            f"Found columns: {list(submission.columns)}"
        )

    if 'prediction' not in submission.columns:
        raise ParticipantVisibleError(
            f"Submission must have a 'prediction' column. "
            f"Found columns: {list(submission.columns)}"
        )

    if 'ground_truth' not in solution.columns:
        raise ParticipantVisibleError("Solution must have a 'ground_truth' column")

    # ========================================
    # VALIDATION: Check for duplicates
    # ========================================

    duplicate_ids = submission[submission[row_id_column_name].duplicated()][row_id_column_name].tolist()
    if duplicate_ids:
        raise ParticipantVisibleError(
            f"Submission contains duplicate row_ids: {duplicate_ids[:5]}{'...' if len(duplicate_ids) > 5 else ''}. "
            f"Each row_id should appear exactly once."
        )

    # ========================================
    # VALIDATION: Check for NaN/empty values
    # ========================================

    # Check for NaN in row_id column
    if submission[row_id_column_name].isna().any():
        raise ParticipantVisibleError(
            f"Submission contains NaN/empty values in '{row_id_column_name}' column. "
            f"All row_ids must be valid."
        )

    # Check for NaN in prediction column
    nan_predictions = submission[submission['prediction'].isna()][row_id_column_name].tolist()
    if nan_predictions:
        raise ParticipantVisibleError(
            f"Submission contains NaN/empty values in 'prediction' column for rows: {nan_predictions[:5]}{'...' if len(nan_predictions) > 5 else ''}. "
            f"All predictions must be valid base64-encoded slices."
        )

    # Check for empty strings in prediction
    empty_predictions = submission[submission['prediction'].astype(str).str.strip() == ''][row_id_column_name].tolist()
    if empty_predictions:
        raise ParticipantVisibleError(
            f"Submission contains empty strings in 'prediction' column for rows: {empty_predictions[:5]}{'...' if len(empty_predictions) > 5 else ''}. "
            f"All predictions must be valid base64-encoded slices."
        )

    # ========================================
    # MERGE: Align submission with solution
    # ========================================

    merged = solution.merge(submission, on=row_id_column_name, how='left')

    # Check for missing predictions after merge
    missing_mask = merged['prediction'].isna()
    if missing_mask.any():
        missing_ids = merged[missing_mask][row_id_column_name].tolist()
        raise ParticipantVisibleError(
            f"Missing predictions for {len(missing_ids)} rows: {missing_ids[:5]}{'...' if len(missing_ids) > 5 else ''}. "
            f"Your submission must include predictions for all test slices."
        )

    # ========================================
    # COMPUTE METRICS
    # ========================================

    all_ssim = []
    all_psnr = []

    for idx, row in merged.iterrows():
        row_id = row[row_id_column_name]

        # Decode ground truth
        try:
            gt_slice = base64_to_slice(row['ground_truth'])
        except ParticipantVisibleError:
            raise
        except Exception as e:
            raise ParticipantVisibleError(f"Error decoding solution for {row_id}: {str(e)}")

        # Decode prediction
        try:
            pred_slice = base64_to_slice(row['prediction'])
        except ParticipantVisibleError as e:
            raise ParticipantVisibleError(f"Error decoding prediction for {row_id}: {str(e)}")
        except Exception as e:
            raise ParticipantVisibleError(
                f"Error decoding prediction for {row_id}: {str(e)}. "
                f"Make sure you're using the provided slice_to_base64() function."
            )

        # Check shape match
        expected_shape = (179, 221)
        if pred_slice.shape != expected_shape:
            raise ParticipantVisibleError(
                f"Shape mismatch for {row_id}: expected {expected_shape}, got {pred_slice.shape}. "
                f"Your predicted slices must be exactly (179, 221) pixels."
            )

        if gt_slice.shape != pred_slice.shape:
            raise ParticipantVisibleError(
                f"Shape mismatch for {row_id}: ground truth is {gt_slice.shape}, prediction is {pred_slice.shape}."
            )

        # Check for NaN/Inf in prediction
        if np.isnan(pred_slice).any() or np.isinf(pred_slice).any():
            raise ParticipantVisibleError(
                f"Prediction for {row_id} contains NaN or Inf values. "
                f"All pixel values must be finite numbers."
            )

        # Compute metrics
        ssim = compute_ssim(gt_slice, pred_slice)
        psnr = compute_psnr(gt_slice, pred_slice)

        all_ssim.append(ssim)
        all_psnr.append(psnr)

    # ========================================
    # COMPUTE FINAL SCORE
    # ========================================

    mean_ssim = np.mean(all_ssim)
    mean_psnr = np.mean(all_psnr)

    # Combined score: 0.5 * SSIM + 0.5 * (PSNR / 50)
    final_score = 0.5 * mean_ssim + 0.5 * (mean_psnr / 50)

    # Ensure score is finite
    if not np.isfinite(final_score):
        raise ParticipantVisibleError("Computed score is not finite. Please check your predictions.")

    return float(final_score)
