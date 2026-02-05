"""
Slice Extraction Utility for Low-Field to High-Field MRI Super-Resolution

This script provides standardized functions for:
1. Extracting all 200 axial slices from a 3D MRI volume
2. Encoding slices to base64 for CSV submission
3. Decoding base64 back to numpy arrays
4. Creating submission rows in the correct format

IMPORTANT: Use these functions to ensure your submissions match the expected format!

Submission format:
  - Each row represents one slice
  - row_id format: "sample_XXX_slice_YYY" (e.g., "sample_019_slice_100")
  - prediction: base64-encoded 2D slice (179 x 221 pixels)
"""

import numpy as np
import nibabel as nib
import base64
import io
import pandas as pd

# All 200 slices are used
NUM_SLICES = 200


def load_nifti(path):
    """
    Load a NIfTI file and return the data array.

    Args:
        path: Path to .nii.gz file

    Returns:
        3D numpy array (x, y, z)
    """
    img = nib.load(path)
    return img.get_fdata()


def slice_to_base64(slice_2d):
    """
    Encode a 2D slice to base64 string for CSV submission.

    The slice is normalized to uint8 (0-255) and compressed.
    Original value range is stored to allow reconstruction.

    Args:
        slice_2d: 2D numpy array

    Returns:
        Base64 encoded string
    """
    slice_min = float(slice_2d.min())
    slice_max = float(slice_2d.max())

    if slice_max - slice_min > 0:
        normalized = ((slice_2d - slice_min) / (slice_max - slice_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(slice_2d, dtype=np.uint8)

    buffer = io.BytesIO()
    np.savez_compressed(buffer,
                        slice=normalized,
                        shape=np.array(slice_2d.shape),
                        min_val=np.array([slice_min]),
                        max_val=np.array([slice_max]))
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode('utf-8')


def base64_to_slice(b64_string):
    """
    Decode a base64 string back to a 2D numpy array.

    Args:
        b64_string: Base64 encoded string from slice_to_base64

    Returns:
        2D numpy array (float32)
    """
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


def volume_to_submission_rows(volume, sample_id):
    """
    Convert a 3D volume to submission rows.

    Extracts all 200 axial slices and encodes them to base64.

    Args:
        volume: 3D numpy array with shape (179, 221, 200)
        sample_id: Sample identifier (e.g., "sample_019")

    Returns:
        List of dictionaries with keys: row_id, prediction
    """
    if volume.shape[2] != NUM_SLICES:
        raise ValueError(f"Volume must have {NUM_SLICES} slices in z-dimension, got {volume.shape[2]}")

    rows = []
    for slice_idx in range(NUM_SLICES):
        row_id = f"{sample_id}_slice_{slice_idx:03d}"
        slice_2d = volume[:, :, slice_idx]
        b64 = slice_to_base64(slice_2d)
        rows.append({"row_id": row_id, "prediction": b64})

    return rows


def create_submission_df(predictions_dict):
    """
    Create a submission DataFrame from a dictionary of predictions.

    Args:
        predictions_dict: Dictionary mapping sample_id to 3D volume
                         e.g., {"sample_019": volume_019, "sample_020": volume_020, ...}

    Returns:
        pandas DataFrame ready for submission
    """
    all_rows = []
    for sample_id, volume in predictions_dict.items():
        rows = volume_to_submission_rows(volume, sample_id)
        all_rows.extend(rows)

    return pd.DataFrame(all_rows)


def nifti_to_submission_rows(nifti_path, sample_id):
    """
    Load a NIfTI file and convert to submission rows.

    Args:
        nifti_path: Path to .nii.gz file
        sample_id: Sample identifier (e.g., "sample_019")

    Returns:
        List of dictionaries with keys: row_id, prediction
    """
    volume = load_nifti(nifti_path)
    return volume_to_submission_rows(volume, sample_id)


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Slice Extraction Utility for MRI Super-Resolution Competition")
        print("=" * 60)
        print("\nUsage: python extract_slices.py <path_to_nifti.nii.gz>")
        print("\nExample:")
        print("  python extract_slices.py train/high_field/sample_001_highfield.nii.gz")
        print("\nThis will extract all 200 slices and show encoding info.")
        print("\n" + "=" * 60)
        print("\nSubmission Format:")
        print("  - Each row represents one slice")
        print("  - row_id: sample_XXX_slice_YYY (e.g., sample_019_slice_100)")
        print("  - prediction: base64-encoded slice")
        print("\nExample submission CSV:")
        print("  row_id,prediction")
        print("  sample_019_slice_000,<base64>")
        print("  sample_019_slice_001,<base64>")
        print("  ...")
        print("  sample_023_slice_199,<base64>")
        print("\nTotal rows for 5 test samples: 5 × 200 = 1000 rows")
        sys.exit(0)

    nifti_path = sys.argv[1]

    # Load volume
    volume = load_nifti(nifti_path)
    print(f"Loaded volume shape: {volume.shape}")
    print(f"Volume depth (z): {volume.shape[2]}")

    if volume.shape[2] != NUM_SLICES:
        print(f"\nWarning: Expected {NUM_SLICES} slices, got {volume.shape[2]}")

    # Extract and encode a few sample slices
    print(f"\nExtracting {NUM_SLICES} slices...")
    sample_slices = [0, 50, 100, 150, 199]

    for idx in sample_slices:
        if idx < volume.shape[2]:
            slice_2d = volume[:, :, idx]
            b64 = slice_to_base64(slice_2d)
            print(f"  Slice {idx:03d}: shape={slice_2d.shape}, base64 length={len(b64)}")

    # Verify round-trip for center slice
    print("\nVerifying round-trip encoding for center slice (100)...")
    if volume.shape[2] > 100:
        original = volume[:, :, 100]
        b64 = slice_to_base64(original)
        recovered = base64_to_slice(b64)
        mse = np.mean((original - recovered) ** 2)
        print(f"  Round-trip MSE: {mse:.6f}")

    # Show example submission creation
    print("\n" + "=" * 60)
    print("Example: Creating submission for a single sample")
    print("=" * 60)
    print("""
# Your code:
from extract_slices import create_submission_df, volume_to_submission_rows
import pandas as pd

# Option 1: Create rows for one sample
predicted_volume = your_model(low_field_input)  # Shape: (179, 221, 200)
rows = volume_to_submission_rows(predicted_volume, 'sample_019')
# rows is a list of 200 dicts: [{'row_id': 'sample_019_slice_000', 'prediction': '...'}, ...]

# Option 2: Create full submission DataFrame
predictions = {
    'sample_019': model(load_nifti('test/low_field/sample_019_lowfield.nii.gz')),
    'sample_020': model(load_nifti('test/low_field/sample_020_lowfield.nii.gz')),
    'sample_021': model(load_nifti('test/low_field/sample_021_lowfield.nii.gz')),
    'sample_022': model(load_nifti('test/low_field/sample_022_lowfield.nii.gz')),
    'sample_023': model(load_nifti('test/low_field/sample_023_lowfield.nii.gz')),
}
submission_df = create_submission_df(predictions)
submission_df.to_csv('submission.csv', index=False)
""")
