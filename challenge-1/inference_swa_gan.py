"""
Inference Script — MRI Low-to-High Field Enhancement
=====================================================
Encoding matches extract_slices.py EXACTLY:
  - uint8 normalization per slice (stores min/max for reconstruction)
  - np.savez_compressed into a BytesIO buffer → base64
  - Column name: 'prediction' (NOT 'data')
  - row_id format: "sample_XXX_slice_YYY" (NOT "sample_XXX_000")

Fixes:
  1. Divisible-by-32: dynamic pad → model → crop back to original H×W
  2. Edge slices: Z-reflect-pad so slices 0 and 199 have valid 2.5D context
  3. Full volume predicted first, then passed to create_submission_df

Usage:
    python inference.py \
        --model_path checkpoints/best_G_ep26_1.3254.pth \
        --data_dir /scratch/fr2471/ni_data \
        --split test \
        --output_csv submission.csv
"""

import os
import io
import glob
import base64
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom
import segmentation_models_pytorch as smp
from tqdm import tqdm

# Use the competition's official utility — must be in the same directory
from extract_slices import create_submission_df


# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
ENCODER  = 'efficientnet-b4'
TARGET_H = 179
TARGET_W = 221
N_SLICES = 200


# ─────────────────────────────────────────────
#  MODEL LOADER
# ─────────────────────────────────────────────
def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation='sigmoid',
        decoder_attention_type='scse',
    ).to(device)

    state = torch.load(model_path, map_location=device)

    # Strip common wrapper prefixes
    for prefix in ('module.', '_orig_mod.'):
        if any(k.startswith(prefix) for k in state.keys()):
            state = {k.replace(prefix, ''): v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"Loaded: {model_path}")
    return model


# ─────────────────────────────────────────────
#  VOLUME HELPERS
# ─────────────────────────────────────────────
def normalise(vol: np.ndarray) -> np.ndarray:
    """Per-volume min-max normalisation to [0, 1]."""
    vol = vol.astype(np.float32)
    return (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)


def trilinear_upsample(lf_vol: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Upsample LF volume to target_shape with trilinear (order=1, no ringing)."""
    factors = tuple(t / s for t, s in zip(target_shape, lf_vol.shape))
    return zoom(lf_vol, factors, order=1).astype(np.float32)


def pad_volume_z(vol: np.ndarray) -> np.ndarray:
    """
    Reflect-pad one slice on each end along Z.
    (H, W, D) → (H, W, D+2)
    Lets the 2.5D window cover original slices 0 and D-1 without OOB.
    """
    return np.concatenate([vol[:, :, :1], vol, vol[:, :, -1:]], axis=2)


# ─────────────────────────────────────────────
#  PAD / CROP FOR DIVISIBILITY BY 32
# ─────────────────────────────────────────────
def pad_for_model(tensor: torch.Tensor):
    """
    Pad (1, C, H, W) so H and W are divisible by 32.
    Reflect padding avoids hard black edges that confuse the decoder.
    Returns (padded_tensor, (pad_h, pad_w)).
    """
    _, _, h, w = tensor.shape
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
    return padded, (pad_h, pad_w)


def crop_output(tensor: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    h = tensor.shape[2] - pad_h
    w = tensor.shape[3] - pad_w
    return tensor[:, :, :h, :w]


# ─────────────────────────────────────────────
#  PER-VOLUME INFERENCE
# ─────────────────────────────────────────────
@torch.no_grad()
def predict_volume(
    model:    torch.nn.Module,
    lf_path:  str,
    hf_shape: tuple,
    device:   torch.device,
) -> np.ndarray:
    """
    Returns float32 predicted volume of shape (H, W, N_SLICES).
    All 200 slices covered via Z-reflect padding.
    Values are in [0, 1] — extract_slices handles uint8 conversion internally.
    """
    lf_vol    = normalise(nib.load(lf_path).get_fdata())
    lf_up     = trilinear_upsample(lf_vol, hf_shape)   # (H, W, D)
    lf_padded = pad_volume_z(lf_up)                    # (H, W, D+2)

    H, W      = lf_up.shape[:2]
    pred_vol  = np.zeros((H, W, N_SLICES), dtype=np.float32)

    for orig_idx in range(N_SLICES):
        padded_idx = orig_idx + 1

        stack  = lf_padded[:, :, padded_idx - 1: padded_idx + 2]   # (H, W, 3)
        tensor = (torch.from_numpy(stack)
                  .permute(2, 0, 1)
                  .unsqueeze(0)
                  .float()
                  .to(device))                                        # (1, 3, H, W)

        padded_t, (ph, pw) = pad_for_model(tensor)
        with torch.autocast('cuda', dtype=torch.float16,
                            enabled=(device.type == 'cuda')):
            out = model(padded_t)
        out_cropped = crop_output(out, ph, pw)                        # (1, 1, H, W)

        pred_vol[:, :, orig_idx] = np.clip(
            out_cropped[0, 0].cpu().float().numpy(), 0.0, 1.0
        )

    return pred_vol   # (179, 221, 200)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_dir',   type=str,
                        default='/scratch/fr2471/ni_data')
    parser.add_argument('--split',      type=str, default='test',
                        choices=['test', 'train'])
    parser.add_argument('--output_csv', type=str, default='submission.csv')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    if device.type == 'cuda':
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    model = load_model(args.model_path, device)

    lf_dir   = os.path.join(args.data_dir, args.split, 'low_field')
    lf_files = sorted(glob.glob(os.path.join(lf_dir, '*.nii*')))
    if not lf_files:
        raise FileNotFoundError(f"No .nii files in {lf_dir}")
    print(f"Found {len(lf_files)} test volumes\n")

    predictions_dict = {}

    for lf_path in tqdm(lf_files, desc='Volumes'):
        fname = os.path.basename(lf_path)

        # Extract sample_id: "sample_019_lowfield.nii.gz" → "sample_019"
        # This must match exactly what create_submission_df uses for row_id
        sample_id = fname.replace('.nii.gz', '').replace('.nii', '')
        sample_id = sample_id.replace('_lowfield', '')   # → "sample_019"

        # Determine HF shape dynamically per file
        hf_path = os.path.join(
            args.data_dir, args.split, 'high_field',
            fname.replace('lowfield', 'highfield')
        )
        if os.path.exists(hf_path):
            hf_shape = tuple(nib.load(hf_path).shape)
            print(f"  {sample_id}: HF shape = {hf_shape}")
        else:
            hf_shape = (TARGET_H, TARGET_W, N_SLICES)
            print(f"  {sample_id}: HF not found — using default {hf_shape}")

        pred_vol = predict_volume(model, lf_path, hf_shape, device)

        # Sanity check shape before storing
        assert pred_vol.shape == (hf_shape[0], hf_shape[1], N_SLICES), \
            f"Shape mismatch: got {pred_vol.shape}, expected ({hf_shape[0]}, {hf_shape[1]}, {N_SLICES})"

        predictions_dict[sample_id] = pred_vol
        print(f"  {sample_id}: predicted {pred_vol.shape}, "
              f"range [{pred_vol.min():.3f}, {pred_vol.max():.3f}]")

    # create_submission_df from extract_slices handles:
    #   - per-slice uint8 normalisation + min/max storage
    #   - np.savez_compressed → base64
    #   - row_id = "sample_XXX_slice_YYY"
    #   - column name = "prediction"
    print("\nBuilding submission with extract_slices.create_submission_df ...")
    df = create_submission_df(predictions_dict)

    df.to_csv(args.output_csv, index=False)
    size_mb = os.path.getsize(args.output_csv) / 1e6
    print(f"\nSaved: {args.output_csv}")
    print(f"Rows  : {len(df)}")
    print(f"Size  : {size_mb:.1f} MB")
    print(f"Cols  : {list(df.columns)}")
    print(f"\nSample rows:")
    print(df.head(3).to_string())


if __name__ == '__main__':
    main()