import os
import glob
import numpy as np
import nibabel as nib
import torch
from scipy.ndimage import zoom
from tqdm import tqdm
import segmentation_models_pytorch as smp
import torch.nn.functional as F

# CRITICAL: Import the competition's provided utility
from extract_slices import create_submission_df

# --- CONFIGURATION ---
CFG = {
    'test_dir':    './test/low_field',
    'model_path':  'checkpoints/best_msssim.pth',   # best checkpoint from training
    'target_shape': (179, 221, 200),                  # must match training upsample target
    'device':      'cuda' if torch.cuda.is_available() else 'cpu',
}

print(f"Running Inference on: {CFG['device'].upper()}")


# ─────────────────────────────────────────────
# HELPERS  (mirror exactly what training does)
# ─────────────────────────────────────────────
def load_and_normalise(path):
    vol = nib.load(path).get_fdata().astype(np.float32)
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    return vol


def upsample_lf(vol, target=(179, 221, 200)):
    z = target[0] / vol.shape[0]
    y = target[1] / vol.shape[1]
    x = target[2] / vol.shape[2]
    return zoom(vol, (z, y, x), order=3).astype(np.float32)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
generator = smp.UnetPlusPlus(
    encoder_name='efficientnet-b6',
    encoder_weights=None,       # no imagenet weights needed at inference
    in_channels=3,              # 2.5D: 3 adjacent slices
    classes=1,
    activation='sigmoid',
).to(CFG['device'])

print(f"Loading weights from {CFG['model_path']} ...")
state_dict = torch.load(CFG['model_path'], map_location=CFG['device'])

# Handle torch.compile wrapping (keys prefixed with '_orig_mod.')
if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

generator.load_state_dict(state_dict)
generator.eval()
print("Model loaded successfully.")


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
test_files = sorted(glob.glob(os.path.join(CFG['test_dir'], '*.nii*')))
if not test_files:
    raise FileNotFoundError(f"No .nii files found in {CFG['test_dir']}")

print(f"Found {len(test_files)} test volumes.")

predictions_dict = {}

with torch.no_grad():
    for fpath in tqdm(test_files, desc="Predicting volumes"):
        fname      = os.path.basename(fpath)
        sample_key = fname.split('_lowfield')[0]

        # Load & preprocess — mirrors training pipeline exactly
        lf_vol = upsample_lf(load_and_normalise(fpath), CFG['target_shape'])
        # lf_vol shape: (179, 221, 200)

        D = lf_vol.shape[2]  # number of slices (200)
        pred_vol = np.zeros((lf_vol.shape[0], lf_vol.shape[1], D), dtype=np.float32)

        for s in range(1, D - 1):   # skip first and last (need neighbours)
            # 2.5D stack: 3 adjacent slices → (H, W, 3)
            stack = lf_vol[:, :, s - 1: s + 2]          
            
            # → (1, 3, H, W)
            tensor = torch.from_numpy(stack).permute(2, 0, 1).unsqueeze(0).to(CFG['device'])

            # --- DYNAMIC PADDING FIX ---
            _, _, h, w = tensor.shape
            pad_h = (32 - (h % 32)) % 32
            pad_w = (32 - (w % 32)) % 32
            
            # Pad format is (left, right, top, bottom)
            # We use 'reflect' padding to avoid hard black edges that might confuse the model
            padded_tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')

            # Predict on the padded tensor
            pred_padded = generator(padded_tensor)                       

            # --- REVERT THE PADDING (CROP) ---
            pred = pred_padded[:, :, :h, :w] 
            
            pred_slice = pred[0, 0].cpu().numpy()        # (H, W)
            pred_vol[:, :, s] = pred_slice

        # Handle edge slices (duplicate neighbour)
        pred_vol[:, :, 0]    = pred_vol[:, :, 1]
        pred_vol[:, :, D-1]  = pred_vol[:, :, D-2]

        # Clip to valid range for MS-SSIM
        pred_vol = np.clip(pred_vol, 0.0, 1.0)

        predictions_dict[sample_key] = pred_vol

# ─────────────────────────────────────────────
# CREATE SUBMISSION
# ─────────────────────────────────────────────
print("Converting volumes to submission format...")
submission_df = create_submission_df(predictions_dict)
submission_df.to_csv('submission_gan_new.csv', index=False)
print(f"Done! submission.csv saved — {len(submission_df)} rows.")