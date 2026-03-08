"""
3D Inference Script — SwinUNETR
================================
Runs full-volume sliding window inference and produces submission.csv
"""

import os
import glob
import numpy as np
import nibabel as nib
import torch
from scipy.ndimage import zoom
from tqdm import tqdm

from monai.networks.nets.swin_unetr import SwinUNETR
from monai.inferers.utils import sliding_window_inference

from extract_slices import create_submission_df

# ─────────────────────────────────────────────
# CONFIGURATION  (must match training)
# ─────────────────────────────────────────────
CFG = {
    'test_dir':      './test/low_field',
    'model_path':    'checkpoints_3d/best_msssim_0.27769.pth',
    'target_shape':  (179, 221, 200),
    'patch_size':    (96, 96, 96),
    'sw_overlap':    0.5,
    'sw_batch_size': 4,
    'device':        'cuda' if torch.cuda.is_available() else 'cpu',
}

print(f"Running 3D Inference on: {CFG['device'].upper()}")

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
model = SwinUNETR(
    in_channels=1,
    out_channels=1,
    feature_size=48,
    use_checkpoint=False,   # Not needed at inference
).to(CFG['device'])

print(f"Loading weights from {CFG['model_path']} ...")
state_dict = torch.load(CFG['model_path'], map_location=CFG['device'])
model.load_state_dict(state_dict)
model.eval()
print("Model loaded.")

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def load_and_preprocess(path, target=(179, 221, 200)):
    vol = nib.load(path).get_fdata().astype(np.float32)
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    if vol.shape != target:
        vol = zoom(
            vol,
            (target[0]/vol.shape[0], target[1]/vol.shape[1], target[2]/vol.shape[2]),
            order=3
        ).astype(np.float32)
    return vol

# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
test_files = sorted(glob.glob(os.path.join(CFG['test_dir'], '*.nii*')))
if not test_files:
    raise FileNotFoundError(f"No .nii files found in {CFG['test_dir']}")

print(f"Found {len(test_files)} test volumes.")

predictions_dict = {}

with torch.no_grad():
    for fpath in tqdm(test_files, desc="Predicting"):
        fname      = os.path.basename(fpath)
        sample_key = fname.split('_lowfield')[0]

        lf = load_and_preprocess(fpath, CFG['target_shape'])

        # (1, 1, D, H, W)
        lf_t = torch.from_numpy(lf).unsqueeze(0).unsqueeze(0).to(CFG['device'])

        pred = sliding_window_inference(
            inputs=lf_t,
            roi_size=CFG['patch_size'],
            sw_batch_size=CFG['sw_batch_size'],
            predictor=model,
            overlap=CFG['sw_overlap'],
            mode='gaussian',        # Gaussian blending at patch edges → no grid artefacts
        )

        pred_vol = torch.sigmoid(pred)[0, 0].cpu().numpy()
        pred_vol = np.clip(pred_vol, 0.0, 1.0)

        predictions_dict[sample_key] = pred_vol

# ─────────────────────────────────────────────
# SUBMISSION
# ─────────────────────────────────────────────
print("Creating submission...")
submission_df = create_submission_df(predictions_dict)
submission_df.to_csv('submission_swin3d.csv', index=False)
print(f"Done! submission.csv — {len(submission_df)} rows.")