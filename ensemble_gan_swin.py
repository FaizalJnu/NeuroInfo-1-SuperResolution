import os
import glob
import cv2
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import zoom

# GAN Imports
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

# 3D Imports
from monai.networks.nets.swin_unetr import SwinUNETR
from monai.inferers.utils import sliding_window_inference
from extract_slices import create_submission_df
# (Ensure you import or define your create_submission_df function here)
# from utils import create_submission_df 

CFG = {
    'test_dir': './test/low_field',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Ensemble Settings
    'gan_weight': 0.5,
    'swin_weight': 0.5,
    
    # GAN Config
    'gan_model_path': 'NewGan_epoch_best_0.0490.pth',
    'batch_size': 32,
    'img_size': 256,
    'orig_shape': (179, 221),
    
    # SwinUNETR Config
    'swin_model_path': 'checkpoints_3d/best_msssim_0.27769.pth',
    'target_shape': (179, 221, 200),
    'patch_size': (96, 96, 96),
    'sw_overlap': 0.5,
    'sw_batch_size': 4,
}

# ─────────────────────────────────────────────
# 1. GAN DATASET & HELPERS
# ─────────────────────────────────────────────
class InferenceDataset(Dataset):
    def __init__(self, lf_vol):
        self.vol = lf_vol
        self.transform = A.Compose([
            A.PadIfNeeded(min_height=CFG['img_size'], min_width=CFG['img_size'], 
                          border_mode=cv2.BORDER_CONSTANT, value=0),
            ToTensorV2()
        ]) #type:ignore

    def __len__(self):
        return self.vol.shape[2]

    def __getitem__(self, idx):
        prev_idx = max(0, idx - 1)
        next_idx = min(self.vol.shape[2] - 1, idx + 1)
        img_stack = np.dstack([self.vol[:, :, prev_idx], self.vol[:, :, idx], self.vol[:, :, next_idx]])
        augmented = self.transform(image=img_stack)
        return augmented['image'], idx

def upsample_volume_gan(vol):
    z_factor = 179 / vol.shape[0]
    y_factor = 221 / vol.shape[1]
    x_factor = 200 / vol.shape[2]
    return zoom(vol, (z_factor, y_factor, x_factor), order=3)

def predict_batch_tta_gan(model, batch_imgs):
    # 1. Standard pass
    pred1 = model(batch_imgs)
    # 2. Horizontal Flip
    imgs_h = torch.flip(batch_imgs, [3])
    pred_h = model(imgs_h)
    pred2 = torch.flip(pred_h, [3])
    # 3. Vertical Flip
    imgs_v = torch.flip(batch_imgs, [2])
    pred_v = model(imgs_v)
    pred3 = torch.flip(pred_v, [2])
    
    return (pred1 + pred2 + pred3) / 3.0

def predict_gan_volume(model, vol, orig_shape=CFG['orig_shape']):
    vol_upsampled = upsample_volume_gan(vol)
    dataset = InferenceDataset(vol_upsampled)
    # Use num_workers=0 to prevent multiprocessing issues during a combined inference script
    loader = DataLoader(dataset, batch_size=CFG['batch_size'], shuffle=False, num_workers=0) 
    
    predictions = []
    pad_h = (CFG['img_size'] - orig_shape[0]) // 2
    pad_w = (CFG['img_size'] - orig_shape[1]) // 2
    
    with torch.no_grad():
        for batch_imgs, _ in loader:
            batch_imgs = batch_imgs.to(CFG['device'])
            preds = predict_batch_tta_gan(model, batch_imgs).cpu().numpy()
            
            for i in range(preds.shape[0]):
                p = preds[i, 0, :, :]
                p_cropped = p[pad_h : pad_h + orig_shape[0], pad_w : pad_w + orig_shape[1]]
                p_cropped = np.clip(p_cropped, 0, 1)
                predictions.append(p_cropped)
                
    return np.stack(predictions, axis=2)

# ─────────────────────────────────────────────
# 2. SWINUNETR DATASET & TTA HELPERS
# ─────────────────────────────────────────────
def upsample_volume_swin(vol, target=CFG['target_shape']):
    if vol.shape != target:
        return zoom(
            vol,
            (target[0]/vol.shape[0], target[1]/vol.shape[1], target[2]/vol.shape[2]),
            order=3
        ).astype(np.float32)
    return vol

def predict_swin_tta(model, lf_t):
    """
    Replicates the custom TTA logic using native PyTorch operations for speed.
    Assumes inputs are (1, 1, D, H, W).
    """
    preds = []
    
    def _infer(inputs):
        return sliding_window_inference(
            inputs=inputs,
            roi_size=CFG['patch_size'],
            sw_batch_size=CFG['sw_batch_size'],
            predictor=model,
            overlap=CFG['sw_overlap'],
            mode='gaussian'
        )
        
    # 1. Original
    preds.append(_infer(lf_t))
    
    # 2. Horizontal Flip (Dim 2)
    lf_t_h = torch.flip(lf_t, dims=[2])
    pred_h = _infer(lf_t_h)
    preds.append(torch.flip(pred_h, dims=[2]))
    
    # 3. Vertical Flip (Dim 3)
    lf_t_v = torch.flip(lf_t, dims=[3])
    pred_v = _infer(lf_t_v)
    preds.append(torch.flip(pred_v, dims=[3]))
    
    # 4. Both Flips (Dims 2 & 3)
    lf_t_hv = torch.flip(lf_t, dims=[2, 3])
    pred_hv = _infer(lf_t_hv)
    preds.append(torch.flip(pred_hv, dims=[2, 3]))
    
    # Average and apply sigmoid
    mean_pred = torch.mean(torch.stack(preds), dim=0)
    out_vol = torch.sigmoid(mean_pred)[0, 0].cpu().numpy()
    return np.clip(out_vol, 0.0, 1.0)

# ─────────────────────────────────────────────
# 3. MAIN ENSEMBLE LOOP
# ─────────────────────────────────────────────
def main():
    print(f"Running Ensemble Inference on: {CFG['device'].upper()}")
    
    # --- Load GAN ---
    print(f"Loading GAN: {CFG['gan_model_path']}")
    gan_model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation='sigmoid'
    ).to(CFG['device'])
    gan_model.load_state_dict(torch.load(CFG['gan_model_path'], map_location=CFG['device']))
    gan_model.eval()
    
    # --- Load SwinUNETR ---
    print(f"Loading SwinUNETR: {CFG['swin_model_path']}")
    swin_model = SwinUNETR(
        in_channels=1,
        out_channels=1,
        feature_size=48,
        use_checkpoint=False,
    ).to(CFG['device'])
    swin_model.load_state_dict(torch.load(CFG['swin_model_path'], map_location=CFG['device']))
    swin_model.eval()
    
    test_files = sorted(glob.glob(os.path.join(CFG['test_dir'], '*.nii*')))
    if not test_files:
        raise FileNotFoundError(f"No .nii files found in {CFG['test_dir']}")
    print(f"Found {len(test_files)} test volumes.")

    predictions_dict = {}

    with torch.no_grad():
        for fpath in tqdm(test_files, desc="Ensemble Predicting"):
            fname = os.path.basename(fpath)
            sample_key = fname.split('_lowfield')[0]

            # Base Load
            vol = nib.load(fpath).get_fdata()
            vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
            vol = vol.astype(np.float32)

            # --- 1. GAN Inference ---
            pred_gan = predict_gan_volume(gan_model, vol)
            
            # --- 2. SwinUNETR Inference ---
            lf_swin = upsample_volume_swin(vol)
            lf_t = torch.from_numpy(lf_swin).unsqueeze(0).unsqueeze(0).to(CFG['device'])
            pred_swin = predict_swin_tta(swin_model, lf_t)
            
            # --- 3. Blend ---
            pred_ensemble = (pred_gan * CFG['gan_weight']) + (pred_swin * CFG['swin_weight'])
            predictions_dict[sample_key] = pred_ensemble

    print("Creating submission...")
    # Make sure create_submission_df is defined or imported
    submission_df = create_submission_df(predictions_dict) 
    submission_df.to_csv('submission_ensemble_ganold_swin.csv', index=False)
    print(f"Done! Saved to submission_ensemble.csv — {len(submission_df)} rows.")

if __name__ == '__main__':
    main()