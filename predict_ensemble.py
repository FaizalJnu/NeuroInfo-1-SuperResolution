import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import DataLoader, Dataset
from scipy.ndimage import zoom
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import cv2
from extract_slices import create_submission_df

# --- CONFIGURATION ---
CFG = {
    'batch_size': 32,
    'img_size': 256,
    'orig_shape': (179, 221),
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'n_folds': 5 
}

# --- DATASET (Same as before) ---
class InferenceDataset(Dataset):
    def __init__(self, lf_vol):
        self.vol = lf_vol
        self.transform = A.Compose([
            A.PadIfNeeded(min_height=CFG['img_size'], min_width=CFG['img_size'], 
                          border_mode=cv2.BORDER_CONSTANT, value=0),
            ToTensorV2()
        ])

    def __len__(self):
        return self.vol.shape[2]

    def __getitem__(self, idx):
        prev_idx = max(0, idx - 1)
        next_idx = min(self.vol.shape[2] - 1, idx + 1)
        img_stack = np.dstack([self.vol[:, :, prev_idx], self.vol[:, :, idx], self.vol[:, :, next_idx]])
        augmented = self.transform(image=img_stack)
        return augmented['image'], idx

def upsample_volume(vol):
    z_factor = 179 / vol.shape[0]
    y_factor = 221 / vol.shape[1]
    x_factor = 200 / vol.shape[2]
    return zoom(vol, (z_factor, y_factor, x_factor), order=3)

# --- ENSEMBLE PREDICTION ---
def predict_volume_ensemble(models, vol_path):
    vol = nib.load(vol_path).get_fdata()
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    vol = vol.astype(np.float32)
    
    print(f"Upsampling {os.path.basename(vol_path)}...")
    vol_upsampled = upsample_volume(vol)
    
    dataset = InferenceDataset(vol_upsampled)
    loader = DataLoader(dataset, batch_size=CFG['batch_size'], shuffle=False, num_workers=4)
    
    pad_h = (CFG['img_size'] - CFG['orig_shape'][0]) // 2
    pad_w = (CFG['img_size'] - CFG['orig_shape'][1]) // 2
    
    # Store predictions for the whole volume
    volume_preds = []

    with torch.no_grad():
        for batch_imgs, _ in tqdm(loader, desc="Ensembling"):
            batch_imgs = batch_imgs.to(CFG['device'])
            
            # Aggregate predictions from all 5 models
            batch_ensemble = 0
            for model in models:
                model.eval()
                # shape: (B, 1, 256, 256)
                preds = model(batch_imgs)
                batch_ensemble += preds.cpu().numpy()
            
            # Average them
            batch_ensemble /= len(models)
            
            # Post-process batch
            for i in range(batch_ensemble.shape[0]):
                p = batch_ensemble[i, 0, :, :]
                p_cropped = p[pad_h : pad_h + 179, pad_w : pad_w + 221]
                p_cropped = np.clip(p_cropped, 0, 1)
                volume_preds.append(p_cropped)
                
    return np.stack(volume_preds, axis=2)

def main():
    # Load all 5 models into memory (fits in 12GB VRAM easily for inference)
    models = []
    print("Loading Ensemble Models...")
    for fold in range(CFG['n_folds']):
        model_path = f"unet_fold{fold}.pth"
        if not os.path.exists(model_path):
            print(f"Warning: {model_path} not found. Skipping.")
            continue
            
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation='sigmoid'
        ).to(CFG['device'])
        model.load_state_dict(torch.load(model_path))
        models.append(model)
    
    print(f"Loaded {len(models)} models.")
    
    test_files = sorted(glob.glob(os.path.join('test', 'low_field', '*.nii*')))
    predictions_dict = {}
    
    for fpath in test_files:
        fname = os.path.basename(fpath)
        sample_key = fname.split('_lowfield')[0]
        
        pred_vol = predict_volume_ensemble(models, fpath)
        predictions_dict[sample_key] = pred_vol
            
    print("Generating submission file...")
    submission_df = create_submission_df(predictions_dict)
    submission_df.to_csv('submission_ensemble.csv', index=False)
    print("Done! Saved to submission_ensemble.csv")

if __name__ == '__main__':
    main()