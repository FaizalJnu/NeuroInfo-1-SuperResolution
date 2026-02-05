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

# CRITICAL: Import the competition's provided utility
from extract_slices import create_submission_df, load_nifti

# --- CONFIGURATION ---
CFG = {
    'batch_size': 32,
    'img_size': 256,
    'orig_shape': (179, 221),
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model_path': 'unet_25d_epoch15.pth' # <--- Ensure this is correct
}

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

def predict_volume(model, vol_path):
    vol = load_nifti(vol_path)
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    vol = vol.astype(np.float32)
    
    print(f"Upsampling {os.path.basename(vol_path)}...")
    vol_upsampled = upsample_volume(vol)
    
    dataset = InferenceDataset(vol_upsampled)
    loader = DataLoader(dataset, batch_size=CFG['batch_size'], shuffle=False, num_workers=4)
    
    model.eval()
    predictions = []
    
    pad_h = (CFG['img_size'] - CFG['orig_shape'][0]) // 2
    pad_w = (CFG['img_size'] - CFG['orig_shape'][1]) // 2
    
    with torch.no_grad():
        for batch_imgs, _ in tqdm(loader, desc="Inferring"):
            batch_imgs = batch_imgs.to(CFG['device'])
            preds = model(batch_imgs).cpu().numpy()
            
            for i in range(preds.shape[0]):
                p = preds[i, 0, :, :]
                p_cropped = p[pad_h : pad_h + 179, pad_w : pad_w + 221]
                p_cropped = np.clip(p_cropped, 0, 1)
                predictions.append(p_cropped)
                
    # Stack back into (179, 221, 200)
    return np.stack(predictions, axis=2)

def main():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation='sigmoid'
    ).to(CFG['device'])
    
    if not os.path.exists(CFG['model_path']):
        print(f"Error: Model file {CFG['model_path']} not found!")
        return

    model.load_state_dict(torch.load(CFG['model_path']))
    test_files = sorted(glob.glob(os.path.join('test', 'low_field', '*.nii*')))
    
    # This dictionary will store our 3D volumes (numpy arrays)
    predictions_dict = {}
    
    for fpath in test_files:
        # Extract numeric ID, e.g., 'sample_019'
        fname = os.path.basename(fpath)
        sample_key = fname.split('_lowfield')[0] # result: 'sample_019'
        
        pred_vol = predict_volume(model, fpath)
        predictions_dict[sample_key] = pred_vol
            
    print("Converting volumes to submission format using competition script...")
    # This function handles the complex Base64 encoding for you
    submission_df = create_submission_df(predictions_dict)
    
    submission_df.to_csv('submission.csv', index=False)
    print("Done! Saved to submission.csv")

if __name__ == '__main__':
    main()