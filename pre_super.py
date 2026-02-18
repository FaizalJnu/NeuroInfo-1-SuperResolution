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
    'batch_size': 16,        # Lowered to 16 because EfficientNet-B4 is VRAM heavy
    'img_size': 256,
    'orig_shape': (179, 221),
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'input_depth': 5,        # 5 Slices (z-2 to z+2)
    # CHANGE THIS TO YOUR SAVED MODEL FILENAME
    'model_path': 'supergan_epoch25.pth' 
}

class InferenceDataset(Dataset):
    def __init__(self, lf_vol):
        self.vol = lf_vol
        self.transform = A.Compose([
            A.PadIfNeeded(min_height=CFG['img_size'], min_width=CFG['img_size'], 
                          border_mode=cv2.BORDER_CONSTANT, value=0), #type: ignore
            ToTensorV2()
        ]) #type: ignore

    def __len__(self):
        return self.vol.shape[2]

    def __getitem__(self, idx):
        # We need 5 slices: [z-2, z-1, z, z+1, z+2]
        # We clamp indices to stay within bounds [0, max_z]
        max_z = self.vol.shape[2] - 1
        
        indices = [
            max(0, idx - 2),
            max(0, idx - 1),
            idx,
            min(max_z, idx + 1),
            min(max_z, idx + 2)
        ]
        
        # Stack slices
        slices = [self.vol[:, :, i] for i in indices]
        img_stack = np.dstack(slices) # Shape: (H, W, 5)
        
        augmented = self.transform(image=img_stack)
        return augmented['image'], idx

def upsample_volume(vol):
    z_factor = 179 / vol.shape[0]
    y_factor = 221 / vol.shape[1]
    x_factor = 200 / vol.shape[2]
    return zoom(vol, (z_factor, y_factor, x_factor), order=3)

# --- TTA PREDICTION ---
def predict_batch_tta(model, batch_imgs):
    """
    TTA for 5-channel input.
    Flips apply to Height (dim 2) and Width (dim 3).
    Channels (dim 1) are preserved.
    """
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
    
    # Average
    return (pred1 + pred2 + pred3) / 3.0

def predict_volume(model, vol_path):
    vol = nib.load(vol_path).get_fdata() #type: ignore
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    vol = vol.astype(np.float32)
    
    print(f"Upsampling {os.path.basename(vol_path)}...")
    vol_upsampled = upsample_volume(vol)
    
    dataset = InferenceDataset(vol_upsampled)
    # Using 0 workers for Windows stability, increase to 4 on Linux
    loader = DataLoader(dataset, batch_size=CFG['batch_size'], shuffle=False, num_workers=0)
    
    model.eval()
    predictions = []
    
    pad_h = (CFG['img_size'] - CFG['orig_shape'][0]) // 2
    pad_w = (CFG['img_size'] - CFG['orig_shape'][1]) // 2
    
    with torch.no_grad():
        for batch_imgs, _ in tqdm(loader, desc="SuperGAN Inferring"):
            batch_imgs = batch_imgs.to(CFG['device'])
            
            # Predict with TTA
            preds = predict_batch_tta(model, batch_imgs)
            preds = preds.cpu().numpy()
            
            for i in range(preds.shape[0]):
                p = preds[i, 0, :, :]
                # Crop back to 179x221
                p_cropped = p[pad_h : pad_h + 179, pad_w : pad_w + 221]
                # Clip safe range
                p_cropped = np.clip(p_cropped, 0, 1)
                predictions.append(p_cropped)
                
    return np.stack(predictions, axis=2)

def main():
    print(f"Loading Super-GAN (U-Net++ B4): {CFG['model_path']}")
    
    # DEFINITION MUST MATCH TRAIN SCRIPT EXACTLY
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",
        encoder_weights=None,
        in_channels=CFG['input_depth'], # 5
        classes=1,
        activation='sigmoid'
    ).to(CFG['device'])
    
    if not os.path.exists(CFG['model_path']):
        print(f"Error: Model file {CFG['model_path']} not found!")
        print("Please train using train_supergan.py first.")
        return

    model.load_state_dict(torch.load(CFG['model_path']))
    
    test_files = sorted(glob.glob(os.path.join('test', 'low_field', '*.nii*')))
    predictions_dict = {}
    
    for fpath in test_files:
        fname = os.path.basename(fpath)
        sample_key = fname.split('_lowfield')[0]
        
        # Run prediction
        pred_vol = predict_volume(model, fpath)
        predictions_dict[sample_key] = pred_vol
            
    print("Generating Super-GAN submission...")
    submission_df = create_submission_df(predictions_dict)
    
    submission_df.to_csv('submission_supergan.csv', index=False)
    print("Done! Saved to submission_supergan.csv")

if __name__ == '__main__':
    main()