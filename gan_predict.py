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
    # POINT THIS TO YOUR NEW GAN WEIGHTS
    'model_path': 'gan_generator_epoch20.pth' 
}

class InferenceDataset(Dataset):
    def __init__(self, lf_vol):
        self.vol = lf_vol
        self.transform = A.Compose([
            A.PadIfNeeded(min_height=CFG['img_size'], min_width=CFG['img_size'], 
                          border_mode=cv2.BORDER_CONSTANT, value=0),
            ToTensorV2()
        ]) #type: ignore

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

# --- GAN PREDICTION (Includes TTA for max score) ---
def predict_batch_tta(model, batch_imgs):
    """
    Even with GANs, TTA (Test Time Augmentation) helps smooth out artifacts.
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
    
    return (pred1 + pred2 + pred3) / 3.0

def predict_volume(model, vol_path):
    vol = nib.load(vol_path).get_fdata()
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    vol = vol.astype(np.float32)
    
    print(f"Upsampling {os.path.basename(vol_path)}...")
    vol_upsampled = upsample_volume(vol)
    
    dataset = InferenceDataset(vol_upsampled)
    loader = DataLoader(dataset, batch_size=CFG['batch_size'], shuffle=False, num_workers=0) # 0 for Windows
    
    model.eval()
    predictions = []
    
    pad_h = (CFG['img_size'] - CFG['orig_shape'][0]) // 2
    pad_w = (CFG['img_size'] - CFG['orig_shape'][1]) // 2
    
    with torch.no_grad():
        for batch_imgs, _ in tqdm(loader, desc="GAN Inferring"):
            batch_imgs = batch_imgs.to(CFG['device'])
            
            # Predict using Generator with TTA
            preds = predict_batch_tta(model, batch_imgs)
            preds = preds.cpu().numpy()
            
            for i in range(preds.shape[0]):
                p = preds[i, 0, :, :]
                # Crop back to original size
                p_cropped = p[pad_h : pad_h + 179, pad_w : pad_w + 221]
                # Clip to safe range [0, 1]
                p_cropped = np.clip(p_cropped, 0, 1)
                predictions.append(p_cropped)
                
    return np.stack(predictions, axis=2)

def main():
    print(f"Loading GAN Generator: {CFG['model_path']}")
    
    # MUST MATCH THE ARCHITECTURE IN train_gan.py
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
    
    predictions_dict = {}
    
    for fpath in test_files:
        fname = os.path.basename(fpath)
        sample_key = fname.split('_lowfield')[0]
        pred_vol = predict_volume(model, fpath)
        predictions_dict[sample_key] = pred_vol
            
    print("Generating GAN submission...")
    submission_df = create_submission_df(predictions_dict)
    
    submission_df.to_csv('submission_gan.csv', index=False)
    print("Done! Saved to submission_gan.csv")

if __name__ == '__main__':
    main()