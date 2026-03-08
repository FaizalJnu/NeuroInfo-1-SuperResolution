import os
import glob
import sys
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from scipy.ndimage import zoom
from pytorch_msssim import SSIM
import segmentation_models_pytorch as smp
import cv2
from sklearn.model_selection import KFold

# CONFIG
CFG = {
    'lr': 3e-4,
    'batch_size': 16,    
    'epochs': 15,        # 15 epochs per fold
    'img_size': 256,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'n_folds': 5,        # 5-Fold CV
    'seed': 42           # Deterministic split
}

print(f"Running 5-Fold CV on: {CFG['device']}")

# DATASET CLASS
class MRIDataset(Dataset):
    def __init__(self, file_list, root_dir, mode='train', transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.lf_files = file_list # pass the specific list of files for this fold
        
        self.samples = []
        for vol_idx, fname in enumerate(self.lf_files):
            fname = os.path.basename(fname)
            for s in range(1, 199): 
                self.samples.append((vol_idx, s, fname))
        
        self.CACHE = True
        self.cache = {}

    def load_volume(self, path):
        vol = nib.load(path).get_fdata() # type: ignore
        vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
        return vol.astype(np.float32)

    def upsample_lf(self, vol):
        z_factor = 179 / vol.shape[0]
        y_factor = 221 / vol.shape[1]
        x_factor = 200 / vol.shape[2]
        return zoom(vol, (z_factor, y_factor, x_factor), order=3)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vol_idx, slice_idx, fname = self.samples[idx]
        
        if self.CACHE and f'lf_{vol_idx}' in self.cache:
            lf_vol = self.cache[f'lf_{vol_idx}']
        else:
            path = os.path.join(self.root_dir, 'train', 'low_field', fname)
            raw_vol = self.load_volume(path)
            lf_vol = self.upsample_lf(raw_vol)
            if self.CACHE: self.cache[f'lf_{vol_idx}'] = lf_vol
            
        target_slice = None
        if self.CACHE and f'hf_{vol_idx}' in self.cache:
            hf_vol = self.cache[f'hf_{vol_idx}']
        else:
            hf_fname = fname.replace('lowfield', 'highfield')
            path = os.path.join(self.root_dir, 'train', 'high_field', hf_fname)
            hf_vol = self.load_volume(path)
            if self.CACHE: self.cache[f'hf_{vol_idx}'] = hf_vol
        
        target_slice = hf_vol[:, :, slice_idx]

        img_stack = lf_vol[:, :, slice_idx-1 : slice_idx+2]
        
        if self.transform:
            t_exp = np.expand_dims(target_slice, axis=2)
            augmented = self.transform(image=img_stack, mask=t_exp)
            img_stack = augmented['image']
            target_slice = augmented['mask'].permute(2, 0, 1)

        return img_stack, target_slice

# TRANSFORMATIONS
# --- SAFE TRANSFORMS ---
# train_aug = A.Compose([
#     # 1. Geometry #type: ignore
#     A.PadIfNeeded(min_height=CFG['img_size'], min_width=CFG['img_size'], border_mode=cv2.BORDER_CONSTANT, value=0),
#     A.RandomCrop(height=CFG['img_size'], width=CFG['img_size']),
#     A.HorizontalFlip(p=0.5),
#     A.VerticalFlip(p=0.5),
#     A.RandomRotate90(p=0.5),
    
#     # 2. Physics (Safe Mode)
#     # Add noise first
#     A.GaussNoise(var_limit=(0.001, 0.01), p=0.5),
    
#     # CRITICAL: Clip values to [0, 1] BEFORE Gamma to prevent NaNs
#     A.ToFloat(max_value=1.0), 
    
#     # Now it's safe to run Gamma
#     A.RandomGamma(gamma_limit=(80, 120), p=0.5),
#     A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    
#     # 3. Final Format
#     ToTensorV2()
# ])

# --- SAFE TRANSFORMS ---
# --- SAFE TRANSFORMS ---
train_aug = A.Compose([
    # 1. Geometry (Safe)
    A.PadIfNeeded(min_height=CFG['img_size'], min_width=CFG['img_size'], border_mode=cv2.BORDER_CONSTANT, value=0),
    A.RandomCrop(height=CFG['img_size'], width=CFG['img_size']),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    
    # 2. Add Noise (This creates values < 0.0 and > 1.0)
    A.GaussNoise(var_limit=(0.001, 0.01), p=0.5),
    
    # 3. CRITICAL: Custom Clip using Lambda
    # This forces all pixels to stay strictly between 0.0 and 1.0
    A.Lambda(image=lambda x, **kwargs: np.clip(x, 0, 1)),
    
    # 4. Intensity Transforms (Now safe because input is [0, 1])
    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    
    # 5. Final Tensor Conversion
    ToTensorV2()
])
val_aug = A.Compose([
    A.PadIfNeeded(min_height=CFG['img_size'], min_width=CFG['img_size'], border_mode=cv2.BORDER_CONSTANT, value=0),
    ToTensorV2()
]) # type: ignore

# LOSS FUNCTION SAME AS EVALUATION SECTION
class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=1)
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        return 0.5 * (1 - self.ssim(pred, target)) + 0.5 * self.l1(pred, target)


def train_fold(fold, train_files, val_files):
    print(f"\n=== FOLD {fold+1}/{CFG['n_folds']} ===")
    print(f"Train Volumes: {len(train_files)} | Val Volumes: {len(val_files)}")

    train_ds = MRIDataset(train_files, root_dir='.', transform=train_aug)
    val_ds = MRIDataset(val_files, root_dir='.', transform=val_aug)

    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CFG['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    model = smp.Unet(
        encoder_name="resnet34",        
        encoder_weights="imagenet",     
        in_channels=3,                  
        classes=1,                      
        activation='sigmoid'
    ).to(CFG['device'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['lr'])
    criterion = CombinedLoss()
    scaler = GradScaler()

    best_score = float('inf')

    for epoch in range(CFG['epochs']):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False)
        
        for imgs, targets in loop:
            imgs, targets = imgs.to(CFG['device']), targets.to(CFG['device'])
            
            with autocast():
                preds = model(imgs)
                loss = criterion(preds, targets)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(CFG['device']), targets.to(CFG['device'])
                preds = model(imgs)
                loss = criterion(preds, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f}")

        # Saving Best Model for this Fold
        if avg_val_loss < best_score:
            best_score = avg_val_loss
            torch.save(model.state_dict(), f"unet_fold{fold}.pth")
            print(f"  >>> Saved Best Fold {fold} Model!")

def main():
    all_files = sorted(glob.glob(os.path.join('train', 'low_field', '*.nii*')))
    
    kf = KFold(n_splits=CFG['n_folds'], shuffle=True, random_state=CFG['seed'])

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_files)):
        train_files = [all_files[i] for i in train_idx]
        val_files = [all_files[i] for i in val_idx]
        
        train_fold(fold, train_files, val_files)

if __name__ == '__main__':
    main()