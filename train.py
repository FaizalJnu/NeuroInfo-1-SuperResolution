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

# CONFIG
CFG = {
    'lr': 3e-4,
    'batch_size': 16,  # 12 GB VRAM Limit
    'epochs': 15,
    'img_size': 256,  
    'target_shape': (179, 221, 200), # Original HF shape
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print(f"Running on: {CFG['device']}")

class MRIDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        search_path = os.path.join(root_dir, 'train' if mode != 'test' else 'test', 'low_field', '*.nii')
        self.lf_files = sorted(glob.glob(search_path))
        
        print(f"[{mode.upper()}] Looking in: {search_path}")
        print(f"[{mode.upper()}] Found {len(self.lf_files)} files.")
        
        if len(self.lf_files) == 0:
            print("ERROR: No files found! Check your folder structure.")
            print(f"Expected to find 'train/low_field' inside '{os.path.abspath(root_dir)}'")
        
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
            path = os.path.join(self.root_dir, 'train' if self.mode != 'test' else 'test', 'low_field', fname)
            raw_vol = self.load_volume(path)
            lf_vol = self.upsample_lf(raw_vol)
            if self.CACHE: self.cache[f'lf_{vol_idx}'] = lf_vol
            
        target_slice = None
        if self.mode == 'train':
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
            if target_slice is not None:
                t_exp = np.expand_dims(target_slice, axis=2)
                augmented = self.transform(image=img_stack, mask=t_exp)
                img_stack = augmented['image']
                target_slice = augmented['mask'].permute(2, 0, 1)
            else:
                augmented = self.transform(image=img_stack)
                img_stack = augmented['image']

        return img_stack, target_slice if target_slice is not None else 0

# TRANSFORMATIONS
train_aug = A.Compose([
    A.PadIfNeeded(min_height=CFG['img_size'], min_width=CFG['img_size'], border_mode=cv2.BORDER_CONSTANT, value=0),
    A.RandomCrop(height=CFG['img_size'], width=CFG['img_size']),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    ToTensorV2()
]) # type: ignore 

val_aug = A.Compose([
    A.PadIfNeeded(min_height=CFG['img_size'], min_width=CFG['img_size'], border_mode=cv2.BORDER_CONSTANT, value=0),
    ToTensorV2()
]) # type: ignore 

# LOSS FUNCTION as given in evaluation section
class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=1)
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        return 0.5 * (1 - self.ssim(pred, target)) + 0.5 * self.l1(pred, target)

def main():
    full_dataset = MRIDataset(root_dir='.', mode='train', transform=train_aug)
    
    if len(full_dataset) == 0:
        print("\n!!! DATASET IS EMPTY - STOPPING !!!")
        sys.exit(1)

    train_loader = DataLoader(full_dataset, batch_size=CFG['batch_size'], shuffle=True, num_workers=4, pin_memory=True)

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

    print("Starting Training...")
    
    min_loss = float('inf')
    for epoch in range(CFG['epochs']):
        model.train()
        loop = tqdm(train_loader, total=len(train_loader))
        avg_loss = 0
        
        for imgs, targets in loop:
            imgs = imgs.to(CFG['device'])
            targets = targets.to(CFG['device'])
            
            with autocast():
                preds = model(imgs)
                loss = criterion(preds, targets)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            avg_loss += loss.item()
            loop.set_description(f"Epoch {epoch+1}")
            loop.set_postfix(loss=loss.item())
        
        print(f"Epoch {epoch+1} Average Loss: {avg_loss / len(train_loader)}")
        
        if avg_loss < min_loss:
            min_loss = avg_loss
            torch.save(model.state_dict(), f"unet_25d_epoch_best.pth")
            print(f"Saved Best Model with Loss: {min_loss / len(train_loader)}")

        torch.save(model.state_dict(), f"unet_25d_epoch_latest.pth")

if __name__ == '__main__':
    main()