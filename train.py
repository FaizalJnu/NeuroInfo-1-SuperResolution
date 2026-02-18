import os
import glob
import sys
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler #type: ignore
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from scipy.ndimage import zoom
from pytorch_msssim import SSIM, MS_SSIM
import segmentation_models_pytorch as smp
import cv2 

# CONFIG
CFG = {
    'lr': 3e-4,
    'batch_size': 32,  # 12 GB VRAM Limit
    'data_dir': 'train_synthetic_npy', # Use synthetic data for better GAN training
    'epochs': 15,
    'img_size': 256,  
    'target_shape': (179, 221, 200), # Original HF shape
    'device': 'cuda',
    'use_amp': True,
    'resume_path': 'pretrain_unet_epoch_latest.pth'
}

# print(f"Running on: {CFG['device']}")

class MRIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Search for .npy files now
        # Note the recursive search to find files inside 'low_field' subfolder
        self.lf_files = sorted(glob.glob(os.path.join(root_dir, CFG['data_dir'], 'low_field', '*.npy')))
        
        print(f"Found {len(self.lf_files)} .npy brains.")
        
        self.samples = []
        for vol_idx, fname in enumerate(self.lf_files):
            fname = os.path.basename(fname)
            # Use specific slice range to avoid empty space
            for s in range(20, 180, 2): 
                self.samples.append((vol_idx, s, fname))
    
    # We remove load_volume and upsample_lf because .npy is already processed
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        vol_idx, slice_idx, fname = self.samples[idx]
        
        # Build paths
        lf_path = os.path.join(self.root_dir, CFG['data_dir'], 'low_field', fname)
        hf_path = os.path.join(self.root_dir, CFG['data_dir'], 'high_field', fname.replace('lowfield', 'highfield'))
        
        # --- THE SPEED HACK: mmap_mode='r' ---
        # This opens the file instantly without reading data into RAM.
        # It creates a "virtual" array linked to the SSD.
        lf_vol = np.load(lf_path, mmap_mode='r')
        hf_vol = np.load(hf_path, mmap_mode='r')
        
        # When we slice [:,:,s], it physically reads ONLY those bytes from disk.
        # .copy() forces the data into RAM as a real array for the GPU.
        img_stack = lf_vol[:, :, slice_idx-1 : slice_idx+2].copy()
        target = hf_vol[:, :, slice_idx].copy()
        
        # Ensure correct shape (H, W, C) for Albumentations
        # If your data is (H, W, D), slice gives (H, W). Stack gives (H, W, 3).
        # Normalization is already done in generation step (0-1 float32)
        
        if self.transform:
            t_exp = np.expand_dims(target, axis=2)
            augmented = self.transform(image=img_stack, mask=t_exp)
            img_stack = augmented['image']
            target = augmented['mask'].permute(2, 0, 1)

        return img_stack, target
    

# class MRIDataset(Dataset):
#     def __init__(self, root_dir, mode='train', transform=None):
#         self.root_dir = root_dir
#         self.mode = mode
#         self.transform = transform
#         # Note the '*' at the end of .nii*
#         search_path = os.path.join(root_dir, 'train_synthetic' if mode != 'test' else 'test', 'low_field', '*.nii*')
#         self.lf_files = sorted(glob.glob(search_path))
        
#         print(f"[{mode.upper()}] Looking in: {search_path}")
#         print(f"[{mode.upper()}] Found {len(self.lf_files)} files.")
        
#         if len(self.lf_files) == 0:
#             print("ERROR: No files found! Check your folder structure.")
#             print(f"Expected to find 'train/low_field' inside '{os.path.abspath(root_dir)}'")
        
#         self.samples = []
#         for vol_idx, fname in enumerate(self.lf_files):
#             fname = os.path.basename(fname) 
#             for s in range(1, 199): 
#                 self.samples.append((vol_idx, s, fname))
        
#         self.CACHE = False
#         self.cache = {}

#     def load_volume(self, path):
#         vol = nib.load(path).get_fdata() # type: ignore 
#         vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
#         return vol.astype(np.float32)

#     def upsample_lf(self, vol):
#         z_factor = 179 / vol.shape[0]
#         y_factor = 221 / vol.shape[1]
#         x_factor = 200 / vol.shape[2]
#         return zoom(vol, (z_factor, y_factor, x_factor), order=3)

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         vol_idx, slice_idx, fname = self.samples[idx]
        
#         # Check cache first
#         cache_key = (vol_idx, fname)
#         lf_path = os.path.join(self.root_dir, CFG['data_dir'], 'low_field', fname)
#         hf_fname = fname.replace('lowfield', 'highfield')
#         hf_path = os.path.join(self.root_dir, CFG['data_dir'], 'high_field', hf_fname)
#         if cache_key not in self.cache:
#             lf_vol = self.upsample_lf(self.load_volume(lf_path))
#             hf_vol = self.load_volume(hf_path)
#             self.cache[cache_key] = (lf_vol, hf_vol)
#         else:
#             lf_vol, hf_vol = self.cache[cache_key]
#         # Load volumes
#         try:
#             lf_vol = self.upsample_lf(self.load_volume(lf_path))
#             hf_vol = self.load_volume(hf_path)
#         except FileNotFoundError:
#             # Fallback debug print if it fails again
#             print(f"FAILED TO LOAD: {lf_path}")
#             raise

#         # 2.5D Stack [z-1, z, z+1]
#         img_stack = lf_vol[:, :, slice_idx-1 : slice_idx+2]
#         target = hf_vol[:, :, slice_idx]
        
#         if self.transform:
#             t_exp = np.expand_dims(target, axis=2)
#             augmented = self.transform(image=img_stack, mask=t_exp)
#             img_stack = augmented['image']
#             target = augmented['mask'].permute(2, 0, 1)

#         return img_stack, target
# TRANSFORMATIONS
train_aug = A.Compose([
    A.PadIfNeeded(min_height=CFG['img_size'], min_width=CFG['img_size'], border_mode=cv2.BORDER_CONSTANT, fill=0),
    A.RandomCrop(height=CFG['img_size'], width=CFG['img_size']),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    ToTensorV2()
]) # type: ignore 

val_aug = A.Compose([
    A.PadIfNeeded(min_height=CFG['img_size'], min_width=CFG['img_size'], border_mode=cv2.BORDER_CONSTANT, fill=0),
    ToTensorV2()
]) # type: ignore 

# LOSS FUNCTION as given in evaluation section
class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # win_size=11 and sigma=1.5 are standard for MS-SSIM
        self.msssim = MS_SSIM(
            data_range=1.0, 
            size_average=True, 
            channel=1, 
            win_size=11
        )
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        # MS-SSIM returns 1 for perfect match, so we use (1 - msssim) to minimize loss
        # Note: MS-SSIM can sometimes return NaN if the image is too small 
        # (needs to be at least 160x160 for 5 scales)
        ms_ssim_loss = 1 - self.msssim(pred, target)
        l1_loss = self.l1(pred, target)
        
        return 0.8 * ms_ssim_loss + 0.2 * l1_loss
def main():
    full_dataset = MRIDataset(root_dir='.', transform=train_aug)
    
    if len(full_dataset) == 0:
        print("\n!!! DATASET IS EMPTY - STOPPING !!!")
        sys.exit(1)

    train_loader = DataLoader(full_dataset, batch_size=CFG['batch_size'], shuffle=True, num_workers=6, pin_memory=True, prefetch_factor=2, persistent_workers=False)

    model = smp.Unet(
        encoder_name="resnet34",        
        encoder_weights="imagenet",     
        in_channels=3,                  
        classes=1,                      
        activation='sigmoid'
    ).to(CFG['device'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['lr'])
    criterion = CombinedLoss()
    scaler = GradScaler('cuda')

    start_epoch = 0
    min_loss = float('inf')
    resume_path = CFG['resume_path']
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=CFG['device'])

        model.load_state_dict(checkpoint)
        optimizer.load_state_dict(checkpoint.get('optimizer', optimizer.state_dict()))
        scaler.load_state_dict(checkpoint.get('scaler', scaler.state_dict()))
        start_epoch = checkpoint.get('epoch', start_epoch)
        min_loss = checkpoint.get('min_loss', min_loss)


    print("Starting Training...")
    for epoch in range(CFG['epochs']):
        model.train()
        loop = tqdm(train_loader, total=len(train_loader))
        avg_loss = 0
        
        for imgs, targets in loop:
            imgs = imgs.to(CFG['device'])
            targets = targets.to(CFG['device'])
            
            with autocast('cuda'):
                preds = model(imgs)
                loss = criterion(preds, targets)

            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'min_loss': min_loss
            }
            
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
            torch.save(checkpoint_data, f"pretrain_unet_epoch_best.pth")
            print(f"Saved Best Model with Loss: {min_loss / len(train_loader)}")

        torch.save(checkpoint_data, f"pretrain_unet_epoch_latest.pth")

if __name__ == '__main__':
    main()