import os
import glob
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from scipy.ndimage import zoom
from pytorch_msssim import SSIM, MS_SSIM
import segmentation_models_pytorch as smp
import cv2

# --- CONFIGURATION ---
CFG = {
    'lr': 5e-5,          # LOWER LR for fine-tuning
    'batch_size': 16,    
    'epochs': 20,        # 20 epochs on real data is plenty
    'img_size': 256,
    'device': 'cuda',
    # POINT THIS TO YOUR PRE-TRAINED MODEL
    'pretrained_path': 'pretrain_unet_epoch_best.pth',
    'resume_path': 'finetuned_unet_latest.pth' # For crash recovery
}

print(f"Running Fine-Tuning on: {CFG['device']}")

# --- DATASET (Back to Real Data) ---
class CompetitionDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Search the REAL competition folder
        self.lf_files = sorted(glob.glob(os.path.join(root_dir, 'train', 'low_field', '*.nii*')))
        
        self.samples = []
        for vol_idx, fname in enumerate(self.lf_files):
            fname = os.path.basename(fname)
            # Use ALL slices for fine-tuning
            for s in range(1, 199): 
                self.samples.append((vol_idx, s, fname))
        
        self.cache = {}

    def load_volume(self, path):
        vol = nib.load(path).get_fdata()
        vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
        return vol.astype(np.float32)

    def upsample_lf(self, vol):
        z, y, x = 179 / vol.shape[0], 221 / vol.shape[1], 200 / vol.shape[2]
        return zoom(vol, (z, y, x), order=3)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vol_idx, slice_idx, fname = self.samples[idx]
        
        # Simple caching for the 18 real brains
        if f'lf_{vol_idx}' not in self.cache:
            lf_path = os.path.join(self.root_dir, 'train', 'low_field', fname)
            hf_path = os.path.join(self.root_dir, 'train', 'high_field', fname.replace('lowfield', 'highfield'))
            
            self.cache[f'lf_{vol_idx}'] = self.upsample_lf(self.load_volume(lf_path))
            self.cache[f'hf_{vol_idx}'] = self.load_volume(hf_path)
            
        lf_vol = self.cache[f'lf_{vol_idx}']
        hf_vol = self.cache[f'hf_{vol_idx}']
        
        img_stack = lf_vol[:, :, slice_idx-1 : slice_idx+2]
        target = hf_vol[:, :, slice_idx]
        
        if self.transform:
            t_exp = np.expand_dims(target, axis=2)
            augmented = self.transform(image=img_stack, mask=t_exp)
            img_stack = augmented['image']
            target = augmented['mask'].permute(2, 0, 1)

        return img_stack, target
    
def clip_image(image, **kwargs):
    return np.clip(image, 0, 1)

# --- SAFE AUGMENTATIONS (From our fix earlier) ---
train_aug = A.Compose([
    A.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_CONSTANT),
    A.RandomCrop(height=256, width=256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    
    # Physics Transforms
    A.GaussNoise(std_range=(0.001, 0.01), p=0.5),
    
    # SAFETY CLIP to prevent NaNs
    A.Lambda(image=clip_image, p=1.0), 
    
    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    
    ToTensorV2()
]) #type: ignore

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
    # 1. Setup Model
    model = smp.Unet(
        encoder_name="resnet34",        
        encoder_weights=None, 
        in_channels=3,                  
        classes=1,                      
        activation='sigmoid'
    ).to(CFG['device'])

    # 2. Load Pre-trained Anatomy Weights (The "Base Knowledge")
    # Only load this if we AREN'T resuming a crashed run
    if os.path.exists(CFG['pretrained_path']) and not os.path.exists(CFG['resume_path']):
        print(f"Loading Pre-trained Anatomy Weights: {CFG['pretrained_path']}")
        
        # Load the whole dictionary
        checkpoint = torch.load(CFG['pretrained_path'], map_location=CFG['device'])
        
        # Extract ONLY the model weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Fallback in case you saved just the weights in some versions
            model.load_state_dict(checkpoint)
    elif not os.path.exists(CFG['pretrained_path']):
        print("WARNING: Pre-trained weights not found! Training from scratch.")

    # 3. Setup Data
    dataset = CompetitionDataset(root_dir='.', transform=train_aug)
    loader = DataLoader(dataset, batch_size=CFG['batch_size'], shuffle=True, num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['lr'])
    criterion = CombinedLoss()
    scaler = GradScaler('cuda')

    print("Starting Fine-Tuning...")
    
    start_epoch = 0
    min_loss = float('inf')
    resume_path = CFG['resume_path']
    
    # 4. Resume Logic (The "Crash Recovery")
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=CFG['device'])

        model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        start_epoch = checkpoint.get('epoch', 0)
        min_loss = checkpoint.get('min_loss', float('inf'))
        
        print(f"Resumed successfully. Starting from epoch {start_epoch}")

    # 5. Training Loop
    for epoch in range(start_epoch, CFG['epochs']):
        model.train()
        loop = tqdm(loader, desc=f"Ep {epoch+1}")
        epoch_loss = 0 # Rename to avoid confusion
        
        for imgs, targets in loop:
            imgs, targets = imgs.to(CFG['device']), targets.to(CFG['device'])
            
            with autocast('cuda'):
                preds = model(imgs)
                loss = criterion(preds, targets)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        # --- CRITICAL FIX: Indentation moved BACK ---
        # This now runs ONCE per epoch, not every batch
        
        # Calculate average loss for the epoch
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'min_loss': min_loss
        }

        # Save Best Model
        if avg_loss < min_loss:
            min_loss = avg_loss
            checkpoint_data['min_loss'] = min_loss # Update min_loss in checkpoint
            torch.save(checkpoint_data, "finetuned_unet_best.pth")
            print(f"  >>> Saved Best Model! Loss: {min_loss:.4f}")

        # Save Latest (for resuming)
        # Renamed to 'finetuned' so you don't confuse it with the pretrain file
        torch.save(checkpoint_data, "finetuned_unet_latest.pth") 

if __name__ == '__main__':
    main()