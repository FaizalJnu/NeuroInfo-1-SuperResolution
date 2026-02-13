import os
import glob
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from scipy.ndimage import zoom
import segmentation_models_pytorch as smp
import cv2

# --- CONFIGURATION ---
CFG = {
    'lr_G': 1e-4,        # Generator learning rate
    'lr_D': 2e-4,        # Discriminator learning rate (usually higher)
    'batch_size': 8,     # Smaller batch size for GANs (VRAM intensive)
    'epochs': 20,
    'img_size': 256,
    'lambda_adv': 0.01,  # Weight for GAN loss (Texture)
    'lambda_l1': 1.0,    # Weight for Pixel loss (Structure)
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'pretrained_unet': 'unet_25d_epoch15.pth' # <--- PATH TO YOUR BEST MODEL
}

print(f"Running GAN Training on: {CFG['device']}")

# --- 1. DATASET & TRANSFORMS (Standardized) ---
class MRIDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = sorted(glob.glob(os.path.join(root_dir, 'train', 'low_field', '*.nii*')))
        self.samples = []
        for vol_idx, fname in enumerate(self.files):
            fname = os.path.basename(fname)
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
        
        if f'lf_{vol_idx}' not in self.cache:
            lf_path = os.path.join(self.root_dir, 'train', 'low_field', fname)
            hf_path = os.path.join(self.root_dir, 'train', 'high_field', fname.replace('lowfield', 'highfield'))
            
            lf_vol = self.upsample_lf(self.load_volume(lf_path))
            hf_vol = self.load_volume(hf_path)
            
            self.cache[f'lf_{vol_idx}'] = lf_vol
            self.cache[f'hf_{vol_idx}'] = hf_vol
            
        lf_vol = self.cache[f'lf_{vol_idx}']
        hf_vol = self.cache[f'hf_{vol_idx}']
        
        # Inputs: 2.5D Stack
        img_stack = lf_vol[:, :, slice_idx-1 : slice_idx+2]
        # Target: Middle Slice
        target = hf_vol[:, :, slice_idx]
        
        if self.transform:
            t_exp = np.expand_dims(target, axis=2)
            augmented = self.transform(image=img_stack, mask=t_exp)
            img_stack = augmented['image']
            target = augmented['mask'].permute(2, 0, 1)

        return img_stack, target

# Safe Augmentations
train_aug = A.Compose([
    A.PadIfNeeded(min_height=CFG['img_size'], min_width=CFG['img_size'], border_mode=cv2.BORDER_CONSTANT, value=0),
    A.RandomCrop(height=CFG['img_size'], width=CFG['img_size']),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.GaussNoise(var_limit=(0.001, 0.005), p=0.5), # Light noise for GAN
    A.Lambda(image=lambda x, **kwargs: np.clip(x, 0, 1)), # Clip safety
    ToTensorV2()
]) #type: ignore

# --- 2. THE DISCRIMINATOR (The Critic) ---
class PatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 1 Channel (The image to judge)
        # Output: 1 Channel Map (Real/Fake decision for each patch)
        
        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(1, 64, normalization=False), # 128x128
            *discriminator_block(64, 128),                    # 64x64
            *discriminator_block(128, 256),                   # 32x32
            *discriminator_block(256, 512),                   # 16x16
            nn.Conv2d(512, 1, 3, padding=1)                   # 16x16 Output Map
        )

    def forward(self, img):
        return self.model(img)

# --- 3. TRAINING LOOP ---
def main():
    # 1. Initialize Generator (Your U-Net)
    generator = smp.Unet(
        encoder_name="resnet34", 
        in_channels=3, 
        classes=1, 
        activation='sigmoid'
    ).to(CFG['device'])
    
    # Load pre-trained weights if available (HIGHLY RECOMMENDED)
    if os.path.exists(CFG['pretrained_unet']):
        print(f"Loading pre-trained Generator: {CFG['pretrained_unet']}")
        generator.load_state_dict(torch.load(CFG['pretrained_unet']))
    else:
        print("WARNING: Starting Generator from scratch (Slower convergence)")

    # 2. Initialize Discriminator
    discriminator = PatchDiscriminator().to(CFG['device'])

    # 3. Optimizers & Loss
    opt_G = torch.optim.Adam(generator.parameters(), lr=CFG['lr_G'], betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=CFG['lr_D'], betas=(0.5, 0.999))
    
    criterion_GAN = nn.MSELoss() # LSGAN Loss (More stable than BCE)
    criterion_pixel = nn.L1Loss()
    
    scaler = GradScaler()
    
    # Data
    dataset = MRIDataset(root_dir='.', transform=train_aug)
    loader = DataLoader(dataset, batch_size=CFG['batch_size'], shuffle=True, num_workers=0) # 0 for Windows safety

    print("Starting GAN Training...")
    
    for epoch in range(CFG['epochs']):
        generator.train()
        discriminator.train()
        
        loop = tqdm(loader, desc=f"Ep {epoch+1}")
        
        for imgs, real_targets in loop:
            imgs = imgs.to(CFG['device'])           # (B, 3, 256, 256)
            real_targets = real_targets.to(CFG['device']) # (B, 1, 256, 256)
            
            # --- Train Generator ---
            opt_G.zero_grad()
            # with autocast():
            with torch.amp.autocast('cuda'):
                # 1. Generate Fake Image
                fake_targets = generator(imgs)
                
                # 2. Fool the Discriminator? (Adversarial Loss)
                # We want D to output '1' (Real) for our fakes
                pred_fake = discriminator(fake_targets)
                valid = torch.ones_like(pred_fake) 
                loss_adv = criterion_GAN(pred_fake, valid)
                
                # 3. Match the Pixels? (L1 Loss)
                loss_pix = criterion_pixel(fake_targets, real_targets)
                
                # Total Generator Loss
                loss_G = (CFG['lambda_adv'] * loss_adv) + (CFG['lambda_l1'] * loss_pix)
            
            scaler.scale(loss_G).backward()
            scaler.step(opt_G)
            
            # --- Train Discriminator ---
            opt_D.zero_grad()
            with autocast():
                # 1. Real Loss (Should predict 1)
                pred_real = discriminator(real_targets)
                valid = torch.ones_like(pred_real)
                loss_real = criterion_GAN(pred_real, valid)
                
                # 2. Fake Loss (Should predict 0)
                # Detach fake_targets so we don't backprop into Generator here
                pred_fake = discriminator(fake_targets.detach())
                fake = torch.zeros_like(pred_fake)
                loss_fake = criterion_GAN(pred_fake, fake)
                
                loss_D = 0.5 * (loss_real + loss_fake)

            scaler.scale(loss_D).backward()
            scaler.step(opt_D)
            scaler.update()
            
            loop.set_postfix(G_loss=loss_G.item(), D_loss=loss_D.item())
            
        # Save every epoch
        torch.save(generator.state_dict(), f"gan_generator_epoch{epoch+1}.pth")

if __name__ == '__main__':
    main()