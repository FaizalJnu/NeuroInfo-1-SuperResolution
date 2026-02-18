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
import segmentation_models_pytorch as smp
import cv2
from pytorch_msssim import SSIM, MS_SSIM

# --- CONFIGURATION ---
CFG = {
    'lr_G': 1e-4,        # Generator learning rate
    'lr_D': 2e-4,        # Discriminator learning rate (usually higher)
    'batch_size': 8,     # Smaller batch size for GANs (VRAM intensive)
    'epochs': 20,
    'img_size': 256,
    'lambda_adv': 0.01,  # Weight for GAN loss (Texture)
    'lambda_l1': 1.0,    # Weight for Pixel loss (Structure)
    'lambda_content': 1.0, # Weight for Perceptual loss (Optional
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'pretrained_unet': 'NewGan.pth' # <--- PATH TO YOUR BEST MODEL
}
print(f"Running GAN Training on: {CFG['device']}")

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
    
def clip_image(image, **kwargs):
    return np.clip(image, 0, 1)

# Safe Augmentations
train_aug = A.Compose([
    A.PadIfNeeded(min_height=CFG['img_size'], min_width=CFG['img_size'], border_mode=cv2.BORDER_CONSTANT),
    A.RandomCrop(height=CFG['img_size'], width=CFG['img_size']),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.GaussNoise(std_range=(0.001, 0.005), p=0.5), # Light noise for GAN
    A.Lambda(image=clip_image, p=1.0),
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
    criterion_content = CombinedLoss().to(CFG['device']) # Perceptual Loss for better textures (Optional)
    scaler = GradScaler('cuda')
    
    # Data
    dataset = MRIDataset(root_dir='.', transform=train_aug)
    loader = DataLoader(dataset, batch_size=CFG['batch_size'], shuffle=True, num_workers=6) # 0 for Windows safety

    print("Starting GAN Training...")
    
    for epoch in range(CFG['epochs']):
        generator.train()
        discriminator.train()
        
        loop = tqdm(loader, desc=f"Ep {epoch+1}")
        
        for imgs, real_targets in loop:
            imgs = imgs.to(CFG['device'])
            real_targets = real_targets.to(CFG['device'])
            
            # ---------------------
            #  Train Generator
            # ---------------------
            opt_G.zero_grad()
            with autocast('cuda'):
                # 1. Generate Fake Image (Sigmoid -> [0, 1])
                fake_targets = generator(imgs)
                
                # 2. Adversarial Loss
                pred_fake = discriminator(fake_targets)
                valid = torch.ones_like(pred_fake) 
                loss_adv = criterion_GAN(pred_fake, valid)
                
                # 3. Content Loss (Your new MS-SSIM + L1)
                # No normalization needed because Sigmoid output is already [0, 1]
                loss_content = criterion_content(fake_targets, real_targets)
                
                # Total Generator Loss
                loss_G = (CFG['lambda_adv'] * loss_adv) + (CFG['lambda_content'] * loss_content)
            
            scaler.scale(loss_G).backward()
            scaler.step(opt_G)
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            opt_D.zero_grad()
            with autocast('cuda'):
                # Real Loss
                pred_real = discriminator(real_targets)
                valid = torch.ones_like(pred_real)
                loss_real = criterion_GAN(pred_real, valid)
                
                # Fake Loss
                pred_fake = discriminator(fake_targets.detach())
                fake = torch.zeros_like(pred_fake)
                loss_fake = criterion_GAN(pred_fake, fake)
                
                loss_D = 0.5 * (loss_real + loss_fake)

            scaler.scale(loss_D).backward()
            scaler.step(opt_D)
            
            scaler.update()
            
            loop.set_postfix(G_content=loss_content.item(), G_adv=loss_adv.item())
        
        if loss_content.item() < 0.05: # Arbitrary threshold for saving
            torch.save(generator.state_dict(), f"NewGan_epoch_best_{loss_content.item():.4f}.pth")
            print(f"Saved Generator at epoch {epoch+1} with Content Loss: {loss_content.item():.4f}")
            
        torch.save(generator.state_dict(), f"NewGan_latest.pth")
if __name__ == '__main__':
    main()