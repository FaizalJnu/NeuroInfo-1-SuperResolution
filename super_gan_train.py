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
from loss_utils import PerceptualLoss

# --- CONFIG ---
CFG = {
    'lr_G': 1e-4,
    'lr_D': 2e-4,
    'batch_size': 8,        # Fits on 12GB with EfficientNet-B0/B1
    'epochs': 25,           # GANs need time to mature
    'img_size': 256,
    'input_depth': 5,       # 2.5D with 5 slices (z-2 to z+2)
    'lambda_adv': 0.05,     # GAN Loss weight
    'lambda_l1': 10.0,      # Pixel Loss weight (Structure)
    'lambda_vgg': 0.1,      # Perceptual Loss weight (Texture)
    'device': 'cuda',
    # Option: Load your previous 0.529 generator to finetune?
    # 'pretrained': 'gan_generator_epoch20.pth' 
    'pretrained': None 
}

# --- IMPROVED DISCRIMINATOR (Spectral Norm for stability) ---
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        def block(in_c, out_c, normalize=True):
            layers = [nn.utils.spectral_norm(nn.Conv2d(in_c, out_c, 4, 2, 1))]
            if normalize: layers.append(nn.InstanceNorm2d(out_c)) #type: ignore
            layers.append(nn.LeakyReLU(0.2, inplace=True)) #type: ignore
            return layers

        self.model = nn.Sequential(
            *block(1, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.Conv2d(512, 1, 3, 1, 1) # Patch output
        )

    def forward(self, x):
        return self.model(x)

# --- PERCEPTUAL LOSS CLASS (Paste snippet from Step 1 here if not importing) ---
# (Assuming you included the class above or imported it)

# --- DATASET (5-Slice Support) ---
class MRIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = sorted(glob.glob(os.path.join(root_dir, 'train', 'low_field', '*.nii*')))
        self.samples = []
        
        # We need padding for 5 slices (skip first 2 and last 2)
        # Slice range: 2 to 198
        for vol_idx, fname in enumerate(self.files):
            fname = os.path.basename(fname)
            for s in range(2, 198): 
                self.samples.append((vol_idx, s, fname))
        self.cache = {}

    def load_volume(self, path):
        vol = nib.load(path).get_fdata() #type: ignore
        vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
        return vol.astype(np.float32)

    def upsample_lf(self, vol):
        z, y, x = 179 / vol.shape[0], 221 / vol.shape[1], 200 / vol.shape[2]
        return zoom(vol, (z, y, x), order=3)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vol_idx, slice_idx, fname = self.samples[idx]
        
        # ... (Caching logic same as before) ...
        if f'lf_{vol_idx}' not in self.cache:
            lf_path = os.path.join(self.root_dir, 'train', 'low_field', fname)
            hf_path = os.path.join(self.root_dir, 'train', 'high_field', fname.replace('lowfield', 'highfield'))
            self.cache[f'lf_{vol_idx}'] = self.upsample_lf(self.load_volume(lf_path))
            self.cache[f'hf_{vol_idx}'] = self.load_volume(hf_path)
            
        lf_vol = self.cache[f'lf_{vol_idx}']
        hf_vol = self.cache[f'hf_{vol_idx}']
        
        # Inputs: 5-Slice Stack [z-2, z-1, z, z+1, z+2]
        img_stack = lf_vol[:, :, slice_idx-2 : slice_idx+3]
        target = hf_vol[:, :, slice_idx]
        
        if self.transform:
            t_exp = np.expand_dims(target, axis=2)
            augmented = self.transform(image=img_stack, mask=t_exp)
            img_stack = augmented['image']
            target = augmented['mask'].permute(2, 0, 1)

        return img_stack, target

# --- TRAINING LOOP ---
def main():
    # 1. Generator: U-Net++ with EfficientNet Encoder
    # This is much stronger than ResNet34 U-Net
    generator = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4", # Strong feature extractor
        encoder_weights="imagenet",
        in_channels=CFG['input_depth'], # 5 Channels
        classes=1,
        activation='sigmoid'
    ).to(CFG['device'])
    
    discriminator = Discriminator().to(CFG['device'])
    
    # Losses
    crit_gan = nn.MSELoss()
    crit_l1 = nn.L1Loss()
    crit_vgg = PerceptualLoss(CFG['device']) # The new secret weapon

    opt_G = torch.optim.Adam(generator.parameters(), lr=CFG['lr_G'], betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=CFG['lr_D'], betas=(0.5, 0.999))
    scaler = GradScaler()

    # Transforms (Keep your "Safe" list)
    train_aug = A.Compose([
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.RandomCrop(height=256, width=256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.GaussNoise(var_limit=(0.001, 0.005), p=0.5),
        A.Lambda(image=lambda x, **kwargs: np.clip(x, 0, 1)),
        # Add slight gamma again if safe, or skip to be careful
        ToTensorV2()
    ]) #type:ignore

    ds = MRIDataset(root_dir='.', transform=train_aug)
    loader = DataLoader(ds, batch_size=CFG['batch_size'], shuffle=True, num_workers=0)

    print("Starting Super-GAN Training...")
    
    for epoch in range(CFG['epochs']):
        generator.train()
        loop = tqdm(loader, desc=f"Ep {epoch+1}")
        
        for imgs, real in loop:
            imgs, real = imgs.to(CFG['device']), real.to(CFG['device'])
            
            # --- Train Generator ---
            opt_G.zero_grad()
            with torch.amp.autocast('cuda'):
                fake = generator(imgs)
                
                # 1. Adversarial Loss (Texture)
                pred_fake = discriminator(fake)
                loss_adv = crit_gan(pred_fake, torch.ones_like(pred_fake))
                
                # 2. Pixel Loss (Structure)
                loss_pix = crit_l1(fake, real)
                
                # 3. Perceptual Loss (Features)
                loss_perc = crit_vgg(fake, real)
                
                # Weighted Sum
                loss_G = (CFG['lambda_adv'] * loss_adv) + \
                         (CFG['lambda_l1'] * loss_pix) + \
                         (CFG['lambda_vgg'] * loss_perc)

            scaler.scale(loss_G).backward()
            scaler.step(opt_G)
            
            # --- Train Discriminator ---
            opt_D.zero_grad()
            with autocast():
                pred_real = discriminator(real)
                pred_fake = discriminator(fake.detach())
                
                loss_real = crit_gan(pred_real, torch.ones_like(pred_real))
                loss_fake = crit_gan(pred_fake, torch.zeros_like(pred_fake))
                loss_D = 0.5 * (loss_real + loss_fake)

            scaler.scale(loss_D).backward()
            scaler.step(opt_D)
            scaler.update()
            
            loop.set_postfix(G=loss_G.item(), D=loss_D.item())
            
        torch.save(generator.state_dict(), f"supergan_epoch{epoch+1}.pth")

if __name__ == '__main__':
    main()