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

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CFG = {
    # Model
    'encoder_name':    'efficientnet-b6',   # Much larger than resnet34
    'encoder_weights': 'imagenet',

    # Training
    'lr_G':            2e-4,
    'lr_D':            4e-4,
    'batch_size':      32,                   # A100 40 GB can handle this at 384px
    'epochs':          60,
    'img_size':        384,                  # Larger crops → better MS-SSIM context across 5 scales

    # Loss weights  (MS-SSIM dominant, as that's the eval metric)
    'lambda_adv':      0.01,
    'lambda_content':  1.0,

    # Paths
    'data_dir':        '/scratch/fr2471/ni_data/',
    'pretrained_unet': 'NewGan.pth',
    'save_dir':        'checkpoints',

    # Hardware
    'device':          'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers':     12,
}

os.makedirs(CFG['save_dir'], exist_ok=True)
print(f"Running GAN Training on: {CFG['device']}")
print(f"Encoder: {CFG['encoder_name']}  |  img_size: {CFG['img_size']}  |  batch: {CFG['batch_size']}")


# ─────────────────────────────────────────────
# LOSS: MS-SSIM dominant
# ─────────────────────────────────────────────
class CombinedLoss(nn.Module):
    """
    MS-SSIM (70%) + single-scale SSIM (20%) + L1 (10%).
    MS-SSIM directly matches the competition metric.
    Single-scale SSIM stabilises early training.
    L1 prevents hallucinations.
    """
    def __init__(self):
        super().__init__()
        self.msssim = MS_SSIM(
            data_range=1.0, size_average=True, channel=1, win_size=11
        )
        self.ssim = SSIM(
            data_range=1.0, size_average=True, channel=1, win_size=11
        )
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        ms_ssim_loss = 1.0 - self.msssim(pred, target)
        ssim_loss    = 1.0 - self.ssim(pred, target)
        l1_loss      = self.l1(pred, target)
        return 0.70 * ms_ssim_loss + 0.20 * ssim_loss + 0.10 * l1_loss


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
def clip_image(image, **kwargs):
    return np.clip(image, 0, 1)

train_aug = A.Compose([
    A.PadIfNeeded(
        min_height=CFG['img_size'], min_width=CFG['img_size'],
        border_mode=cv2.BORDER_CONSTANT
    ),
    A.RandomCrop(height=CFG['img_size'], width=CFG['img_size']),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.3),
    A.GaussNoise(std_range=(0.001, 0.005), p=0.4),
    A.Lambda(image=clip_image, p=1.0),
    ToTensorV2()
])  # type: ignore


class MRIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir  = root_dir
        self.transform = transform
        self.files     = sorted(
            glob.glob(os.path.join(root_dir, 'train', 'low_field', '*.nii*'))
        )
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
        z = 179 / vol.shape[0]
        y = 221 / vol.shape[1]
        x = 200 / vol.shape[2]
        return zoom(vol, (z, y, x), order=3)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vol_idx, slice_idx, fname = self.samples[idx]

        if f'lf_{vol_idx}' not in self.cache:
            lf_path = os.path.join(self.root_dir, 'train', 'low_field', fname)
            hf_path = os.path.join(
                self.root_dir, 'train', 'high_field',
                fname.replace('lowfield', 'highfield')
            )
            self.cache[f'lf_{vol_idx}'] = self.upsample_lf(self.load_volume(lf_path))
            self.cache[f'hf_{vol_idx}'] = self.load_volume(hf_path)

        lf_vol = self.cache[f'lf_{vol_idx}']
        hf_vol = self.cache[f'hf_{vol_idx}']

        # 2.5D stack (3 adjacent slices → 3 input channels)
        img_stack = lf_vol[:, :, slice_idx - 1: slice_idx + 2]
        target    = hf_vol[:, :, slice_idx]

        if self.transform:
            t_exp     = np.expand_dims(target, axis=2)
            augmented = self.transform(image=img_stack, mask=t_exp)
            img_stack = augmented['image']
            target    = augmented['mask'].permute(2, 0, 1)

        return img_stack, target


# ─────────────────────────────────────────────
# DISCRIMINATOR
# ─────────────────────────────────────────────
class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator — judges 16×16 patches.
    Deeper than before to match the larger generator.
    """
    def __init__(self):
        super().__init__()

        def block(in_f, out_f, norm=True):
            layers = [nn.Conv2d(in_f, out_f, 4, stride=2, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(1,    64,  norm=False),  # 192
            *block(64,   128),              # 96
            *block(128,  256),              # 48
            *block(256,  512),              # 24
            *block(512,  512),              # 12
            nn.Conv2d(512, 1, 3, padding=1) # 12×12 patch map
        )

    def forward(self, img):
        return self.model(img)


# ─────────────────────────────────────────────
# MS-SSIM VALIDATION UTILITY
# ─────────────────────────────────────────────
def compute_val_msssim(generator, loader, device, n_batches=20):
    """Quick per-epoch MS-SSIM estimate on a subset of training data."""
    msssim_fn = MS_SSIM(data_range=1.0, size_average=True, channel=1, win_size=11).to(device)
    generator.eval()
    scores = []
    with torch.no_grad():
        for i, (imgs, targets) in enumerate(loader):
            if i >= n_batches:
                break
            imgs    = imgs.to(device)
            targets = targets.to(device)
            preds   = generator(imgs)
            score   = msssim_fn(preds, targets).item()
            scores.append(score)
    generator.train()
    return float(np.mean(scores))


# ─────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────
def main():
    device = CFG['device']

    # ── Generator ──────────────────────────────────────────────────────
    generator = smp.UnetPlusPlus(          # UNet++ for better multi-scale features
        encoder_name=CFG['encoder_name'],
        encoder_weights=CFG['encoder_weights'],
        in_channels=3,
        classes=1,
        activation='sigmoid',
    ).to(device)

    if os.path.exists(CFG['pretrained_unet']):
        print(f"Loading pre-trained Generator: {CFG['pretrained_unet']}")
        generator.load_state_dict(
            torch.load(CFG['pretrained_unet'], map_location=device), strict=False
        )
    else:
        print("WARNING: Starting Generator from scratch (slower convergence)")

    # ── Discriminator ──────────────────────────────────────────────────
    discriminator = PatchDiscriminator().to(device)

    # ── torch.compile (free ~20% speedup on A100 with PyTorch 2.x) ────
    try:
        generator     = torch.compile(generator)
        discriminator = torch.compile(discriminator)
        print("torch.compile enabled")
    except Exception as e:
        print(f"torch.compile not available ({e}), skipping")

    # ── Optimisers ─────────────────────────────────────────────────────
    opt_G = torch.optim.Adam(generator.parameters(),     lr=CFG['lr_G'], betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=CFG['lr_D'], betas=(0.5, 0.999))

    # Cosine annealing: smoothly decays LR to near-zero → better final MS-SSIM
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_G, T_max=CFG['epochs'], eta_min=1e-6
    )
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_D, T_max=CFG['epochs'], eta_min=1e-7
    )

    # ── Losses ─────────────────────────────────────────────────────────
    criterion_GAN     = nn.MSELoss()                          # LSGAN (stable)
    criterion_content = CombinedLoss().to(device)

    scaler = GradScaler('cuda')

    # ── Data ───────────────────────────────────────────────────────────
    dataset = MRIDataset(root_dir=CFG['data_dir'], transform=train_aug)
    loader  = DataLoader(
        dataset,
        batch_size=CFG['batch_size'],
        shuffle=True,
        num_workers=CFG['num_workers'],
        pin_memory=True,
        persistent_workers=True,
    )

    best_msssim = 0.0
    print(f"\nStarting training: {len(dataset)} samples, {len(loader)} batches/epoch\n")

    for epoch in range(CFG['epochs']):
        generator.train()
        discriminator.train()

        running = {'G_content': 0.0, 'G_adv': 0.0, 'D': 0.0}
        loop = tqdm(loader, desc=f"Ep {epoch+1:02d}/{CFG['epochs']}")

        for imgs, real_targets in loop:
            imgs         = imgs.to(device, non_blocking=True)
            real_targets = real_targets.to(device, non_blocking=True)

            # ── Train Generator ────────────────────────────────────────
            opt_G.zero_grad(set_to_none=True)
            with autocast('cuda'):
                fake_targets = generator(imgs)

                pred_fake = discriminator(fake_targets)
                valid     = torch.ones_like(pred_fake)
                loss_adv  = criterion_GAN(pred_fake, valid)

                loss_content = criterion_content(fake_targets, real_targets)

                loss_G = CFG['lambda_adv'] * loss_adv + CFG['lambda_content'] * loss_content

            scaler.scale(loss_G).backward()
            # Gradient clipping: prevents GAN instability
            scaler.unscale_(opt_G)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            scaler.step(opt_G)

            # ── Train Discriminator ────────────────────────────────────
            opt_D.zero_grad(set_to_none=True)
            with autocast('cuda'):
                pred_real = discriminator(real_targets)
                loss_real = criterion_GAN(pred_real, torch.ones_like(pred_real))

                pred_fake_d = discriminator(fake_targets.detach())
                loss_fake   = criterion_GAN(pred_fake_d, torch.zeros_like(pred_fake_d))

                loss_D = 0.5 * (loss_real + loss_fake)

            scaler.scale(loss_D).backward()
            scaler.unscale_(opt_D)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            scaler.step(opt_D)

            scaler.update()

            running['G_content'] += loss_content.item()
            running['G_adv']     += loss_adv.item()
            running['D']         += loss_D.item()

            loop.set_postfix(
                G_ms=f"{loss_content.item():.4f}",
                G_adv=f"{loss_adv.item():.4f}",
                D=f"{loss_D.item():.4f}",
            )

        scheduler_G.step()
        scheduler_D.step()

        # ── Epoch-level MS-SSIM score ──────────────────────────────────
        val_msssim = compute_val_msssim(generator, loader, device, n_batches=30)
        n = len(loader)
        print(
            f"\n[Epoch {epoch+1:02d}] "
            f"G_content={running['G_content']/n:.4f}  "
            f"G_adv={running['G_adv']/n:.4f}  "
            f"D={running['D']/n:.4f}  "
            f"MS-SSIM={val_msssim:.5f}  "
            f"lr_G={scheduler_G.get_last_lr()[0]:.2e}\n"
        )

        # Save best model by MS-SSIM (the actual eval metric)
        if val_msssim > best_msssim:
            best_msssim = val_msssim
            ckpt_path = os.path.join(CFG['save_dir'], f"best_msssim_{val_msssim:.5f}.pth")
            # Unwrap compiled model if necessary
            state = (
                generator._orig_mod.state_dict()
                if hasattr(generator, '_orig_mod')
                else generator.state_dict()
            )
            torch.save(state, ckpt_path)
            print(f"  ★ New best MS-SSIM: {val_msssim:.5f} → saved to {ckpt_path}")

        # Always keep a latest checkpoint
        latest_state = (
            generator._orig_mod.state_dict()
            if hasattr(generator, '_orig_mod')
            else generator.state_dict()
        )
        torch.save(latest_state, os.path.join(CFG['save_dir'], 'latest.pth'))


if __name__ == '__main__':
    main()
