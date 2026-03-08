import os
import glob
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.amp import autocast, GradScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from scipy.ndimage import zoom
import segmentation_models_pytorch as smp
import cv2
from pytorch_msssim import MS_SSIM

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
CFG = {
    # Model
    'encoder':          'efficientnet-b4',
    'pretrained_unet':  'NewGan.pth',       # Warm-start path

    # Training
    'epochs':           40,
    'img_size':         256,
    'batch_size':       32,                 # A100 40GB handles this
    'grad_accum_steps': 4,                  # Effective batch = 128

    # LR
    'lr_G':             5e-5,
    'lr_D':             1e-4,
    'lr_swa':           1e-5,
    'swa_start_epoch':  30,                 # SWA kicks in at 75% training

    # Loss weights
    'lambda_adv':       0.05,
    'lambda_pixel':     1.0,                # MS-SSIM + L1
    'lambda_freq':      0.5,                # FFT frequency domain loss

    # Misc
    'label_smooth':     0.1,
    'num_workers':      8,
    'device':           'cuda' if torch.cuda.is_available() else 'cpu',
    'use_bf16':         True,               # A100 native bfloat16
    'pin_memory':       True,
    'save_dir':         './checkpoints',
}

os.makedirs(CFG['save_dir'], exist_ok=True)
print(f"Device : {CFG['device']}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    CFG['use_bf16'] = CFG['use_bf16'] and torch.cuda.is_bf16_supported()
print(f"AMP    : {'bfloat16' if CFG['use_bf16'] else 'float16'}")


# ─────────────────────────────────────────────
#  LOSSES
# ─────────────────────────────────────────────
class CombinedPixelLoss(nn.Module):
    """MS-SSIM + L1 — structural + pixel accuracy."""
    def __init__(self):
        super().__init__()
        self.msssim = MS_SSIM(data_range=1.0, size_average=True, channel=1, win_size=11)
        self.l1     = nn.L1Loss()

    def forward(self, pred, target):
        return 0.8 * (1 - self.msssim(pred, target)) + 0.2 * self.l1(pred, target)


class FrequencyLoss(nn.Module):
    """
    FFT-based loss in frequency domain.
    Directly penalises high-frequency texture errors that MS-SSIM smooths over.
    Magnitude loss recovers overall spectral energy; phase loss recovers fine structure.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # fft2 does not support bfloat16 — cast to float32 for this op only
        pred_f32   = pred.float()
        target_f32 = target.float()

        pred_fft   = torch.fft.fft2(pred_f32,   norm='ortho')
        target_fft = torch.fft.fft2(target_f32, norm='ortho')

        loss_mag   = F.l1_loss(torch.abs(pred_fft),   torch.abs(target_fft))
        loss_phase = F.l1_loss(torch.angle(pred_fft), torch.angle(target_fft))

        return loss_mag + 0.1 * loss_phase


# ─────────────────────────────────────────────
#  DATASET  —  Trilinear upsample to HF shape
# ─────────────────────────────────────────────
class MRIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir  = root_dir
        self.transform = transform

        lf_files = sorted(glob.glob(
            os.path.join(root_dir, 'train', 'low_field', '*.nii*')
        ))
        self.files   = [os.path.basename(f) for f in lf_files]
        self.samples = []
        for vol_idx, fname in enumerate(self.files):
            # We'll determine slice count after loading HF volume;
            # use 1..197 as a safe default (slice 0 and -1 need neighbours)
            for s in range(1, 198):
                self.samples.append((vol_idx, s, fname))

        self.cache = {}
        print(f"Dataset: {len(self.files)} volumes  →  {len(self.samples)} slices (pre-cache)")

    # ── Volume helpers ──────────────────────────
    @staticmethod
    def normalise(vol: np.ndarray) -> np.ndarray:
        vol = vol.astype(np.float32)
        mn, mx = vol.min(), vol.max()
        return (vol - mn) / (mx - mn + 1e-8)

    @staticmethod
    def trilinear_upsample(lf_vol: np.ndarray, hf_shape: tuple) -> np.ndarray:
        """
        Upsample LF volume to exactly match HF shape using trilinear
        interpolation (scipy zoom order=1).  Per-file and dynamic.
        """
        zf = hf_shape[0] / lf_vol.shape[0]
        yf = hf_shape[1] / lf_vol.shape[1]
        xf = hf_shape[2] / lf_vol.shape[2]
        return zoom(lf_vol, (zf, yf, xf), order=1)   # order=1 = trilinear

    # ── Cache loader ────────────────────────────
    def _load_pair(self, vol_idx: int, fname: str):
        lf_path = os.path.join(self.root_dir, 'train', 'low_field',  fname)
        hf_path = os.path.join(self.root_dir, 'train', 'high_field',
                               fname.replace('lowfield', 'highfield'))

        hf_vol = self.normalise(nib.load(hf_path).get_fdata())
        lf_raw = self.normalise(nib.load(lf_path).get_fdata())

        # Dynamic upsample: LF → exact HF shape
        lf_vol = self.trilinear_upsample(lf_raw, hf_vol.shape)

        self.cache[f'lf_{vol_idx}'] = lf_vol
        self.cache[f'hf_{vol_idx}'] = hf_vol

    # ── Dataset interface ───────────────────────
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vol_idx, slice_idx, fname = self.samples[idx]

        if f'lf_{vol_idx}' not in self.cache:
            self._load_pair(vol_idx, fname)

        lf_vol = self.cache[f'lf_{vol_idx}']
        hf_vol = self.cache[f'hf_{vol_idx}']

        # Guard against slice index exceeding this volume's depth
        max_s = lf_vol.shape[2] - 2
        slice_idx = min(slice_idx, max_s)

        # 2.5D input stack (H, W, 3) — three consecutive upsampled LF slices
        img_stack = lf_vol[:, :, slice_idx - 1 : slice_idx + 2]   # (H, W, 3)
        target    = hf_vol[:, :, slice_idx]                        # (H, W)

        if self.transform:
            t_exp     = np.expand_dims(target, axis=2)             # (H, W, 1)
            augmented = self.transform(image=img_stack, mask=t_exp)
            img_stack = augmented['image']                          # (3, H, W) tensor
            target    = augmented['mask'].permute(2, 0, 1)         # (1, H, W) tensor

        return img_stack, target


def clip_image(image, **kwargs):
    return np.clip(image, 0, 1)


train_aug = A.Compose([
    A.PadIfNeeded(min_height=CFG['img_size'], min_width=CFG['img_size'],
                  border_mode=cv2.BORDER_CONSTANT),
    A.RandomCrop(height=CFG['img_size'], width=CFG['img_size']),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.3),
    A.ElasticTransform(alpha=10, sigma=5, p=0.2),   # mild anatomical variation
    A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.3),
    A.GaussNoise(std_range=(0.001, 0.005), p=0.4),
    A.Lambda(image=clip_image, p=1.0),
    ToTensorV2()
])  # type: ignore


# ─────────────────────────────────────────────
#  DISCRIMINATOR  (Spectral Norm)
# ─────────────────────────────────────────────
class PatchDiscriminator(nn.Module):
    """PatchGAN with Spectral Normalization on every conv layer."""
    def __init__(self):
        super().__init__()

        def block(in_c, out_c, norm=True):
            layers = [nn.utils.spectral_norm(
                nn.Conv2d(in_c, out_c, 4, stride=2, padding=1)
            )]
            if norm:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(1,   64,  norm=False),   # 128×128
            *block(64,  128),               # 64×64
            *block(128, 256),               # 32×32
            *block(256, 512),               # 16×16
            nn.utils.spectral_norm(nn.Conv2d(512, 1, 3, padding=1))  # 16×16 map
        )

    def forward(self, img):
        return self.model(img)


# ─────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────
def main():
    device   = CFG['device']
    use_bf16 = CFG['use_bf16']
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    # ── Generator ──────────────────────────────
    generator = smp.Unet(
        encoder_name=CFG['encoder'],
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,
        activation='sigmoid',
        decoder_attention_type='scse',      # squeeze-excitation attention
    ).to(device)

    if os.path.exists(CFG['pretrained_unet']):
        print(f"Loading pre-trained generator: {CFG['pretrained_unet']}")
        state = torch.load(CFG['pretrained_unet'], map_location=device)
        try:
            generator.load_state_dict(state, strict=True)
            print("  → strict load OK")
        except RuntimeError:
            miss, unexp = generator.load_state_dict(state, strict=False)
            print(f"  → partial load: {len(miss)} missing, {len(unexp)} unexpected keys")
    else:
        print("WARNING: No pretrained weights — training from scratch")

    # ── Discriminator ──────────────────────────
    discriminator = PatchDiscriminator().to(device)

    # ── Optimisers ─────────────────────────────
    opt_G = torch.optim.AdamW(generator.parameters(),
                               lr=CFG['lr_G'], betas=(0.5, 0.999), weight_decay=1e-4)
    opt_D = torch.optim.AdamW(discriminator.parameters(),
                               lr=CFG['lr_D'], betas=(0.5, 0.999), weight_decay=1e-4)

    sched_G = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt_G, T_0=10, T_mult=2, eta_min=1e-6)
    sched_D = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt_D, T_0=10, T_mult=2, eta_min=1e-6)

    # SWA
    swa_model  = AveragedModel(generator)
    swa_sched  = SWALR(opt_G, swa_lr=CFG['lr_swa'])
    swa_active = False

    # ── Losses ─────────────────────────────────
    crit_GAN   = nn.MSELoss()                       # LSGAN
    crit_pixel = CombinedPixelLoss().to(device)
    crit_freq  = FrequencyLoss().to(device)

    # bf16 → no GradScaler needed (no fp16 overflow risk)
    scaler = GradScaler('cuda', enabled=not use_bf16)

    # ── Data ───────────────────────────────────
    dataset = MRIDataset(root_dir='/scratch/fr2471/ni_data', transform=train_aug)
    loader  = DataLoader(
        dataset,
        batch_size=CFG['batch_size'],
        shuffle=True,
        num_workers=CFG['num_workers'],
        pin_memory=CFG['pin_memory'],
        persistent_workers=True,
        prefetch_factor=2,
    )

    grad_accum        = CFG['grad_accum_steps']
    best_pixel_loss   = float('inf')

    print(f"\nTraining: {CFG['epochs']} epochs | "
          f"batch {CFG['batch_size']} × accum {grad_accum} = effective {CFG['batch_size']*grad_accum}\n")

    for epoch in range(CFG['epochs']):
        generator.train()
        discriminator.train()

        if epoch >= CFG['swa_start_epoch'] and not swa_active:
            swa_active = True
            print(f"\n[Epoch {epoch+1}] SWA activated")

        loop   = tqdm(loader, desc=f"Ep {epoch+1}/{CFG['epochs']}")
        sum_pixel = sum_freq = sum_adv = sum_D = 0.0
        n = 0

        opt_G.zero_grad()
        opt_D.zero_grad()

        for batch_idx, (imgs, real_targets) in enumerate(loop):
            imgs         = imgs.to(device, non_blocking=True)
            real_targets = real_targets.to(device, non_blocking=True)

            is_update = ((batch_idx + 1) % grad_accum == 0) or \
                        (batch_idx + 1 == len(loader))

            # ══ Generator ════════════════════════
            with autocast('cuda', dtype=amp_dtype):
                fake_targets = generator(imgs)                     # [0, 1] via sigmoid

                pred_fake = discriminator(fake_targets)
                valid     = torch.ones_like(pred_fake) * (1.0 - CFG['label_smooth'])
                loss_adv  = crit_GAN(pred_fake, valid)

                loss_pixel = crit_pixel(fake_targets, real_targets)
                loss_freq  = crit_freq(fake_targets, real_targets)

                loss_G = (CFG['lambda_adv']   * loss_adv   +
                          CFG['lambda_pixel'] * loss_pixel +
                          CFG['lambda_freq']  * loss_freq) / grad_accum

            if use_bf16:
                loss_G.backward()
            else:
                scaler.scale(loss_G).backward()

            if is_update:
                if use_bf16:
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
                    opt_G.step()
                else:
                    scaler.unscale_(opt_G)
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
                    scaler.step(opt_G)
                opt_G.zero_grad()

            # ══ Discriminator ════════════════════
            with autocast('cuda', dtype=amp_dtype):
                pred_real  = discriminator(real_targets)
                valid_real = torch.ones_like(pred_real) * (1.0 - CFG['label_smooth'])
                loss_real  = crit_GAN(pred_real, valid_real)

                pred_fake_d = discriminator(fake_targets.detach())
                fake_label  = torch.zeros_like(pred_fake_d) + CFG['label_smooth']
                loss_fake   = crit_GAN(pred_fake_d, fake_label)

                loss_D = 0.5 * (loss_real + loss_fake) / grad_accum

            if use_bf16:
                loss_D.backward()
            else:
                scaler.scale(loss_D).backward()

            if is_update:
                if use_bf16:
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                    opt_D.step()
                else:
                    scaler.unscale_(opt_D)
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                    scaler.step(opt_D)
                opt_D.zero_grad()
                if not use_bf16:
                    scaler.update()

            # Logging
            sum_pixel += loss_pixel.item()
            sum_freq  += loss_freq.item()
            sum_adv   += loss_adv.item()
            sum_D     += loss_D.item() * grad_accum
            n         += 1

            loop.set_postfix(
                pixel=f"{loss_pixel.item():.4f}",
                freq=f"{loss_freq.item():.4f}",
                adv=f"{loss_adv.item():.4f}",
                D=f"{loss_D.item()*grad_accum:.4f}",
            )

        # ── End of epoch ───────────────────────
        avg_pixel = sum_pixel / n
        avg_freq  = sum_freq  / n

        if swa_active:
            swa_model.update_parameters(generator)
            swa_sched.step()
        else:
            sched_G.step()
            sched_D.step()

        print(f"\nEp {epoch+1} | pixel={avg_pixel:.4f} | freq={avg_freq:.4f} | "
              f"adv={sum_adv/n:.4f} | D={sum_D/n:.4f}")

        # Save best
        if avg_pixel < best_pixel_loss:
            best_pixel_loss = avg_pixel
            ckpt = os.path.join(CFG['save_dir'],
                                f"best_G_ep{epoch+1}_{avg_pixel:.4f}.pth")
            torch.save(generator.state_dict(), ckpt)
            print(f"  ✓ Best model saved: {ckpt}")

        # Save latest always
        torch.save(generator.state_dict(),
                   os.path.join(CFG['save_dir'], 'latest_G.pth'))

    # ── SWA finalisation ───────────────────────
    if swa_active:
        print("\nUpdating SWA BatchNorm stats...")
        bn_loader = DataLoader(dataset, batch_size=CFG['batch_size'],
                               shuffle=True, num_workers=4)
        torch.optim.swa_utils.update_bn(bn_loader, swa_model, device=device)
        swa_path = os.path.join(CFG['save_dir'], 'swa_final_G.pth')
        torch.save(swa_model.module.state_dict(), swa_path)
        print(f"SWA model saved: {swa_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()

