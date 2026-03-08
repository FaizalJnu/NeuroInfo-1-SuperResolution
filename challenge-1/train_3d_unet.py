import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from monai.transforms.compose import Compose
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.utility.dictionary import EnsureChannelFirstd, EnsureTyped
from monai.transforms.spatial.dictionary import Resized
from monai.transforms.croppad.dictionary import RandSpatialCropd
from monai.transforms.intensity.dictionary import ScaleIntensityd
from monai.networks.nets.basic_unet import BasicUNet
from monai.data.dataset import CacheDataset
from monai.data.dataloader import DataLoader
from monai.inferers.utils import sliding_window_inference
from pytorch_msssim import MS_SSIM  # Differentiable MS-SSIM loss
import numpy as np

# --- CONFIGURATION ---
CONFIG = {
    "data_dir": "./data", 
    "seed": 42,
    "folds": 5,
    # A100 40GB allows much larger batches and patches
    "batch_size": 4,         # Increased from 2
    "patch_size": (192, 192, 192), # BIG upgrade (Solves MS-SSIM crash)
    "target_shape": (179, 221, 200),
    "lr": 1e-4,
    "epochs": 150,           # A100 trains fast, so we can do more epochs
    "device": "cuda"
}

# --- TRANSFORMS ---
# Key Step: We resize the Low-Field input to match the High-Field dimensions
# BEFORE the network sees it. The network learns to "de-blur" this interpolated volume.
train_transforms = Compose([
    LoadImaged(keys=["low", "high"]),
    EnsureChannelFirstd(keys=["low", "high"]),
    
    # Resample Low-Field to match High-Field geometry (Super-Resolution step 1)
    Resized(keys=["low"], spatial_size=CONFIG["target_shape"], mode="trilinear"),
    
    # Normalize intensities to 0-1 range (Critical for MS-SSIM)
    ScaleIntensityd(keys=["low", "high"], minv=0.0, maxv=1.0),
    
    # Extract random 3D patches for training to fit in VRAM
    RandSpatialCropd(
        keys=["low", "high"],
        roi_size=CONFIG["patch_size"],
        random_size=False
    ),
    EnsureTyped(keys=["low", "high"]),
])

val_transforms = Compose([
    LoadImaged(keys=["low", "high"]),
    EnsureChannelFirstd(keys=["low", "high"]),
    Resized(keys=["low"], spatial_size=CONFIG["target_shape"], mode="trilinear"),
    ScaleIntensityd(keys=["low", "high"], minv=0.0, maxv=1.0),
    EnsureTyped(keys=["low", "high"]),
])

# --- LOSS FUNCTION ---
# Combined Loss: 80% MS-SSIM (Structure) + 20% L1 (Pixel Accuracy)
# --- HIGH QUALITY LOSS (A100 ONLY) ---

class HybridLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        # No need to hack weights! 
        # Standard 5-scale MS-SSIM works because patch_size (192) > 160
        self.msssim = MS_SSIM(
            data_range=1.0, 
            size_average=True, 
            channel=1
        ).to(device)
        
        self.l1 = nn.L1Loss()

    def forward(self, preds, target):
        # 84% Structure (MS-SSIM), 16% Pixel Accuracy (L1) - Standard ratio
        return 0.84 * (1 - self.msssim(preds, target)) + 0.16 * self.l1(preds, target)
# --- MAIN TRAINING LOOP ---
def run_training():
    print(f"Running on: {CONFIG['device']} | VRAM Limit: ~12GB Optimized")
    
    # 1. Prepare Data List
    # Assuming filenames match: subject001_low.nii.gz and subject001_high.nii.gz
    low_field_files = sorted(glob.glob(os.path.join(CONFIG["data_dir"], "train", "low_field", "*.nii*")))
    # print(f"path to low-field files: {low_field_files}")
    high_field_files = sorted(glob.glob(os.path.join(CONFIG["data_dir"], "train", "high_field", "*.nii*")))
    # print(f"path to high-field files: {high_field_files}")

    data_dicts = [
        {"low": low, "high": high} for low, high in zip(low_field_files, high_field_files)
    ]
    
    # 2. Cross Validation Setup
    kf = KFold(n_splits=CONFIG["folds"], shuffle=True, random_state=CONFIG["seed"])
    
    X_indices = np.arange(len(data_dicts))
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_indices)):
        print(f"\n--- STARTING FOLD {fold+1}/{CONFIG['folds']} ---")
        
        train_files = [data_dicts[i] for i in train_idx]
        val_files = [data_dicts[i] for i in val_idx]
        
        # CacheDataset loads data into RAM (32GB is plenty for 18 subjects) to speed up training
        train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
        val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=2)
        
        train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False) # Batch 1 for validation (full volume)

        # 3. Model Definition
        # BasicUNet is VRAM efficient. 
        model = BasicUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            features=(32, 64, 128, 256, 512, 32), # Lightweight features
        ).to(CONFIG["device"])
        
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
        loss_function = HybridLoss(CONFIG["device"])
        scaler = torch.amp.GradScaler('cuda') # Mixed Precision for VRAM saving

        best_metric = -1
        
        # 4. Training Epochs
        for epoch in range(CONFIG["epochs"]):
            model.train()
            epoch_loss = 0
            step = 0
            
            for batch_data in train_loader:
                inputs, targets = batch_data["low"].to(CONFIG["device"]), batch_data["high"].to(CONFIG["device"])
                
                optimizer.zero_grad()
                
                # AMP Context (Automatic Mixed Precision)
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = loss_function(outputs, targets)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += loss.item()
                step += 1
            
            print(f"Epoch {epoch+1}/{CONFIG['epochs']} - Train Loss: {epoch_loss/step:.4f}")

            # 5. Validation (Every 5 epochs to save time)
            if (epoch + 1) % 5 == 0:
                model.eval()
                val_msssim_accum = 0
                with torch.no_grad():
                    for val_data in val_loader:
                        val_inputs, val_targets = val_data["low"].to(CONFIG["device"]), val_data["high"].to(CONFIG["device"])
                        
                        # Sliding Window Inference
                        # We cannot feed the full volume at once. We infer patch-by-patch and stitch.
                        val_outputs = sliding_window_inference(
                            inputs=val_inputs, 
                            roi_size=CONFIG["patch_size"], 
                            sw_batch_size=4, 
                            predictor=model,
                            overlap=0.25 # Overlap reduces edge artifacts
                        )
                        
                        val_loss = loss_function(val_outputs, val_targets)
                        val_loss_accum += val_loss.item()

                        # 2. Calculate PURE MS-SSIM Score (For the "Best Model" check)
                        # We call the msssim component directly. 
                        # This returns a score between 0 and 1 (Higher is Better).
                        msssim_score = loss_function.msssim(val_outputs, val_targets)
                        val_msssim_accum += msssim_score.item()
                
                # Averages
                avg_val_loss = val_loss_accum / len(val_loader)
                avg_val_msssim = val_msssim_accum / len(val_loader)
                
                print(f"Epoch {epoch+1} Validation | Loss: {avg_val_loss:.4f} | MS-SSIM Score: {avg_val_msssim:.4f}")
                
                if avg_val_msssim > best_metric:
                    print(f"  >>> Improved MS-SSIM ({best_metric:.4f} -> {avg_val_msssim:.4f}). Saving model...")
                    best_metric = avg_val_msssim
                    torch.save(model.state_dict(), f"best_model_fold{fold}.pth")
                    print(f">>> Saved {save_path}")
    
                    # FORCE DISK WRITE (Crucial for HPC/Jupyter)
                    # Sometimes HPC storage lags; this forces the file to be written immediately.
                    os.sync()

if __name__ == "__main__":
    run_training()