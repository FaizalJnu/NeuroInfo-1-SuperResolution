import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from scipy.ndimage import zoom
from tqdm import tqdm
import nibabel as nib
import cv2
from pytorch_msssim import MS_SSIM

# --- CONFIGURATION ---
CFG = {
    'batch_size': 32,
    'img_size': 256,
    'orig_shape': (179, 221), # Original spatial, before depth
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model_path': 'NewGan_epoch_best_0.0493_aligned.pth',
    
    # PATHS (Point these to where your paired data is)
    'val_lf_dir': 'train/low_field',   
    'val_hf_dir': 'train/high_field',  
    'num_val_samples': 5 # How many volumes to evaluate (set None for all)
}

# --- METRIC CALCULATOR ---
class MetricMonitor:
    def __init__(self):
        self.msssim_metric = MS_SSIM(data_range=1.0, size_average=True, channel=1)
        self.scores = []

    def update(self, pred_vol, target_vol):
        # Convert to torch tensors on device for fast MS-SSIM calc
        # Input shape: (Depth, H, W) -> (Depth, 1, H, W)
        p = torch.from_numpy(pred_vol).unsqueeze(1).to(CFG['device'])
        t = torch.from_numpy(target_vol).unsqueeze(1).to(CFG['device'])
        
        # Calculate per slice to avoid OOM on huge 3D tensors, or simple batching
        val = self.msssim_metric(p, t)
        self.scores.append(val.item())
        return val.item()

    def get_avg_score(self):
        return np.mean(self.scores)

# --- DATASETS & UTILS (Reused) ---
def upsample_volume(vol):
    # Standard Cubic Upsampling
    z_factor = 179 / vol.shape[0]
    y_factor = 221 / vol.shape[1]
    x_factor = 200 / vol.shape[2]
    return zoom(vol, (z_factor, y_factor, x_factor), order=3)

class InferenceDataset(Dataset):
    def __init__(self, lf_vol):
        self.vol = lf_vol
        self.transform = A.Compose([
            A.PadIfNeeded(min_height=CFG['img_size'], min_width=CFG['img_size'], 
                          border_mode=cv2.BORDER_CONSTANT, value=0),
            ToTensorV2()
        ])

    def __len__(self):
        return self.vol.shape[2]

    def __getitem__(self, idx):
        prev_idx = max(0, idx - 1)
        next_idx = min(self.vol.shape[2] - 1, idx + 1)
        img_stack = np.dstack([self.vol[:, :, prev_idx], self.vol[:, :, idx], self.vol[:, :, next_idx]])
        augmented = self.transform(image=img_stack)
        return augmented['image']

# --- PREDICTION ---
def predict_volume(model, vol_path):
    vol = nib.load(vol_path).get_fdata()
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    vol = vol.astype(np.float32)
    
    vol_upsampled = upsample_volume(vol)
    
    dataset = InferenceDataset(vol_upsampled)
    loader = DataLoader(dataset, batch_size=CFG['batch_size'], shuffle=False, num_workers=0)
    
    model.eval()
    predictions = []
    
    pad_h = (CFG['img_size'] - CFG['orig_shape'][0]) // 2
    pad_w = (CFG['img_size'] - CFG['orig_shape'][1]) // 2
    
    with torch.no_grad():
        for batch_imgs in tqdm(loader, leave=False):
            batch_imgs = batch_imgs.to(CFG['device'])
            
            # Simple inference (No TTA for speed in eval, add TTA if desired)
            preds = model(batch_imgs)
            preds = preds.cpu().numpy()
            
            for i in range(preds.shape[0]):
                p = preds[i, 0, :, :]
                p_cropped = p[pad_h : pad_h + 179, pad_w : pad_w + 221]
                p_cropped = np.clip(p_cropped, 0, 1)
                predictions.append(p_cropped)
                
    return np.stack(predictions, axis=2)

# --- VISUALIZATION ---
def save_comparison_plot(lf_vol, hf_vol, pred_vol, fname, score):
    """
    Saves a plot of 3 orthogonal views (Axial, Sagittal, Coronal)
    comparing Input (Upsampled) vs Prediction vs Ground Truth
    """
    # Pick a middle slice
    idx_ax = lf_vol.shape[2] // 2
    idx_sag = lf_vol.shape[1] // 2
    idx_cor = lf_vol.shape[0] // 2

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    plt.suptitle(f"Sample: {fname} | MS-SSIM: {score:.4f}", fontsize=16)
    
    # Headers
    cols = ['LF Input (Upsampled)', 'GAN Prediction', 'HF Ground Truth']
    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    # Axial (XY plane)
    axes[0, 0].imshow(lf_vol[:, :, idx_ax], cmap='gray')
    axes[0, 1].imshow(pred_vol[:, :, idx_ax], cmap='gray')
    axes[0, 2].imshow(hf_vol[:, :, idx_ax], cmap='gray')

    # Sagittal (YZ plane) - Rotated for viewing
    axes[1, 0].imshow(np.rot90(lf_vol[:, idx_sag, :]), cmap='gray')
    axes[1, 1].imshow(np.rot90(pred_vol[:, idx_sag, :]), cmap='gray')
    axes[1, 2].imshow(np.rot90(hf_vol[:, idx_sag, :]), cmap='gray')

    # Coronal (XZ plane)
    axes[2, 0].imshow(np.rot90(lf_vol[idx_cor, :, :]), cmap='gray')
    axes[2, 1].imshow(np.rot90(pred_vol[idx_cor, :, :]), cmap='gray')
    axes[2, 2].imshow(np.rot90(hf_vol[idx_cor, :, :]), cmap='gray')

    for ax in axes.flat:
        ax.axis('off')

    os.makedirs('eval_plots', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"eval_plots/comparison_{fname}.png")
    plt.close()

def plot_score_distribution(scores):
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of MS-SSIM Scores (Mean: {np.mean(scores):.4f})')
    plt.xlabel('MS-SSIM Score')
    plt.ylabel('Count')
    plt.axvline(np.mean(scores), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {np.mean(scores):.3f}')
    plt.legend()
    plt.savefig('eval_plots/score_distribution.png')
    plt.close()

# --- MAIN EVAL LOOP ---
def main():
    print(f"Loading GAN Generator: {CFG['model_path']}")
    model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1, activation='sigmoid').to(CFG['device'])
    model.load_state_dict(torch.load(CFG['model_path']))

    # Get file list
    lf_files = sorted(glob.glob(os.path.join(CFG['val_lf_dir'], '*.nii*')))
    
    if CFG['num_val_samples']:
        lf_files = lf_files[:CFG['num_val_samples']]
    
    metric_monitor = MetricMonitor()
    
    print(f"Starting Evaluation on {len(lf_files)} volumes...")

    for fpath in tqdm(lf_files):
        fname = os.path.basename(fpath)
        hf_path = os.path.join(CFG['val_hf_dir'], fname.replace('lowfield', 'highfield'))
        
        if not os.path.exists(hf_path):
            print(f"Skipping {fname}, HF counterpart not found.")
            continue
            
        # 1. Predict
        pred_vol = predict_volume(model, fpath)
        
        # 2. Load Ground Truth & Normalize
        hf_vol = nib.load(hf_path).get_fdata()
        hf_vol = (hf_vol - hf_vol.min()) / (hf_vol.max() - hf_vol.min() + 1e-8)
        hf_vol = hf_vol.astype(np.float32)
        
        # 3. Load LF (Just for visualization comparison)
        lf_vol_raw = nib.load(fpath).get_fdata()
        lf_vol_raw = (lf_vol_raw - lf_vol_raw.min()) / (lf_vol_raw.max() - lf_vol_raw.min() + 1e-8)
        lf_vol_upsampled = upsample_volume(lf_vol_raw)

        # 4. Calculate Score
        # Ensure shapes match (Crop HF if necessary due to upsampling rounding errors)
        min_d = min(pred_vol.shape[2], hf_vol.shape[2])
        min_h = min(pred_vol.shape[0], hf_vol.shape[0])
        min_w = min(pred_vol.shape[1], hf_vol.shape[1])
        
        pred_vol = pred_vol[:min_h, :min_w, :min_d]
        hf_vol = hf_vol[:min_h, :min_w, :min_d]
        lf_vol_upsampled = lf_vol_upsampled[:min_h, :min_w, :min_d]

        score = metric_monitor.update(pred_vol, hf_vol)
        
        # 5. Save Plot
        save_comparison_plot(lf_vol_upsampled, hf_vol, pred_vol, fname, score)

    print("-" * 30)
    print(f"FINAL AVERAGE MS-SSIM: {metric_monitor.get_avg_score():.5f}")
    print("-" * 30)
    
    plot_score_distribution(metric_monitor.scores)
    print("Graphs saved to /eval_plots folder.")

if __name__ == '__main__':
    main()