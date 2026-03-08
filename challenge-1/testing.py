import torch
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import os
import glob
import segmentation_models_pytorch as smp
from scipy.ndimage import zoom

# CONFIG
TEST_FILE = sorted(glob.glob('test/low_field/*.nii*'))[0] # Just check the first test brain
MODEL_SINGLE = 'unet_25d_epoch15.pth'  # Your good model
MODEL_FOLD0 = 'unet_fold4.pth'         # One of your CV models

def load_model(path):
    model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1, activation='sigmoid').cuda()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def prepare_input(path):
    # Load and normalize
    vol = nib.load(path).get_fdata()
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    # Upsample
    z_fac = 179/vol.shape[0]; y_fac = 221/vol.shape[1]; x_fac = 200/vol.shape[2]
    vol = zoom(vol, (z_fac, y_fac, x_fac), order=3)
    # Take center slice (z=100) and neighbors
    idx = 100
    stack = np.dstack([vol[:,:,idx-1], vol[:,:,idx], vol[:,:,idx+1]])
    # Pad to 256x256 (Center padding manual to be safe)
    pad_stack = np.zeros((256, 256, 3), dtype=np.float32)
    # 256 - 179 = 77 (top 38, bot 39)
    # 256 - 221 = 35 (left 17, right 18)
    pad_stack[38:38+179, 17:17+221, :] = stack
    
    # To Tensor (B, C, H, W)
    tensor = torch.from_numpy(pad_stack).permute(2,0,1).unsqueeze(0).cuda()
    return tensor

# Run
inp = prepare_input(TEST_FILE)
m1 = load_model(MODEL_SINGLE)
m2 = load_model(MODEL_FOLD0)

with torch.no_grad():
    out1 = m1(inp).cpu().numpy()[0,0,:,:]
    out2 = m2(inp).cpu().numpy()[0,0,:,:]

# Plot
plt.figure(figsize=(10,5))
plt.subplot(1,3,1); plt.title("Single Model (0.51)"); plt.imshow(out1, cmap='gray')
plt.subplot(1,3,2); plt.title("CV Fold 0 (0.26?)"); plt.imshow(out2, cmap='gray')
plt.subplot(1,3,3); plt.title("Difference"); plt.imshow(np.abs(out1-out2), cmap='hot')
plt.savefig("debug_comparison.png")
print("Saved debug_comparison.png - Open it and look!")