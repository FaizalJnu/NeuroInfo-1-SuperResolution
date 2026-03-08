import os
import glob
import numpy as np
import torch
import segmentation_models_pytorch as smp
from scipy.ndimage import zoom
from tqdm import tqdm

# Import the competition's utility functions
from extract_slices import create_submission_df, load_nifti

# --- CONFIGURATION ---
CFG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model_path': 'GAN_EMA_epoch_best_0.0492.pth', # <-- Point this to your best EMA weights
    'test_dir': 'test/low_field',
    'out_file': 'submission_final.csv',
    'target_shape': (179, 221, 200), # X, Y, Z
    'pad_size': 256,                 # Model input size
    'in_channels': 5
}

def upsample_lf(vol):
    """Upsample the 64mT volume to 3T dimensions."""
    z, y, x = CFG['target_shape'][0] / vol.shape[0], \
              CFG['target_shape'][1] / vol.shape[1], \
              CFG['target_shape'][2] / vol.shape[2]
    return zoom(vol, (z, y, x), order=3)

def pad_for_inference(vol):
    """
    1. Pads X and Y to 256x256 to match training dimensions.
    2. Pads Z by 2 on each side (edge replication) to allow 5-slice context at the boundaries.
    """
    pad_x = CFG['pad_size'] - vol.shape[0] # 256 - 179 = 77
    pad_y = CFG['pad_size'] - vol.shape[1] # 256 - 221 = 35
    
    # Calculate symmetrical padding
    px_left, px_right = pad_x // 2, pad_x - (pad_x // 2)
    py_top, py_bottom = pad_y // 2, pad_y - (pad_y // 2)
    
    # Z gets 2 slices of padding on both ends using 'edge' mode so it doesn't feed zeros to the model
    padded_vol = np.pad(
        vol, 
        ((px_left, px_right), (py_top, py_bottom), (2, 2)), 
        mode='edge'
    )
    
    return padded_vol, (px_left, py_top)

@torch.no_grad()
def process_volume(model, vol_path):
    """Run full inference on a single low-field NIfTI volume."""
    # 1. Load and Normalize
    vol = load_nifti(vol_path)
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    vol = vol.astype(np.float32)
    
    # 2. Upsample to 179x221x200
    vol_up = upsample_lf(vol)
    
    # 3. Pad Spatial and Z-dimensions
    vol_padded, (px_left, py_top) = pad_for_inference(vol_up)
    
    # 4. Prepare output array (179 x 221 x 200)
    out_vol = np.zeros(CFG['target_shape'], dtype=np.float32)
    
    # 5. Predict slice by slice
    for z in range(CFG['target_shape'][2]):
        # The padded volume has 2 extra slices on each end, so Z in original maps to Z+2 in padded.
        # We extract Z to Z+5 (which is 5 slices centered on Z+2)
        z_pad = z + 2
        img_stack = vol_padded[:, :, z_pad-2 : z_pad+3] 
        
        # Convert to tensor: [1, 5, 256, 256]
        x_tensor = torch.from_numpy(img_stack).permute(2, 0, 1).unsqueeze(0).to(CFG['device'])
        
        # Forward pass
        pred = model(x_tensor)
        pred = pred.squeeze().cpu().numpy() # Shape: [256, 256]
        
        # Crop back to 179x221
        pred_cropped = pred[px_left : px_left + CFG['target_shape'][0], 
                            py_top : py_top + CFG['target_shape'][1]]
        
        out_vol[:, :, z] = np.clip(pred_cropped, 0, 1)
        
    return out_vol

def main():
    print("Loading Model...")
    # Initialize the exact same architecture used in training
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4", 
        in_channels=CFG['in_channels'], 
        classes=1, 
        activation='sigmoid'
    ).to(CFG['device'])
    
    # Load the EMA weights for the smoothest perceptual results
    model.load_state_dict(torch.load(CFG['model_path'], map_location=CFG['device']))
    model.eval()
    
    # Find test files (sample_019 to sample_023)
    test_files = sorted(glob.glob(os.path.join(CFG['test_dir'], '*.nii*')))
    
    if not test_files:
        raise FileNotFoundError(f"No NIfTI files found in {CFG['test_dir']}. Check your paths!")

    predictions = {}
    
    print(f"Starting inference on {len(test_files)} volumes...")
    for fpath in tqdm(test_files, desc="Processing Volumes"):
        # Extract sample ID, e.g., 'sample_019' from 'sample_019_lowfield.nii.gz'
        fname = os.path.basename(fpath)
        sample_id = fname.split('_lowfield')[0] 
        
        # Predict the 3D high-field volume
        pred_vol = process_volume(model, fpath)
        predictions[sample_id] = pred_vol
        
    print("\nEncoding to Base64 and generating submission.csv...")
    # This utilizes the competition's utility file
    submission_df = create_submission_df(predictions)
    submission_df.to_csv(CFG['out_file'], index=False)
    
    print(f"Done! Saved {len(submission_df)} rows to {CFG['out_file']}.")

if __name__ == '__main__':
    main()