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
from monai.data.dataset import Dataset
from monai.data.dataloader import DataLoader
from monai.inferers.utils import sliding_window_inference
from pytorch_msssim import MS_SSIM  # Differentiable MS-SSIM loss
import numpy as np
from tqdm import tqdm


# CRITICAL: Import the competition's provided utility
from extract_slices import create_submission_df

# --- CONFIGURATION ---
CFG = {
    'test_dir': './data/test/low_field', # Update this to your local test directory
    'model_path': 'best_model_fold0.pth', # Update with your downloaded weights
    'target_shape': (179, 221, 200),
    'patch_size': (192, 192, 192), 
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# --- INFERENCE TRANSFORMS ---
# Notice we only process "low" here, as we don't have the "high" targets
infer_transforms = Compose([
    LoadImaged(keys=["low"]),
    EnsureChannelFirstd(keys=["low"]),
    
    # 1. Upsample 40 slices to 200 slices immediately
    Resized(keys=["low"], spatial_size=CFG["target_shape"], mode="trilinear"),
    
    # 2. Normalize to 0-1 (Crucial because the model was trained this way)
    ScaleIntensityd(keys=["low"], minv=0.0, maxv=1.0),
    EnsureTyped(keys=["low"]),
])

def main():
    print(f"Running Inference on: {CFG['device'].upper()}")
    
    # 1. Initialize 3D U-Net Model
    model = BasicUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        features=(32, 64, 128, 256, 512, 32),
    ).to(CFG['device'])
    
    # 2. Load Weights
    if not os.path.exists(CFG['model_path']):
        print(f"Error: Model file '{CFG['model_path']}' not found!")
        return

    print(f"Loading weights from {CFG['model_path']}...")
    checkpoint = torch.load(CFG['model_path'], map_location=CFG['device'], weights_only=True)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval() # Set to evaluation mode!

    # 3. Find Test Files
    test_files = sorted(glob.glob(os.path.join(CFG['test_dir'], '*.nii*')))
    if not test_files:
        print(f"No test files found in '{CFG['test_dir']}'. Check your paths!")
        return
    
    # Create MONAI Dataset/Loader
    data_dicts = [{"low": f} for f in test_files]
    test_ds = Dataset(data=data_dicts, transform=infer_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    predictions_dict = {}

    # 4. Inference Loop
    print("Starting 3D Inference...")
    with torch.no_grad(): # Disable gradients to save VRAM locally
        for i, batch_data in enumerate(tqdm(test_loader, desc="Predicting Volumes")):
            inputs = batch_data["low"].to(CFG['device'])
            
            # File path mapping to extract "sample_XXX"
            fpath = test_files[i]
            fname = os.path.basename(fpath)
            # Example: 'sample_019_lowfield.nii.gz' -> 'sample_019'
            sample_key = fname.split('_lowfield')[0] 
            
            # Use Sliding Window to prevent VRAM overflow on the RTX 5070 Ti
            outputs = sliding_window_inference(
                inputs=inputs, 
                roi_size=CFG["patch_size"], 
                sw_batch_size=1, # 1 window at a time
                predictor=model,
                overlap=0.25 
            )
            
            # 5. Post-Process Prediction
            # Strip MONAI MetaTensor wrapper if present
            if hasattr(outputs, "as_tensor"):
                outputs = outputs.as_tensor()
                
            # Move to CPU, convert to numpy, and remove Batch & Channel dims
            # Shape goes from (1, 1, 179, 221, 200) -> (179, 221, 200)
            pred_vol = outputs[0, 0].cpu().numpy()
            
            # Clip values strictly to 0-1 to avoid MS-SSIM out-of-bounds issues
            pred_vol = np.clip(pred_vol, 0.0, 1.0)
            
            predictions_dict[sample_key] = pred_vol

    # 6. Create Submission
    print("Converting volumes to submission format...")
    # The provided extract_slices utility handles the Base64 encoding
    submission_df = create_submission_df(predictions_dict)
    
    submission_df.to_csv('submission.csv', index=False)
    print("Done! Submission file saved as 'submission.csv'")

if __name__ == '__main__':
    main()