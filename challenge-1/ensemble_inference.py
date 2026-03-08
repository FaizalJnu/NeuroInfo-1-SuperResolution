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

    # 1. Find Test Files
    test_files = sorted(glob.glob(os.path.join(CFG['test_dir'], '*.nii*')))
    if not test_files:
        print(f"No test files found in '{CFG['test_dir']}'. Check your paths!")
        return
    
    # Create MONAI Dataset/Loader
    data_dicts = [{"low": f} for f in test_files]
    test_ds = Dataset(data=data_dicts, transform=infer_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # Initialize a dictionary to hold the SUM of all predictions
    # Shape initialized to zeros: (179, 221, 200)
    predictions_dict = {
        os.path.basename(f).split('_lowfield')[0]: np.zeros(CFG['target_shape'], dtype=np.float32) 
        for f in test_files
    }

    # 2. Initialize 3D U-Net Model
    model = BasicUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        features=(32, 64, 128, 256, 512, 32),
    ).to(CFG['device'])

    # 3. Loop through all 5 models (Ensemble)
    fold_paths = [f"best_model_fold{i}.pth" for i in range(5)]
    valid_folds = 0

    for fold_path in fold_paths:
        if not os.path.exists(fold_path):
            print(f"Warning: '{fold_path}' not found. Skipping...")
            continue

        print(f"\n--- Loading weights from {fold_path} ---")
        checkpoint = torch.load(fold_path, map_location=CFG['device'], weights_only=True)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        valid_folds += 1

        # 4. Inference Loop for the current model
        with torch.no_grad():
            for i, batch_data in enumerate(tqdm(test_loader, desc=f"Predicting (Fold {valid_folds})")):
                inputs = batch_data["low"].to(CFG['device'])
                
                fpath = test_files[i]
                sample_key = os.path.basename(fpath).split('_lowfield')[0] 
                
                outputs = sliding_window_inference(
                    inputs=inputs, 
                    roi_size=CFG["patch_size"], 
                    sw_batch_size=1, 
                    predictor=model,
                    overlap=0.25 
                )
                
                if hasattr(outputs, "as_tensor"):
                    outputs = outputs.as_tensor()
                    
                pred_vol = outputs[0, 0].cpu().numpy()
                
                # Accumulate the predictions (We will average them later)
                predictions_dict[sample_key] += pred_vol

    if valid_folds == 0:
        print("Error: No valid model files found. Exiting.")
        return

    # 5. Average and Post-Process
    print(f"\nAveraging predictions across {valid_folds} models...")
    for key in predictions_dict:
        # Divide by total number of models to get the mean
        predictions_dict[key] = predictions_dict[key] / valid_folds
        
        # Clip strictly to 0-1 range after averaging
        predictions_dict[key] = np.clip(predictions_dict[key], 0.0, 1.0)

    # 6. Create Submission
    print("Converting volumes to submission format...")
    submission_df = create_submission_df(predictions_dict)
    
    submission_df.to_csv('submission.csv', index=False)
    print("Done! Submission file saved as 'submission.csv'")

if __name__ == '__main__':
    main()