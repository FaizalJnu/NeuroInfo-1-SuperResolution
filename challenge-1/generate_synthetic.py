import torchio as tio
import os
import glob
from tqdm import tqdm
import multiprocessing

# --- CONFIGURATION ---
# Point this to your downloaded folder
SOURCE_DIR = 'IXI-T1'  
# Where the new pairs will be saved
OUTPUT_DIR = 'train_synthetic'             
# Exact dimensions required by the competition
TARGET_SHAPE = (179, 221, 200)             

def process_subject(nii_path):
    # Get the filename ID (e.g., "IXI002-Guys-0828-T1")
    subject_id = os.path.basename(nii_path).split('.')[0]
    
    # 1. Load the High Quality IXI Brain
    subject = tio.Subject(
        mri=tio.ScalarImage(nii_path),
    )
    
    # 2. Preprocessing (Make IXI look like Competition Ground Truth)
    # We resize it to match the competition's 3T Geometry exactly
    preprocess = tio.Compose([
        # Standardize orientation to RAS (Right, Anterior, Superior)
        tio.ToCanonical(), 
        # Resample to 1mm isotropic first to standardize resolution
        tio.Resample(1.0), 
        # Crop or Pad to the exact competition target dimensions
        tio.CropOrPad(TARGET_SHAPE),
        # Normalize intensities to 0-1 range
        tio.RescaleIntensity((0, 1))
    ])
    
    try:
        clean_subject = preprocess(subject)
    except Exception as e:
        # Skip files that are corrupt or weird shapes
        return 
        
    # 3. Degradation Pipeline (Simulate the 64mT Scanner)
    # This turns the "Clean" brain into a "Fake Low-Field" brain
    degrade = tio.Compose([
        # A. Simulate Low Resolution (Downsample then Upsample)
        # We downsample to ~2mm voxel size to mimic the blur, then back up
        tio.Resample(2.0), 
        tio.Resample(target=clean_subject.mri), 
        
        # B. Simulate Rician Noise (Specific to MRI)
        # Competition data is VERY noisy, so we use a high noise level
        tio.RandomNoise(p=1.0, std=(0.02, 0.05)), 
        
        # C. Simulate Field Inhomogeneity (The "Bias Field")
        # This creates the "bright center, dark corners" effect seen in 64mT
        tio.RandomBiasField(p=1.0, coefficients=(0.2, 0.5)),
        
        # D. Gamma (Intensity non-linearity)
        tio.RandomGamma(p=0.5, log_gamma=(-0.3, 0.3))
    ])
    
    # Generate the fake input
    dirty_subject = degrade(clean_subject)
    
    # 4. Save the Pair
    # We create the exact folder structure your train.py expects
    lf_dir = os.path.join(OUTPUT_DIR, 'low_field')
    hf_dir = os.path.join(OUTPUT_DIR, 'high_field')
    
    # Ensure directories exist (multiprocessing safe-ish)
    os.makedirs(lf_dir, exist_ok=True)
    os.makedirs(hf_dir, exist_ok=True)
    
    # Save Fake Low Field (Input)
    dirty_subject.mri.save(os.path.join(lf_dir, f"{subject_id}_lowfield.nii.gz"))
    # Save Real High Field (Target)
    clean_subject.mri.save(os.path.join(hf_dir, f"{subject_id}_highfield.nii.gz"))

def main():
    # Find all NIfTI files
    files = sorted(glob.glob(os.path.join(SOURCE_DIR, '*.nii*')))
    
    if len(files) == 0:
        print(f"ERROR: No files found in {SOURCE_DIR}")
        print("Please check the folder name and ensure .nii.gz files are inside.")
        return

    print(f"Found {len(files)} IXI brains. Starting synthesis...")
    print(f"Target Shape: {TARGET_SHAPE}")
    print(f"Saving to: {OUTPUT_DIR}")
    
    # Use multiprocessing to speed this up (it's CPU heavy)
    # Windows users: If this crashes, change processes=1
    num_workers = min(4, os.cpu_count())
    pool = multiprocessing.Pool(processes=num_workers) 
    
    # Run the processing
    list(tqdm(pool.imap(process_subject, files), total=len(files)))
    
    pool.close()
    pool.join()
    
    print(f"\nDone! Synthetic data generated.")

if __name__ == '__main__':
    main()