import os
import glob
import numpy as np
import nibabel as nib
from tqdm import tqdm

# CONFIG
SRC_DIR = 'train_synthetic'      # Your current .nii.gz folder
DST_DIR = 'train_synthetic_npy'  # New fast folder

def convert_file(path):
    # Determine new path structure
    rel_path = os.path.relpath(path, SRC_DIR)
    new_path = os.path.join(DST_DIR, rel_path).replace('.nii.gz', '.npy')
    
    # Skip if already done
    if os.path.exists(new_path): return

    # Ensure subfolders exist
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    
    try:
        # Load the slow compressed file
        img = nib.load(path)
        # Get raw data as float32 (standard for DL)
        vol = img.get_fdata().astype(np.float32)
        
        # Save as uncompressed numpy array
        np.save(new_path, vol)
    except Exception as e:
        print(f"Error converting {path}: {e}")

def main():
    # Recursive search for all .nii.gz files
    files = glob.glob(os.path.join(SRC_DIR, '**', '*.nii.gz'), recursive=True)
    
    if not files:
        print(f"No files found in {SRC_DIR}. Check path!")
        return

    print(f"Converting {len(files)} volumes to .npy for speed...")
    
    # Simple loop is safer for disk operations than multiprocessing here
    for f in tqdm(files):
        convert_file(f)
        
    print(f"\nDone! New fast data is in {DST_DIR}")

if __name__ == '__main__':
    main()