import os
import glob
import nibabel as nib
import nibabel.processing
from tqdm import tqdm

def align_and_resample(lf_dir='train/low_field', hf_dir='train/high_field', out_dir='train/low_field_aligned'):
    """
    Resamples Low-Field NIfTI files to perfectly match the physical space, 
    voxel spacing, and dimensions of the High-Field target files.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    lf_files = sorted(glob.glob(os.path.join(lf_dir, '*.nii*')))
    print(f"Aligning {len(lf_files)} volumes...")
    
    for lf_path in tqdm(lf_files):
        filename = os.path.basename(lf_path)
        hf_path = os.path.join(hf_dir, filename.replace('lowfield', 'highfield'))
        
        if not os.path.exists(hf_path):
            print(f"Missing HF pair for {filename}")
            continue
            
        # Load NIfTI objects (which contain both the array and the spatial affine matrix)
        lf_img = nib.load(lf_path)
        hf_img = nib.load(hf_path)
        
        # Resample LF to exactly match the HF grid (order=3 means cubic interpolation)
        lf_resampled_img = nib.processing.resample_from_to(lf_img, hf_img, order=3)
        
        # Save out the corrected file
        out_path = os.path.join(out_dir, filename)
        nib.save(lf_resampled_img, out_path)

if __name__ == "__main__":
    align_and_resample()
    print("Done! Your aligned Low-Field data is in 'train/low_field_aligned'.")