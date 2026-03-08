import os
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def audit_dataset(lf_dir='train/low_field', hf_dir='train/high_field', output_dir='qa_reports'):
    os.makedirs(output_dir, exist_ok=True)
    
    lf_files = sorted(glob.glob(os.path.join(lf_dir, '*.nii*')))
    
    print(f"Found {len(lf_files)} patients for audit.\n")
    
    for lf_path in lf_files:
        filename = os.path.basename(lf_path)
        # Adjust string replacement based on your exact naming convention
        hf_path = os.path.join(hf_dir, filename.replace('lowfield', 'highfield')) 
        
        if not os.path.exists(hf_path):
            print(f"❌ Missing matching HF file for: {filename}")
            continue
            
        # 1. Load Data
        lf_vol = nib.load(lf_path).get_fdata()
        hf_vol = nib.load(hf_path).get_fdata()
        
        # 2. Shape Mismatch Check
        if lf_vol.shape != hf_vol.shape:
            print(f"❌ SHAPE MISMATCH [{filename}]: LF {lf_vol.shape} vs HF {hf_vol.shape}")
            continue
            
        # 3. Normalization/Outlier Check
        lf_max, lf_99 = np.max(lf_vol), np.percentile(lf_vol, 99.9)
        hf_max, hf_99 = np.max(hf_vol), np.percentile(hf_vol, 99.9)
        
        lf_spike_ratio = lf_max / (lf_99 + 1e-8)
        hf_spike_ratio = hf_max / (hf_99 + 1e-8)
        
        print(f"--- QA for {filename} ---")
        if lf_spike_ratio > 1.5 or hf_spike_ratio > 1.5:
            print(f"   ⚠️ WARNING: Severe outliers detected.")
            print(f"   LF Max: {lf_max:.1f} | LF 99.9%: {lf_99:.1f} (Ratio: {lf_spike_ratio:.2f})")
            print(f"   HF Max: {hf_max:.1f} | HF 99.9%: {hf_99:.1f} (Ratio: {hf_spike_ratio:.2f})")
            print(f"   Action: Switch from min-max to 1st-99th percentile clipping.")
        else:
            print("   ✅ Intensity distribution looks safe for min-max.")

        # 4. Generate QA Visualization
        # Normalize to [0,1] just for visualization
        lf_vis = np.clip((lf_vol - np.min(lf_vol)) / (lf_99 - np.min(lf_vol) + 1e-8), 0, 1)
        hf_vis = np.clip((hf_vol - np.min(hf_vol)) / (hf_99 - np.min(hf_vol) + 1e-8), 0, 1)
        
        # Grab the middle slice of the Z axis
        z_mid = lf_vol.shape[2] // 2
        lf_slice = lf_vis[:, :, z_mid]
        hf_slice = hf_vis[:, :, z_mid]
        
        # Create False-Color Composite (Red=LF, Green=HF)
        rgb_composite = np.zeros((*lf_slice.shape, 3))
        rgb_composite[..., 0] = lf_slice  # Red channel
        rgb_composite[..., 1] = hf_slice  # Green channel
        
        # Create Checkerboard
        checkerboard = np.copy(lf_slice)
        grid_size = 15
        for i in range(0, lf_slice.shape[0], grid_size):
            for j in range(0, lf_slice.shape[1], grid_size):
                if (i // grid_size + j // grid_size) % 2 == 0:
                    checkerboard[i:i+grid_size, j:j+grid_size] = hf_slice[i:i+grid_size, j:j+grid_size]

        # Plotting
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(lf_slice, cmap='gray')
        axes[0].set_title("Low Field (Original)")
        axes[1].imshow(hf_slice, cmap='gray')
        axes[1].set_title("High Field (Target)")
        axes[2].imshow(rgb_composite)
        axes[2].set_title("False Color Overlay\n(Yellow = Good, Red/Green edges = Misaligned)")
        axes[3].imshow(checkerboard, cmap='gray')
        axes[3].set_title("Checkerboard\n(Look for broken anatomy lines)")
        
        for ax in axes:
            ax.axis('off')
            
        save_path = os.path.join(output_dir, f"{filename.replace('.nii.gz', '')}_QA.png")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()

if __name__ == "__main__":
    audit_dataset()
    print("\nAudit complete! Check the 'qa_reports' folder for visual overlays.")