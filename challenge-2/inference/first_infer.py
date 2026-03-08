import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import os
from tqdm import tqdm
from torch.amp import autocast, GradScaler #type: ignore
import torch.nn as nn
import math
from scipy.ndimage import gaussian_filter1d

class RobustSessionScaler:
    def __init__(self, eps=1e-8):
        self.medians = None
        self.mads = None
        self.eps = eps

    def fit_transform(self, sbp_data, mask=None, smooth_sigma=2.0):
        working_data = sbp_data.copy().astype(float)
        if mask is not None:
            working_data[mask] = np.nan
            
        self.medians = np.nanmedian(working_data, axis=0)
        abs_dev = np.abs(working_data - self.medians)
        self.mads = np.nanmedian(abs_dev, axis=0) * 1.4826
        self.mads[self.mads < self.eps] = 1.0 
        
        normalized_sbp = (working_data - self.medians) / self.mads
        
        # Optional Smoothing
        smoothed_sbp = np.zeros_like(normalized_sbp)
        for c in range(96):
            valid_idx = ~np.isnan(normalized_sbp[:, c])
            if np.any(valid_idx):
                smoothed_sbp[valid_idx, c] = gaussian_filter1d(normalized_sbp[valid_idx, c], sigma=smooth_sigma)
        
        if mask is not None:
            smoothed_sbp[mask] = 0.0
            
        return smoothed_sbp

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, 1, d_model)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].transpose(0, 1) # type:ignore
        return self.dropout(x)

class NeuralReconstructor(nn.Module):
    def __init__(self, sbp_dim=96, kin_dim=4, d_model=128, nhead=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(sbp_dim + kin_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, 
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, sbp_dim)

    def forward(self, sbp, kinematics):
        x = torch.cat([sbp, kinematics], dim=-1)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return self.output_proj(x)
    
def generate_submission_with_ttt(test_dir, test_mask_csv_path, checkpoint_path, output_path="submission_ttt.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Test-Time Training (TTT) on: {device}")
    
    # Kaggle test set coordinates
    sub_df = pd.read_csv(test_mask_csv_path)
    test_sessions = sub_df['session_id'].unique()
    
    # TTT Hyperparameters
    SEQ_LEN = 100       # Match your new training sequence length!
    TTT_EPOCHS = 2      # Just 1 or 2 epochs. We want to gently nudge the manifold, not shatter it.
    TTT_LR = 1e-5       # Very tiny learning rate for fine-tuning
    NUM_TTT_MASKED = 15 # Out of the 66 visible channels, hide 15 for self-supervision
    
    predictions_dict = {}
    
    for sess_id in tqdm(test_sessions, desc="Evaluating Sessions"):
        # ---------------------------------------------------------
        # 1. THE RESET RULE: Reload the pristine base model
        # ---------------------------------------------------------
        model = NeuralReconstructor(d_model=128, num_layers=4).to(device)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Optimizer specifically for this session's TTT
        optimizer = optim.AdamW(model.parameters(), lr=TTT_LR, weight_decay=1e-4)
        scaler = GradScaler('cuda')
        criterion = nn.MSELoss()
        
        # Load this session's data
        sbp_path = os.path.join(test_dir, f"{sess_id}_sbp_masked.npy")
        mask_path = os.path.join(test_dir, f"{sess_id}_mask.npy")
        kin_path = os.path.join(test_dir, f"{sess_id}_kinematics.npy")
        
        sbp_masked = np.load(sbp_path)
        actual_mask = np.load(mask_path)
        kinematics = np.load(kin_path)
        
        scaler_obj = RobustSessionScaler()
        sbp_norm = scaler_obj.fit_transform(sbp_masked, mask=actual_mask)
        
        sbp_tensor = torch.FloatTensor(sbp_norm).unsqueeze(0).to(device)
        kin_tensor = torch.FloatTensor(kinematics).unsqueeze(0).to(device)
        actual_mask_tensor = torch.BoolTensor(actual_mask).unsqueeze(0).to(device)
        
        session_length = sbp_tensor.shape[1]
        
        # ---------------------------------------------------------
        # 2. TEST-TIME TRAINING (Self-Supervision on the Test Set)
        # ---------------------------------------------------------
        model.train() # Turn ON training mode for TTT
        
        for epoch in range(TTT_EPOCHS):
            # We will just do one massive stride through the session for TTT to save time
            stride = SEQ_LEN 
            for start in range(0, session_length - SEQ_LEN + 1, stride):
                end = start + SEQ_LEN
                
                sbp_chunk = sbp_tensor[:, start:end, :]
                kin_chunk = kin_tensor[:, start:end, :]
                mask_chunk = actual_mask_tensor[:, start:end, :] # The real 30 masked channels
                
                # --- Smart TTT Masking ---
                # We need to pick NUM_TTT_MASKED channels that are NOT ALREADY MASKED
                rand_vals = torch.rand(1, SEQ_LEN, 96, device=device)
                
                # Force already-masked channels to -1.0 so they are never picked by topk
                rand_vals[mask_chunk] = -1.0 
                
                _, ttt_masked_indices = torch.topk(rand_vals, NUM_TTT_MASKED, dim=2)
                
                ttt_mask = torch.zeros_like(mask_chunk, dtype=torch.bool, device=device)
                ttt_mask.scatter_(2, ttt_masked_indices, True)
                
                # Corrupt the input further (Real Mask + TTT Mask)
                sbp_corrupted = sbp_chunk.clone()
                sbp_corrupted[ttt_mask] = 0.0 
                # (Note: sbp_corrupted already has 0.0 where mask_chunk is True from our preprocessing)
                
                optimizer.zero_grad()
                with autocast('cuda'):
                    pred_chunk = model(sbp_corrupted, kin_chunk)
                    # ONLY compute loss on the channels we artificially hid for TTT
                    loss = criterion(pred_chunk[ttt_mask], sbp_chunk[ttt_mask])
                
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()

        # ---------------------------------------------------------
        # 3. ACTUAL INFERENCE (Predicting the real missing values)
        # ---------------------------------------------------------
        model.eval() # Turn OFF training mode for actual inference
        
        session_preds_norm = torch.zeros_like(sbp_tensor)
        prediction_counts = torch.zeros_like(sbp_tensor)
        
        # Overlapping inference stride for smooth predictions
        stride = SEQ_LEN // 2
        
        with torch.no_grad():
            for start in range(0, session_length - SEQ_LEN + 1, stride):
                end = start + SEQ_LEN
                sbp_chunk = sbp_tensor[:, start:end, :]
                kin_chunk = kin_tensor[:, start:end, :]
                
                with autocast('cuda'):
                    pred_chunk = model(sbp_chunk, kin_chunk)
                    
                session_preds_norm[:, start:end, :] += pred_chunk
                prediction_counts[:, start:end, :] += 1
                
            # Trailing edge catch
            if start + SEQ_LEN < session_length:
                start = session_length - SEQ_LEN
                end = session_length
                sbp_chunk = sbp_tensor[:, start:end, :]
                kin_chunk = kin_tensor[:, start:end, :]
                
                with autocast('cuda'):
                    pred_chunk = model(sbp_chunk, kin_chunk)
                
                session_preds_norm[:, start:end, :] += pred_chunk
                prediction_counts[:, start:end, :] += 1
                
        # Average, Inverse Transform, and Extract
        session_preds_norm = session_preds_norm / torch.clamp(prediction_counts, min=1.0)
        session_preds_norm = session_preds_norm.squeeze(0).cpu().numpy()
        
        session_preds_raw = (session_preds_norm * scaler_obj.mads) + scaler_obj.medians
        
        sess_df = sub_df[sub_df['session_id'] == sess_id]
        for _, row in sess_df.iterrows():
            samp_id = int(row['sample_id'])
            t_bin = int(row['time_bin'])
            ch = int(row['channel']) 
            predictions_dict[samp_id] = session_preds_raw[t_bin, ch]

    # Save format
    print("\nFormatting final submission file...")
    sub_df['predicted_sbp'] = sub_df['sample_id'].map(predictions_dict)
    final_sub = sub_df[['sample_id', 'predicted_sbp']]
    final_sub.to_csv(output_path, index=False)
    print(f"Success! TTT Submission saved to '{output_path}'.")

# --- Execution ---
# Adjust these paths to your Windows folder structure
TEST_DIR = r'C:\Coding\NeuroInfo-1-SuperResolution\challenge-2\long-term-intracortical-neural-activity-decoding-part-1-cs-gy-9223\kaggle_data\test'
TEST_MASK_CSV = r'C:\Coding\NeuroInfo-1-SuperResolution\challenge-2\long-term-intracortical-neural-activity-decoding-part-1-cs-gy-9223\kaggle_data\test_mask.csv'
CHECKPOINT = r'C:\Coding\NeuroInfo-1-SuperResolution\challenge-2\training\bci_checkpoint.pth'
OUTPUT_CSV = 'submission_bci_first_epoch50.csv'

generate_submission_with_ttt(TEST_DIR, TEST_MASK_CSV, CHECKPOINT, OUTPUT_CSV)