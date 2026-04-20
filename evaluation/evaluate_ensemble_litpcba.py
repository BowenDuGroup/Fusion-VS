#!/usr/bin/env python3
import os
import pickle
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Global Configuration
TARGET = "FEN1" 
FEAT_DIR = f"/root/autodl-tmp/AI4S1/LIT-PCBA/AVE_unbiased_Processed/{TARGET}/features"
MODEL_WEIGHTS = "/root/autodl-tmp/AI4S1/result_2/best_fusion_mlp.pth"

# 1. 4096d Fusion MLP Architecture
class FusionRegressor(nn.Module):
    def __init__(self, embed_dim=512, fp_dim=2048, dropout=0.0):
        super().__init__()
        in_dim = (embed_dim * 4) + fp_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 2048), nn.BatchNorm1d(2048), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2048, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, p_embed, m_embed_3d, m_embed_2d):
        product_feat = p_embed * m_embed_3d
        diff_feat = torch.abs(p_embed - m_embed_3d)
        
        # Assemble 3D features (512 * 4 = 2048)
        feat_3d = torch.cat([p_embed, m_embed_3d, product_feat, diff_feat], dim=-1)
        
        # Fuse 2D Morgan fingerprints (2048 + 2048 = 4096)
        x = torch.cat([feat_3d, m_embed_2d], dim=-1)
        return self.mlp(x).squeeze(-1)

# 2. Evaluation Metrics
def compute_ef(preds, labels, top_ratio=0.01):
    total_samples = len(labels)
    actual_actives_count = labels.sum().float()
    if actual_actives_count == 0: return 0.0
    
    top_k = max(1, int(total_samples * top_ratio))
    _, top_indices = torch.topk(preds, top_k)
    
    hits = labels[top_indices].sum().float()
    return ((hits / actual_actives_count) / top_ratio).item()

# 3. Core Evaluation Logic (Ensemble Docking)
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*50)
    print(f"Starting Ensemble Docking evaluation on LIT-PCBA target {TARGET}")
    print(f"Device: {device}")
    print("="*50)

    # --- A. Load pocket features ---
    pocket_pkl = f"{FEAT_DIR}/pocket_reps_512d.pkl"
    if not os.path.exists(pocket_pkl):
        print(f"Pocket features not found: {pocket_pkl}")
        return
        
    with open(pocket_pkl, 'rb') as f:
        p_names, p_reps = pickle.load(f)
    # Average the 6-fold outputs
    pocket_reps = np.mean(p_reps, axis=1) 
    print(f"Loaded {len(pocket_reps)} pocket conformations.")

    # --- B. Load ligand 3D features ---
    mol_pkl = f"{FEAT_DIR}/mol_reps_512d.pkl"
    with open(mol_pkl, 'rb') as f:
        m_names, m_reps = pickle.load(f)
    # Average the 6-fold outputs
    mol_reps_3d = np.mean(m_reps, axis=1)
    
    # --- C. Extract SMILES and compute 2D Morgan FP ---
    print("Extracting SMILES and computing 2D Morgan FP...")
    pkl_data_path = f"{FEAT_DIR}/mol.pkl_data"
    with open(pkl_data_path, 'rb') as f:
        mol_data_list = pickle.load(f)
    
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    
    mol_dict = {}
    for data in mol_data_list:
        name = data['mol_name']
        smi = data.get('smi', '')
        
        # Label matching logic
        if "all_inactives" in name:
            label = 0.0
        elif "all_actives" in name:
            label = 1.0
        else:
            label = 0.0
        
        mol_2d = np.zeros(2048, dtype=np.float32)
        if smi:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                mol_2d = mfpgen.GetFingerprintAsNumPy(mol).astype(np.float32)
        
        mol_dict[name] = {"2d": mol_2d, "label": label}

    # --- D. Load model weights ---
    model = FusionRegressor(dropout=0.0).to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    print(f"Evaluating {len(m_names)} molecules via Ensemble Docking...")
    with torch.no_grad():
        for i, m_name in enumerate(tqdm(m_names, desc="Scoring via Fusion MLP")):
            m_3d = torch.tensor(mol_reps_3d[i], dtype=torch.float32).unsqueeze(0).to(device)
            m_2d = torch.tensor(mol_dict[m_name]["2d"], dtype=torch.float32).unsqueeze(0).to(device)
            label = mol_dict[m_name]["label"]
            
            # Core logic: Ensemble Docking 
            # Score the ligand against all N pocket conformations
            scores_for_this_mol = []
            for p_rep in pocket_reps:
                p_3d = torch.tensor(p_rep, dtype=torch.float32).unsqueeze(0).to(device)
                score = model(p_3d, m_3d, m_2d).item()
                scores_for_this_mol.append(score)
            
            # Max Pooling for final affinity score
            final_max_score = max(scores_for_this_mol)
            
            all_preds.append(final_max_score)
            all_labels.append(label)

    # --- E. Final Results ---
    preds_tensor = torch.tensor(all_preds)
    labels_tensor = torch.tensor(all_labels)
    
    ef1 = compute_ef(preds_tensor, labels_tensor, top_ratio=0.01)
    ef5 = compute_ef(preds_tensor, labels_tensor, top_ratio=0.05)
    
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = 0.0 # Prevent sklearn error if all labels are 0 or 1

    print("\n" + "★"*50)
    print("🏆 LIT-PCBA (AVE unbiased) Evaluation Results 🏆")
    print("★"*50)
    print(f"Target        : {TARGET}")
    print(f"Total testing : {len(labels_tensor)}")
    print(f"Actives       : {int(labels_tensor.sum().item())}")
    print(f"Decoys        : {len(labels_tensor) - int(labels_tensor.sum().item())}")
    print("-" * 50)
    print(f"Enrichment Factor 1% (EF1%) : {ef1:.3f}")
    print(f"Enrichment Factor 5% (EF5%) : {ef5:.3f}")
    print(f"ROC-AUC                     : {auc:.4f}")
    print("★"*50)

if __name__ == "__main__":
    main()