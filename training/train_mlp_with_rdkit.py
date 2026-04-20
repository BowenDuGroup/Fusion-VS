import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import pickle
import numpy as np
from tqdm import tqdm
import lmdb 
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

def compute_ef(preds, labels, top_ratio=0.01, active_ratio=0.05):
    total_samples = len(labels)
    if total_samples == 0: return 0.0
    num_actives = max(1, int(total_samples * active_ratio))
    try: active_threshold = torch.topk(labels, num_actives)[0][-1]
    except RuntimeError: return 0.0
    is_active = labels >= active_threshold
    actual_actives_count = is_active.sum().float()
    if actual_actives_count == 0: return 0.0
    top_k = max(1, int(total_samples * top_ratio))
    _, top_indices = torch.topk(preds, top_k)
    hits = is_active[top_indices].sum().float()
    return ((hits / actual_actives_count) / top_ratio).item()

class PairwiseRankingAndRegressionLoss(nn.Module):
    def __init__(self, margin=1.0, rank_weight=2.0):
        super().__init__()
        self.reg_loss = nn.SmoothL1Loss() 
        self.margin = margin
        self.rank_weight = rank_weight

    def forward(self, preds, labels):
        loss_reg = self.reg_loss(preds, labels)
        pred_diff = preds.unsqueeze(1) - preds.unsqueeze(0)
        label_diff = labels.unsqueeze(1) - labels.unsqueeze(0)
        mask = label_diff > 0
        rank_loss_matrix = torch.clamp(self.margin - pred_diff, min=0.0)
        weighted_rank_loss = rank_loss_matrix * mask * label_diff
        valid_pairs_count = mask.sum()
        loss_rank = weighted_rank_loss.sum() / valid_pairs_count if valid_pairs_count > 0 else torch.tensor(0.0, device=preds.device)
        return loss_reg + self.rank_weight * loss_rank

def extract_names_and_smiles_from_lmdb(lmdb_path):
    print(f"Extracting mol names and SMILES from LMDB: {lmdb_path}")
    env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
    names, smiles_list = [], []
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            data_dict = pickle.loads(value)
            names.append(data_dict['mol_name'])
            smiles_list.append(data_dict.get('smi', '')) 
    env.close()
    return names, smiles_list

# Convert SMILES to 2048d Morgan FP
def smiles_to_morgan(smiles_list):
    print(f"Calculating Morgan Fingerprints for {len(smiles_list)} molecules...")
    fps = []
    
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    
    for smi in tqdm(smiles_list, desc="Morgan FP"):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = mfpgen.GetFingerprintAsNumPy(mol)
                fps.append(fp.astype(np.float32))
            else:
                fps.append(np.zeros(2048, dtype=np.float32))
        except:
            fps.append(np.zeros(2048, dtype=np.float32))
    return fps

class FusionDataset(Dataset):
    def __init__(self, pocket_pkl, mol_reps_npy, mol_lmdb_path, label_txt):
        print("[1/4] Loading Pocket 3D features...")
        with open(pocket_pkl, 'rb') as f:
            p_names, p_reps = pickle.load(f)
        self.pocket_dict = {name.decode('utf-8') if isinstance(name, bytes) else str(name): np.mean(rep, axis=0) for name, rep in zip(p_names, p_reps)} 
            
        print("[2/4] Loading Ligand 3D features & Extracting SMILES...")
        m_reps = np.load(mol_reps_npy)
        m_names, m_smiles = extract_names_and_smiles_from_lmdb(mol_lmdb_path)
        
        if len(m_names) != m_reps.shape[0]: raise ValueError("Mismatch!")
        
        # Store 3D features
        self.mol_dict_3d = {str(name): np.mean(rep, axis=0) for name, rep in zip(m_names, m_reps)}
        
        print("[3/4] Calculating 2D Morgan Fingerprints...")
        m_fps = smiles_to_morgan(m_smiles)
        
        # Store 2D features
        self.mol_dict_2d = {str(name): fp for name, fp in zip(m_names, m_fps)}
        
        print("[4/4] Aligning Pairs...")
        self.data_pairs = []
        with open(label_txt, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    p_path, m_path, affinity_str = parts[0], parts[1], parts[2]
                    if p_path in self.pocket_dict and m_path in self.mol_dict_3d:
                        try:
                            self.data_pairs.append({
                                'p_feat': self.pocket_dict[p_path],
                                'm_feat_3d': self.mol_dict_3d[m_path],
                                'm_feat_2d': self.mol_dict_2d[m_path],
                                'label': float(affinity_str)
                            })
                        except ValueError: continue
                        
        print(f"Fusion Dataset Built: {len(self.data_pairs)} valid pairs.\n")

    def __len__(self): return len(self.data_pairs)

    def __getitem__(self, idx):
        item = self.data_pairs[idx]
        return (
            torch.tensor(item['p_feat'], dtype=torch.float32), 
            torch.tensor(item['m_feat_3d'], dtype=torch.float32), 
            torch.tensor(item['m_feat_2d'], dtype=torch.float32), 
            torch.tensor(item['label'], dtype=torch.float32)
        )

# Fusion MLP: Accepts 3D and 2D features
class FusionRegressor(nn.Module):
    def __init__(self, embed_dim=512, fp_dim=2048, dropout=0.3):
        super().__init__()
        # 3D dims (2048) + 2D dims (2048) = 4096
        in_dim = (embed_dim * 4) + fp_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 2048), nn.BatchNorm1d(2048), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2048, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 1) 
        )

    def forward(self, p_embed, m_embed_3d, m_embed_2d):
        # 1. Build 3D cross features
        product_feat = p_embed * m_embed_3d
        diff_feat = torch.abs(p_embed - m_embed_3d)
        feat_3d = torch.cat([p_embed, m_embed_3d, product_feat, diff_feat], dim=-1)
        
        # 2. Concatenate 2D fingerprints
        x = torch.cat([feat_3d, m_embed_2d], dim=-1)
        
        return self.mlp(x).squeeze(-1)

def main():
    POCKET_PKL = '/root/autodl-tmp/AI4S1/result_2/pocket_reps_512d.pkl'
    MOL_REPS_NPY = '/root/autodl-tmp/AI4S1/result_2/mol_reps0None.npy'  
    MOL_LMDB_PATH = '/root/autodl-tmp/AI4S1/result_2/mol.lmdb'          
    LABEL_TXT = '/root/autodl-tmp/AI4S1/bigbind_label.txt'              
    SAVE_MODEL_PATH = '/root/autodl-tmp/AI4S1/result_2/best_fusion_mlp.pth'
    
    dataset = FusionDataset(POCKET_PKL, MOL_REPS_NPY, MOL_LMDB_PATH, LABEL_TXT)
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    BATCH_SIZE = 4096
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionRegressor(dropout=0.3).to(device)
    
    epochs = 200 
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))
    criterion = PairwiseRankingAndRegressionLoss(margin=1.0, rank_weight=2.0).to(device) 
    scaler = GradScaler()

    best_ef1 = 0.0
    print(f"\nUltimate Fusion (3D+2D) MLP Started! (Batch Size: {BATCH_SIZE})")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{epochs} [Train]", leave=False)
        for p_f, m_f_3d, m_f_2d, lbl in pbar:
            p_f, m_f_3d, m_f_2d, lbl = p_f.to(device), m_f_3d.to(device), m_f_2d.to(device), lbl.to(device)
            optimizer.zero_grad()
            
            with autocast():
                preds = model(p_f, m_f_3d, m_f_2d)
                loss = criterion(preds, lbl)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            
            scale_before = scaler.get_scale()
            scaler.update()
            scale_after = scaler.get_scale()
            if scale_before <= scale_after: scheduler.step()
            
            train_loss += loss.item() * p_f.size(0)
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        train_loss /= train_size

        model.eval()
        val_loss, all_preds, all_labels = 0.0, [], []
        
        with torch.no_grad():
            for p_f, m_f_3d, m_f_2d, lbl in val_loader:
                p_f, m_f_3d, m_f_2d, lbl = p_f.to(device), m_f_3d.to(device), m_f_2d.to(device), lbl.to(device)
                with autocast():
                    preds = model(p_f, m_f_3d, m_f_2d)
                    loss = criterion(preds, lbl)
                val_loss += loss.item() * p_f.size(0)
                all_preds.append(preds.detach().cpu().float())
                all_labels.append(lbl.detach().cpu().float())
                
        val_loss /= val_size
        val_ef1 = compute_ef(torch.cat(all_preds), torch.cat(all_labels), top_ratio=0.01, active_ratio=0.05)
        
        if val_ef1 > best_ef1:
            best_ef1 = val_ef1
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            print(f"Epoch [{epoch+1:03d}/{epochs}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} | NEW BEST Val EF1%: {best_ef1:.2f}%")
        elif (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1:03d}/{epochs}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} | Val EF1%: {val_ef1:.2f}%")

if __name__ == "__main__":
    main()