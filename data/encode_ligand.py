#!/usr/bin/env python3 -u
import logging
import os
import sys
import pickle
import torch
import numpy as np
from tqdm import tqdm
import warnings
import glob

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.error')

sys.path.insert(0, "/root/unimol/Drug-The-Whole-Genome")

import unimol
import unimol.tasks
import unimol.models
import unimol.tasks.drugclip
import unimol.models.drugclip

import unicore
from unicore import checkpoint_utils, distributed_utils, options
from unicore import tasks

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unimol.inference")
warnings.filterwarnings('ignore')


# ==================== Pickle 模拟 LMDB 的 Dataset ====================
class PickleLMDBDataset:
    """用 pickle 文件模拟 LMDBDataset 的接口，完全不碰 lmdb"""
    def __init__(self, pkl_path):
        logger.info(f"Loading PickleLMDBDataset from {pkl_path}")
        with open(pkl_path, 'rb') as f:
            self._data = pickle.load(f)  # list of dicts
        logger.info(f"Loaded {len(self._data)} entries")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


def monkey_patch_lmdb_dataset():
    """猴子补丁：劫持 LMDBDataset，让它从 .pkl 读取而不是 .lmdb"""
    import unimol.data as udata

    _OrigLMDBDataset = udata.LMDBDataset

    class PatchedLMDBDataset:
        def __init__(self, data_path, *args, **kwargs):
            # 如果对应的 .pkl 文件存在，用 pickle 读
            pkl_path = data_path.replace('.lmdb', '.pkl_data')
            if os.path.exists(pkl_path):
                logger.info(f"[PATCHED] Loading from pickle: {pkl_path}")
                with open(pkl_path, 'rb') as f:
                    self._data = pickle.load(f)
                self._use_pickle = True
            else:
                logger.info(f"[PATCHED] No pickle found, falling back to original LMDB: {data_path}")
                self._orig = _OrigLMDBDataset(data_path, *args, **kwargs)
                self._use_pickle = False

        def __len__(self):
            if self._use_pickle:
                return len(self._data)
            return len(self._orig)

        def __getitem__(self, idx):
            if self._use_pickle:
                return self._data[idx]
            return self._orig[idx]

    udata.LMDBDataset = PatchedLMDBDataset

    # 同时补丁 tasks.drugclip 里的引用
    try:
        import unimol.tasks.drugclip as dc_module
        dc_module.LMDBDataset = PatchedLMDBDataset
    except:
        pass

    logger.info("[PATCHED] LMDBDataset has been replaced with pickle-compatible version")


def get_unique_sdf_files(label_path):
    unique_sdfs = set()
    if not os.path.exists(label_path):
        logger.error(f"Label file not found: {label_path}")
        return unique_sdfs
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                unique_sdfs.add(parts[1])
    logger.info(f"Scanned label file. Found {len(unique_sdfs)} UNIQUE absolute SDF paths.")
    return unique_sdfs


def process_one_sdfdir(out_dir, label_path, name='mol'):
    """解析 SDF，直接存成 pickle 列表，完全不碰 lmdb"""
    pkl_out_path = os.path.join(out_dir, f'{name}.pkl_data')

    if os.path.exists(pkl_out_path):
        logger.info(f"Found existing pickle data: {pkl_out_path}, skipping build.")
        return 0

    unique_sdfs = get_unique_sdf_files(label_path)
    if not unique_sdfs:
        return 1

    all_mols = []
    for sdf_file in unique_sdfs:
        if not os.path.exists(sdf_file):
            continue
        logger.info(f"📂 正在解析: {os.path.basename(sdf_file)}")
        try:
            suppl = Chem.SDMolSupplier(sdf_file, removeHs=False, sanitize=True)
            for i, mol in enumerate(tqdm(suppl, desc=f"Parsing {os.path.basename(sdf_file)}")):
                if mol is not None:
                    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
                    coords = [list(mol.GetConformer().GetAtomPosition(j)) for j in range(mol.GetNumAtoms())]
                    try:
                        smi = Chem.MolToSmiles(mol)
                    except:
                        smi = ""
                    all_mols.append({
                        'mol_name': f"{sdf_file}:::{i}",
                        'atoms': atoms,
                        'coordinates': [np.array(coords, dtype=np.float32)],
                        'affinity': 0.0,
                        'smi': smi
                    })
        except Exception as e:
            logger.error(f"Failed to parse {sdf_file}: {e}")

    if not all_mols:
        logger.error("No valid SDF data processed.")
        return 1

    logger.info(f"✅ 解析完成: {len(all_mols)} 个配体，保存为 pickle...")
    with open(pkl_out_path, 'wb') as f:
        pickle.dump(all_mols, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"✅ 已保存到 {pkl_out_path}")
    return 0


def find_npy_files(out_dir, prefix, start, end):
    candidates = [
        os.path.join(out_dir, f"{prefix}{start}{end}.npy"),
        os.path.join(out_dir, f"{prefix}_{start}_{end}.npy"),
        os.path.join(out_dir, f"{prefix}.npy"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    pattern = os.path.join(out_dir, f"{prefix}*.npy")
    matches = glob.glob(pattern)
    if matches:
        matches.sort(key=os.path.getmtime, reverse=True)
        return matches[0]
    return None


def main(args):
    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu
    if use_cuda:
        torch.cuda.set_device(args.device_id)

    # ===== 激活猴子补丁，让下游代码用 pickle 代替 lmdb =====
    monkey_patch_lmdb_dataset()

    logger.info("Loading model from {}".format(args.path))
    task = tasks.setup_task(args)
    model = task.build_model(args)
    state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    model.load_state_dict(state["model"], strict=False)

    import torch.nn as nn
    model.mol_project = nn.Identity()
    logger.info("Successfully bypassed `mol_project` using `nn.Identity()`.")

    if use_fp16:
        model.half()
    if use_cuda:
        model.cuda()
    model.eval()

    out_dir = args.results_path if args.results_path else args.save_dir
    os.makedirs(out_dir, exist_ok=True)

    # ===== 第一步：解析 SDF -> pickle（不碰 lmdb）=====
    ret = process_one_sdfdir(out_dir, args.label_file)
    if ret != 0:
        logger.error("Failed to build molecule data, aborting.")
        return

    # ===== 第二步：让下游以为 mol.lmdb 存在（实际读 mol.pkl_data）=====
    lmdb_path = os.path.join(out_dir, "mol.lmdb")
    pkl_data_path = os.path.join(out_dir, "mol.pkl_data")

    # 创建一个空的占位文件，让下游路径检查不报错
    if not os.path.exists(lmdb_path):
        with open(lmdb_path, 'w') as f:
            f.write("")  # 空占位文件

    logger.info("Extracting embeddings...")
    existing_npys_before = set(glob.glob(os.path.join(out_dir, "*.npy")))

    result = task.encode_mols_multi_folds(
        model, args.batch_size, lmdb_path, out_dir, use_cuda,
        write_npy=True, write_h5=False, start=args.start, end=args.end
    )

    # ===== 第三步：收集结果 =====
    if result is not None:
        try:
            if isinstance(result, tuple) and len(result) == 2:
                mol_reps, mol_names = result
                pkl_path = os.path.join(out_dir, "mol_reps_512d.pkl")
                with open(pkl_path, "wb") as f:
                    pickle.dump((mol_names, mol_reps), f)
                logger.info(f"Successfully saved features to {pkl_path}")
                return
        except Exception as e:
            logger.warning(f"Failed to unpack return: {e}")

    npy_names_path = find_npy_files(out_dir, "mol_names", args.start, args.end)
    npy_reps_path = find_npy_files(out_dir, "mol_reps", args.start, args.end)

    if npy_reps_path:
        mol_reps = np.load(npy_reps_path)
        if npy_names_path:
            mol_names = np.load(npy_names_path, allow_pickle=True)
        else:
            # 从 pickle 重建 names
            logger.info("Rebuilding mol_names from pickle data...")
            with open(pkl_data_path, 'rb') as f:
                data = pickle.load(f)
            mol_names = np.array([d.get('mol_name', str(i)) for i, d in enumerate(data)][:mol_reps.shape[0]])

        pkl_path = os.path.join(out_dir, "mol_reps_512d.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump((mol_names, mol_reps), f)
        logger.info(f"Successfully saved features to {pkl_path}")
    else:
        all_npys = glob.glob(os.path.join(out_dir, "*.npy"))
        logger.error(f"Could not find .npy files! All in {out_dir}: {all_npys}")


def cli_main():
    parser = options.get_validation_parser()
    parser.add_argument("--label-file", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default="")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--write-npy", action="store_true")
    parser.add_argument("--write-h5", action="store_true")
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)
    distributed_utils.call_main(args, main)

if __name__ == "__main__":
    cli_main()