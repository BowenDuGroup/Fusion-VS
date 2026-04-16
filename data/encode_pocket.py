#!/usr/bin/env python3 -u
# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import pickle
import torch
import numpy as np
from tqdm import tqdm
import lmdb
import warnings
import shutil
import uuid

# ================= 强行劫持：优先加载 THU-ATOM 的魔改版 Uni-Mol =================
sys.path.insert(0, "/root/unimol/Drug-The-Whole-Genome")

import unimol
import unimol.tasks
import unimol.models
import unimol.tasks.drugclip
import unimol.models.drugclip

import unicore
from unicore import checkpoint_utils, distributed_utils, options
from unicore import tasks

from Bio.PDB import PDBParser
from Bio.PDB.StructureBuilder import PDBConstructionWarning

# ==============================================================================

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unimol.inference")

warnings.filterwarnings(
    action='ignore',
    category=PDBConstructionWarning
)


def write_lmdb(data, lmdb_path, start_idx=0):
    """AutoDL 专用安全写入方案 (SSD 缓存加速)"""
    for suffix in ['', '-lock']:
        p = lmdb_path + suffix
        if os.path.exists(p):
            if os.path.isdir(p):
                shutil.rmtree(p)
            else:
                os.remove(p)

    kv_pairs = []
    num = start_idx
    for d in data:
        key = str(num).zfill(8).encode('ascii')
        val = pickle.dumps(d)
        kv_pairs.append((key, val))
        num += 1

    kv_pairs.sort(key=lambda x: x[0])
    total_bytes = sum(len(k) + len(v) for k, v in kv_pairs)
    map_size = max(total_bytes * 3, 100 * 1024 * 1024)

    temp_path = f"/tmp/lmdb_build_{uuid.uuid4().hex[:8]}.lmdb"
    logger.info(f"Building LMDB on local SSD: {temp_path}  "
                f"(entries={len(kv_pairs)}, map_size={map_size // 1024 // 1024}MB)")

    env = lmdb.open(
        temp_path, subdir=False, readonly=False, lock=False,
        readahead=False, meminit=False, map_size=map_size, writemap=True,
    )

    batch_size = 5000
    for i in range(0, len(kv_pairs), batch_size):
        batch = kv_pairs[i:i + batch_size]
        with env.begin(write=True) as txn:
            for key, val in batch:
                txn.put(key, val, append=True)

    env.close()
    logger.info(f"Copying LMDB to network storage: {lmdb_path}")
    shutil.copy2(temp_path, lmdb_path)

    for suffix in ['', '-lock']:
        p = temp_path + suffix
        if os.path.exists(p):
            os.remove(p)
    return num


def get_unique_pocket_files(label_path):
    """
    极速去重：提取第一列的绝对路径。
    """
    unique_pockets = set()
    if not os.path.exists(label_path):
        logger.error(f"Label file not found: {label_path}")
        return unique_pockets

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1:
                # 第一列直接就是绝对路径
                pocket_file = parts[0]
                unique_pockets.add(pocket_file)

    logger.info(f"Scanned label file. Found {len(unique_pockets)} UNIQUE absolute pocket paths.")
    return unique_pockets


def process_one_pdbdir(pocket_base_dir, out_dir, label_path, name='pocket'):
    """统一解析逻辑：直接读取绝对路径"""
    lmdb_out_path = os.path.join(out_dir, f'{name}.lmdb')

    if os.path.exists(lmdb_out_path):
        logger.info(f"Found existing LMDB database: {lmdb_out_path}, skipping build.")
        return 0

    unique_pocket_files = get_unique_pocket_files(label_path)
    if not unique_pocket_files:
        return 1

    all_pocket = []
    p_parser = PDBParser(QUIET=True)

    for pocket_file in tqdm(unique_pocket_files, desc="Parsing Unique Pockets"):
        # pocket_file 已经是绝对路径了，无需拼接
        if not os.path.exists(pocket_file):
            logger.warning(f"Pocket file not found on disk: {pocket_file}")
            continue

        # ==============================================================
        # 核心修改：直接用完整绝对路径作为特征的 ID (Key)！
        # 这样训练时读到 label.txt 第一列时，能一字不差地精准命中这个特征
        # ==============================================================
        pocket_id = pocket_file 
        
        # 给 PDBParser 用的一个简短名字（防止内部报错，不影响外部）
        short_id = os.path.basename(pocket_file).split('.')[0]

        try:
            structure = p_parser.get_structure(short_id, pocket_file)
            pocket_atom_type = []
            pocket_coord = []

            for model in structure:
                for chain in model:
                    for res in chain:
                        for atom in res:
                            if atom.element != 'H':
                                pocket_atom_type.append(atom.element)
                                pocket_coord.append(list(atom.coord))

            if len(pocket_atom_type) > 0:
                all_pocket.append({
                    'pocket': pocket_id, # 保存时带入完整的绝对路径
                    'pocket_atoms': pocket_atom_type,
                    'pocket_coordinates': pocket_coord,
                    'affinity': 0.0
                })
            else:
                logger.warning(f"No valid heavy atoms extracted from {pocket_file}.")

        except Exception as e:
            logger.error(f"Failed to parse {pocket_file}: {e}")

    if len(all_pocket) == 0:
        logger.error("No valid pocket data processed.")
        return 1

    logger.info(f"Successfully parsed {len(all_pocket)} unique pockets. Writing to LMDB...")
    write_lmdb(all_pocket, lmdb_out_path, 0)
    return 0


def main(args):
    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)

    logger.info("loading model(s) from {}".format(args.path))
    task = tasks.setup_task(args)
    model = task.build_model(args)

    state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    model.load_state_dict(state["model"], strict=False)

    import torch.nn as nn
    model.pocket_project = nn.Identity()
    logger.info("Successfully bypassed `pocket_project` using `nn.Identity()`. Will output 512d features.")

    if use_fp16:
        model.half()
    if use_cuda:
        model.cuda()

    model.eval()

    out_dir = args.results_path if args.results_path else args.pocket_dir
    os.makedirs(out_dir, exist_ok=True)
    lmdb_path = os.path.join(out_dir, "pocket.lmdb")

    if not os.path.exists(lmdb_path):
        ret = process_one_pdbdir(args.pocket_dir, out_dir, args.label_file)
        if ret != 0:
            logger.error("Failed to build pocket LMDB data, aborting.")
            return

    pocket_reps, pocket_names = task.encode_pockets_multi_folds(model, args.pocket_dir, lmdb_path)

    # 保存 512 维特征
    with open(os.path.join(out_dir, "pocket_reps_512d.pkl"), "wb") as f:
        pickle.dump((pocket_names, pocket_reps), f)
    logger.info(f"Successfully saved 512d features to {os.path.join(out_dir, 'pocket_reps_512d.pkl')}")

def cli_main():
    parser = options.get_validation_parser()
    parser.add_argument("--pocket-dir", type=str, default="", help="path for pocket dir (not strictly used if absolute paths provided)")
    parser.add_argument("--label-file", type=str, required=True,
                        help="path to the txt file containing absolute pocket paths in the first column")

    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)

    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()