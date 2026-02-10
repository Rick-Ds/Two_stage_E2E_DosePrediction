# -*- coding: utf-8 -*-
"""
A_2ndPhase_Precache_to_pt.py

Preprocess stage-2 training data by converting per-case NIfTI volumes into .pt tensors.

Each case directory is expected to contain:
- ct_256_64.nii.gz
- 9 structure masks listed in STRUCT_ORDER (suffix: _256_64.nii.gz)
- 4 strip masks listed in STRIP_FILES
- dose_plan_256_64.nii.gz

The script normalizes CT to [0, 1] after clipping, normalizes dose to [0, 1] using DOSE_NORM,
and saves one .pt per case with keys: {"x", "y", "id"}.

Author: Boda Ning
"""

import os
import numpy as np
import torch
from tqdm import tqdm
from util import fileutil, imgutil


DATA_DIR = r"/opt/data/private/Prj_DoseFormer_torch/Data_nii/3.458_train_val_test_256_45Gy/111/"
OUT_DIR  = r"/opt/data/private/Prj_DoseFormer_torch/Data_PT/val_2nd_PT/"


STRUCT_ORDER = ['PTV','Bladder','Rectum','Small_Intestine',
                'Marrow','SpinalCord','FemurHead_L','FemurHead_R','Body']

STRIP_FILES = {
    "theta000": "strip_full_theta000.nii.gz",
    "theta045": "strip_full_theta045.nii.gz",
    "theta090": "strip_full_theta090.nii.gz",
    "theta135": "strip_full_theta135.nii.gz",
}


CT_CLIP = (-1000.0, 1000.0)
CT_NORM = (-1000.0, 1000.0)
DOSE_NORM = (0.0, 52.0)

def normalize_linear(arr, min_val, max_val):
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.clip(arr, min_val, max_val)
    return (arr - min_val) / (max_val - min_val + 1e-12)

def load_bin(path):
    m = imgutil.load_nii(path, dtype=np.float32, zfirst=True)
    return (m > 0).astype(np.float32)

def find_first_exists(root, names):
    for n in names:
        p = os.path.join(root, n)
        if os.path.isfile(p): return p
    raise FileNotFoundError(f"在 {root} 未找到任一 coarse 文件名：{names}")

def build_case(case_dir):

    ct = imgutil.load_nii(os.path.join(case_dir, "ct_256_64.nii.gz"),
                          dtype=np.float32, zfirst=True)
    ct = np.clip(ct, *CT_CLIP)
    ct = normalize_linear(ct, *CT_NORM)


    struct_maps = []
    for name in STRUCT_ORDER:
        seg = load_bin(os.path.join(case_dir, f"{name}_256_64.nii.gz"))
        struct_maps.append(seg)


    dose_gt = imgutil.load_nii(os.path.join(case_dir, "dose_plan_256_64.nii.gz"),
                               dtype=np.float32, zfirst=True)
    y = normalize_linear(dose_gt, *DOSE_NORM)[None, ...]


    strips = []
    for key in ("theta000","theta045","theta090","theta135"):
        sp = os.path.join(case_dir, STRIP_FILES[key])
        strips.append(load_bin(sp))


    x = np.concatenate(
        [ct[None, ...],
         np.asarray(struct_maps, dtype=np.float32),
         np.asarray(strips, dtype=np.float32)],
        axis=0
    ).astype(np.float32)

    assert x.shape[0] == 14, f"通道数应为 14，实际 {x.shape[0]}"
    return x, y

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cases = fileutil.list_dir(DATA_DIR)
    print(f"发现 {len(cases)} 例，开始预处理到 {OUT_DIR}")
    for case_dir in tqdm(cases):
        pid = os.path.basename(os.path.normpath(case_dir))
        x_np, y_np = build_case(case_dir)
        torch.save({"x": torch.from_numpy(x_np),
                    "y": torch.from_numpy(y_np),
                    "id": pid},
                   os.path.join(OUT_DIR, f"{pid}.pt"))
    print("✅ 二阶段 .pt 预处理完成。")

if __name__ == "__main__":
    main()
