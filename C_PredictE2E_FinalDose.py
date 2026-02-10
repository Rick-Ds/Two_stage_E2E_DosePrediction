# -*- coding: utf-8 -*-
"""
C_PredictE2E_FinalDose.py

Full-volume NIfTI inference script for the end-to-end two-stage model (UNet3D + ResUNet3D).

For each case directory under ROOT_DIR, the script loads CT, structure masks, and strip masks,
runs TwoStageDoseNet, rescales predictions to Gy, masks dose by Body, and writes the output NIfTI.

Author: Boda Ning
"""


import os
import glob
from typing import List
import numpy as np
import SimpleITK as sitk
import torch

from util import imgutil, fileutil
from Network.Unet3D import UNet3D
from Network.ResUNet3D import ResUNet3D


ROOT_DIR = r"D:\Data\test"


CKPT_PATH = r"../Prj_DoseFormer_torch/model_B_E2E/model_Baseline_Lband_Unet_ResUnet_64_256_256_linear_epoch61.pth"


OUT_NAME_FINAL  = "E2E_Baseline_Lband_Pred_Dose_256_64.nii.gz"


DEVICE      = "cuda:0"
USE_AMP     = False
DOSE_MAX_GY = 52.0


STRUCT_ORDER: List[str] = [
    'PTV',
    'Bladder',
    'Rectum',
    'Small_Intestine',
    'Marrow',
    'SpinalCord',
    'FemurHead_L',
    'FemurHead_R',
    'Body'
]


STRIP_FILES: List[str] = [
    "strip_full_theta000.nii.gz",
    "strip_full_theta045.nii.gz",
    "strip_full_theta090.nii.gz",
    "strip_full_theta135.nii.gz",
]

IN_CHANNELS_2ND = 14
OUT_CHANNELS = 1
ACTIVATION = "linear"


class TwoStageDoseNet(torch.nn.Module):
    def __init__(self,
                 base_filters_stage1: int = 8,
                 base_filters_stage2: int = 16,
                 activation_stage2: str = ACTIVATION):
        super().__init__()

        self.stage1 = UNet3D(
            in_channels=10,
            out_channels=1,
            base_num_filters=base_filters_stage1,
            activation=ACTIVATION,
        )

        self.stage2 = ResUNet3D(
            in_channels=IN_CHANNELS_2ND + 1,
            out_channels=OUT_CHANNELS,
            base_num_filters=base_filters_stage2,
            activation=activation_stage2,
        )

    def forward(self, x_2nd: torch.Tensor):
        ct     = x_2nd[:, 0:1]
        masks  = x_2nd[:, 1:10]
        strips = x_2nd[:, 10:14]


        x_stage1    = torch.cat([ct, masks], dim=1)
        coarse_pred = self.stage1(x_stage1)


        x_stage2  = torch.cat([ct, masks, coarse_pred, strips], dim=1)
        dose_pred = self.stage2(x_stage2)

        return dose_pred, coarse_pred


@torch.no_grad()
def main():

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    dev = torch.device(DEVICE if torch.cuda.is_available() else "cpu")


    model = TwoStageDoseNet().to(dev)

    ckpt = torch.load(CKPT_PATH, map_location=dev)
    if isinstance(ckpt, dict):

        if "model" in ckpt:
            state = ckpt["model"]
        elif "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt

    model.load_state_dict(state, strict=True)
    model.eval()
    print("[INFO] TwoStageDoseNet 与权重已加载。")


    case_dirs = sorted([p for p in glob.glob(os.path.join(ROOT_DIR, "*")) if os.path.isdir(p)])
    print(f"[INFO] 共发现 {len(case_dirs)} 个病例。")

    for idx, case_dir in enumerate(case_dirs, 1):
        case_id = os.path.basename(os.path.normpath(case_dir))
        print(f"\n[{idx}/{len(case_dirs)}] Case: {case_id}")


        ct_path = os.path.join(case_dir, "ct_256_64.nii.gz")
        ct = imgutil.load_nii(ct_path, dtype=np.float32, zfirst=True)


        D, H, W = ct.shape
        input_data = np.zeros((1, IN_CHANNELS_2ND, D, H, W), dtype=np.float32)

        ct_clip = np.clip(ct, -1000.0, 1000.0)
        ct_norm = (ct_clip - (-1000.0)) / (1000.0 - (-1000.0))
        input_data[0, 0, ...] = ct_norm


        for i, struct in enumerate(STRUCT_ORDER, start=1):
            struct_path = os.path.join(case_dir, f"{struct}_256_64.nii.gz")
            if not os.path.exists(struct_path):
                raise FileNotFoundError(f"缺少结构文件: {struct_path}")
            struct_data = imgutil.load_nii(struct_path, dtype=np.float32, zfirst=True)
            input_data[0, i, ...] = struct_data


        for j, strip_name in enumerate(STRIP_FILES):
            strip_path = os.path.join(case_dir, strip_name)
            if not os.path.exists(strip_path):
                raise FileNotFoundError(f"缺少条带文件: {strip_path}")
            strip_data = imgutil.load_nii(strip_path, dtype=np.float32, zfirst=True)
            input_data[0, 10 + j, ...] = strip_data


        x_t = torch.from_numpy(input_data).to(dev, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            dose_pred_t, coarse_pred_t = model(x_t)

        dose_pred = dose_pred_t.squeeze(0).squeeze(0).detach().cpu().numpy()
        coarse_pred = coarse_pred_t.squeeze(0).squeeze(0).detach().cpu().numpy()


        if ACTIVATION == "sigmoid":
            dose_pred = np.clip(dose_pred, 0.0, 1.0)
            coarse_pred = np.clip(coarse_pred, 0.0, 1.0)
        else:

            dose_pred = np.maximum(dose_pred, 0.0)
            coarse_pred = np.maximum(coarse_pred, 0.0)

        dose_pred *= DOSE_MAX_GY
        coarse_pred *= DOSE_MAX_GY


        body = imgutil.load_nii(os.path.join(case_dir, "Body_256_64.nii.gz"), dtype=np.float32, zfirst=True)
        dose_pred *= body
        coarse_pred *= body


        ct_nii = sitk.ReadImage(ct_path)
        out_final_path  = os.path.join(case_dir, OUT_NAME_FINAL)


        imgutil.write_nii(dose_pred, out_final_path, spacing=ct_nii.GetSpacing(), origin=ct_nii.GetOrigin())


        print(f"  [OK] 最终剂量已保存:  {out_final_path}")


    print("\n[ALL DONE] E2E 推理完成。")


if __name__ == "__main__":
    main()
