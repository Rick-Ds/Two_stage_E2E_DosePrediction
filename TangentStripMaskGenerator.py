# -*- coding: utf-8 -*-
"""
TangentStripMaskGenerator.py

Generate per-case 3D strip masks for stage-2 refinement without using planning parameters.

For each axial slice and each angle in ANGLES_DEG, the script computes the projection interval of
the PTV along the corresponding normal direction and marks voxels inside the tangent band, masked
by Body. Outputs strip_full_thetaXXX.nii.gz in each case directory.

Author: Boda Ning
"""


import os, math, numpy as np, nibabel as nib


PATH = r"E:\DosePred\DATA\3.486_all_niiData_256_45Gy(no_chongfu)"
ANGLES_DEG = [0.0, 45.0, 90.0, 135.0]
OUT_MM = 0.0

def load_nii(p):
    img = nib.load(p)
    arr = img.get_fdata(dtype=np.float32)
    return arr, img.affine, img.header

def xyz_to_dhw(a): return np.transpose(a, (2,1,0)).astype(np.float32, copy=False)
def dhw_to_xyz(a): return np.transpose(a, (2,1,0)).astype(np.float32, copy=False)
def ensure_bin(m, t=0.5): return (m.astype(np.float32) > t).astype(np.float32)

def main():
    root = PATH
    for pat_dir in os.listdir(root):
        pat_path = os.path.join(root, pat_dir)
        body_p = os.path.join(pat_path, "Body_256_64.nii.gz")
        ptv_p  = os.path.join(pat_path, "PTV_256_64.nii.gz")
        assert os.path.isfile(body_p) and os.path.isfile(ptv_p), "必须在病例目录运行，并包含 Body_256_64.nii.gz 与 PTV_256_64.nii.gz"

        body_xyz, aff, hdr = load_nii(body_p)
        ptv_xyz, _, _ = load_nii(ptv_p)

        body = ensure_bin(xyz_to_dhw(body_xyz))
        ptv = ensure_bin(xyz_to_dhw(ptv_xyz))
        D, H, W = body.shape

        dx, dy, dz = hdr.get_zooms()[:3]
        sz, sy, sx = float(dz), float(dy), float(dx)

        pix_mm = math.hypot(sx, sy)
        out_pix = OUT_MM / max(1e-6, pix_mm)


        Y_abs = np.arange(H, dtype=np.float32)[:, None]
        X_abs = np.arange(W, dtype=np.float32)[None, :]

        for deg in ANGLES_DEG:
            th = math.radians(deg)


            nx, ny = math.cos(th), math.sin(th)


            full_3d = np.zeros((D, H, W), dtype=np.float32)

            for z in range(D):
                body2d = body[z];
                ptv2d = ptv[z]
                if ptv2d.max() < 0.5 or body2d.max() < 0.5:
                    continue

                ys, xs = np.nonzero(ptv2d)

                proj = xs.astype(np.float32) * nx + ys.astype(np.float32) * ny
                pmin = float(proj.min());
                pmax = float(proj.max())


                P = X_abs * nx + Y_abs * ny


                full2d = ((P >= (pmin - out_pix)) & (P <= (pmax + out_pix))).astype(np.float32)
                full2d *= body2d

                full_3d[z] = full2d


            def save_vol(vol_dhw, name):
                vol_xyz = dhw_to_xyz(vol_dhw)
                nib.save(nib.Nifti1Image(vol_xyz.astype(np.float32), affine=aff, header=hdr),
                         os.path.join(pat_path, name))

            save_vol(full_3d, f"strip_full_theta{int(round(deg)):03d}.nii.gz")
            print(f"θ={deg:5.1f}° -> strip_full_* / strip_edge_* 已保存。")

        print("\n完成：逐层相切、方向与期望一致的条带已生成。")
        print("训练建议：L = MAE(Body内) + λ * mean_k MAE( strip_edge_theta_k 内 )，仍为 MAE/MSE 家族。")


if __name__ == '__main__':
    main()
