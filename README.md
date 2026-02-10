# E2E Dose Prediction (Two-Stage 3D UNet → ResUNet Refinement)

Research code for **end-to-end 3D radiotherapy dose prediction** on full volumes using a **two-stage network**:

- **Stage 1 (coarse):** `UNet3D` takes **CT + 9 structure masks** (10 channels) and predicts a coarse normalized dose.
- **Stage 2 (refine):** `ResUNet3D` takes **CT + 9 masks + coarse dose + 4 tangent strip masks** (15 channels total) and predicts the final dose.

This repository also includes:
- DVH- and dose-based evaluation utilities (OpenKBP-style DVH score / dose score).
- Loss functions used for training (voxel, gradient, SSIM, band/strip-weighted, DVH-inspired surrogates).
- Scripts for strip-mask generation, data pre-caching to `.pt`, DDP training, and full-volume NIfTI inference.

> **Disclaimer**: This is **research** code. It is **not** a medical device and must not be used for clinical decision-making.

---

## Repository layout

Place the provided files into the following structure (recommended):

```
.
├─ Network/
│  ├─ Unet3D.py
│  └─ ResUNet3D.py
├─ loss/
│  └─ Loss_Func.py
├─ evaluation/
│  ├─ metrics.py
│  └─ DVHScoreDoseScore.py
├─ scripts/
│  ├─ A_2ndPhase_Precache_to_pt.py
│  ├─ B_E2E_Train_DDP_DataLoader.py
│  ├─ B_E2E_Train_DDP_2loss.py
│  ├─ B_E2E_Train_DDP_2loss_debug.py
│  ├─ C_PredictE2E_FinalDose.py
│  ├─ TangentStripMaskGenerator.py
│  └─ train_utils.py
└─ README.md
```

**Important**
- These scripts assume the folders (`Network/`, `loss/`, `evaluation/`) are importable Python packages.
  Add empty `__init__.py` files to each folder if needed.

---

## Environment

### Dependencies

Core dependencies are listed in `requirements.txt`. Install PyTorch following the official instructions for your CUDA version, then:

```bash
pip install -r requirements.txt
```

---

## Data conventions

### NIfTI per-case directory

Most scripts assume **one case per folder**, containing (example naming):

- `ct_256_64.nii.gz`
- `PTV_256_64.nii.gz`
- `Bladder_256_64.nii.gz`
- `Rectum_256_64.nii.gz`
- `Small_Intestine_256_64.nii.gz`
- `Marrow_256_64.nii.gz`
- `SpinalCord_256_64.nii.gz`
- `FemurHead_L_256_64.nii.gz`
- `FemurHead_R_256_64.nii.gz`
- `Body_256_64.nii.gz`
- `dose_plan_256_64.nii.gz` (ground truth plan dose, in Gy)
- strip masks (generated): `strip_full_theta000.nii.gz`, `strip_full_theta045.nii.gz`, `strip_full_theta090.nii.gz`, `strip_full_theta135.nii.gz`

### Normalization (as used in scripts)

- CT is clipped to **[-1000, 1000] HU**, then linearly scaled to **[0, 1]**
- Dose is scaled to **[0, 1]** using `DOSE_MAX_GY` (default **52.0 Gy**), and restored to Gy at inference.

---

## Workflow

### 1) Generate tangent strip masks

Edit the path constants in `scripts/TangentStripMaskGenerator.py` (e.g., `PATH`) to point to your dataset root, then run:

```bash
python scripts/TangentStripMaskGenerator.py
```

This writes `strip_full_thetaXXX.nii.gz` into each case directory.

### 2) Pre-cache to `.pt` tensors (stage-2 training format)

Edit `DATA_DIR` and `OUT_DIR` in `scripts/A_2ndPhase_Precache_to_pt.py`, then run:

```bash
python scripts/A_2ndPhase_Precache_to_pt.py
```

Each output `.pt` contains:
- `x`: `(14, D, H, W)` = **[CT, 9 structure masks, 4 strip masks]**
- `y`: `(1, D, H, W)` = normalized GT dose
- `id`: case identifier

> The **coarse dose** is produced on-the-fly during training by Stage 1 and concatenated with `x` to form the Stage-2 input (15 channels).

### 3) Train (DDP optional)

Edit paths (e.g., `TRAIN_PT`, `VAL_PT`, `TB_LOGDIR`, `CKPT_BEST`) at the top of:
- `scripts/B_E2E_Train_DDP_2loss.py` (main)
- `scripts/B_E2E_Train_DDP_2loss_debug.py` (prints loss components)

Single process:

```bash
python scripts/B_E2E_Train_DDP_2loss.py
```

DDP (recommended):

```bash
torchrun --nproc_per_node=NUM_GPUS scripts/B_E2E_Train_DDP_2loss.py
```

Training logs are written to TensorBoard (`TB_LOGDIR`).

### 4) Inference: predict final dose as NIfTI

Edit `ROOT_DIR` (cases), `CKPT_PATH` (checkpoint), and output name in `scripts/C_PredictE2E_FinalDose.py`, then run:

```bash
python scripts/C_PredictE2E_FinalDose.py
```

The script writes `E2E_Baseline_Lband_Pred_Dose_256_64.nii.gz` into each case folder (masked by `Body`).

---

## Evaluation

### Common metrics (evaluation/metrics.py)

- `mae_metric`: voxel-wise MAE (Gy)
- `ssim3d_metric`: 3D SSIM
- `dvh_score`: fixed PTV + OAR endpoint set (PTV Dmean/D98/D2, V95/V100, CI, HI; plus selected OAR endpoints)
- `dvh_score_openkbp`: OpenKBP-style DVH endpoints (PTV D99/D95/D1; OAR D0.1cc/mean)

### OpenKBP-style scores (evaluation/DVHScoreDoseScore.py)

Provides:
- **DVH score**: mean absolute error over DVH endpoints
- **Dose score**: mean absolute voxel error within the possible-dose region

---

## Notes / common pitfalls

- **Package imports**: ensure `Network/`, `loss/`, `evaluation/` are packages (`__init__.py`).
- **Hard-coded paths**: scripts are written for internal experiments; update the path constants before running.
- **Optional nnUNet import**: training scripts may include an import for an `nnUNet3D` variant. If you do not provide that file, remove the unused import line.

---

## Citation

If you use this code in academic work, please cite your corresponding paper/project. A `CITATION.cff` template is included.

---

## License

A default MIT license template is provided in `LICENSE`. Update the copyright holder and year.

