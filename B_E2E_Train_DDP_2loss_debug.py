# -*- coding: utf-8 -*-
"""
B_E2E_Train_DDP_2loss_debug.py

Debug variant of the end-to-end two-stage training script (UNet3D + ResUNet3D) with optional DDP.
It prints per-component loss values for early iterations to help diagnose loss scaling.

Author: Boda Ning
"""


import math
import time
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from B_E2E_Train_DDP_DataLoader import FullVolumePTDataset2nd
from Network.nnUnet3D import nnUNet3D_V1
from Network.Unet3D import UNet3D
from Network.ResUNet3D import ResUNet3D
from loss.Loss_Func import BandWeightedMAE, BandWeightedSSIM, DMeanLoss, VxLoss, gradient_l1_loss
from evaluation.metrics import mae_metric, ssim3d_metric, dvh_score, dvh_score, dvh_score_openkbp


USE_DDP = False


TRAIN_PT  = r"/opt/data/private/Prj_DoseFormer_torch/Data_PT/train_2nd_PT/"
VAL_PT    = r"/opt/data/private/Prj_DoseFormer_torch/Data_PT/val_2nd_PT/"
CKPT_BEST = r"../Prj_DoseFormer_torch/model/Baseline_Lband/model_Baseline_Lband_Lgrad_LDVH_Unet_ResUnet_64_256_256_linear_best.pth"

CKPT_PREFIX = CKPT_BEST.replace("_best.pth", "")
TB_LOGDIR = r"../Prj_DoseFormer_torch/logs/model_Baseline_Lband_Lgrad_LDVH_Unet_ResUnet_64_256_256_linear"


STRUCT_ORDER = ['PTV','Bladder','Rectum','Small_Intestine','Marrow', 'SpinalCord','FemurHead_L','FemurHead_R','Body']
CT_IDX      = 0
STRIP_IDXS  = [10,11,12,13]
BODY_IDX    = STRUCT_ORDER.index('Body') + 1


IN_CHANNELS_2ND = 15
BATCH_SIZE  = 2
EPOCHS      = 400
LR          = 1e-3
NUM_WORKERS = 4
DOSE_MAX_GY = 52.0
ACTIVATION = 'linear'


LAMBDA_STAGE1_WEIGHT = 0.5
LAMBDA_STAGE2_WEIGHT = 0.5


LAMBDA_BODY_MAE   = 0.7
LAMBDA_BAND_MAE   = 0.1
LAMBDA_GRAD       = 0.1
LAMBDA_L_DMEAN    = 0.05
LAMBDA_L_VX       = 0.05


device = torch.device("cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


def extract_body_mask(x: torch.Tensor) -> torch.Tensor:
    return (x[:, BODY_IDX:BODY_IDX+1] > 0.5).float()

def extract_ptv_oars(x: torch.Tensor):
    start = 1
    ptv = (x[:, start + STRUCT_ORDER.index('PTV'): start + STRUCT_ORDER.index('PTV') + 1] > 0.5).float()
    oars = {}
    for i, name in enumerate(STRUCT_ORDER):
        if name in ('PTV', 'Body'):
            continue
        oars[name] = (x[:, start + i:start + i + 1] > 0.5).float()
    return ptv, oars

def extract_strips(x: torch.Tensor):
    return [(x[:, idx:idx+1] > 0.5).float() for idx in STRIP_IDXS]


band_mae_loss  = BandWeightedMAE()
band_ssim_loss = BandWeightedSSIM()


dmean_loss = DMeanLoss(dose_max_gy=DOSE_MAX_GY, reduction="mean")


VX_THR_DICT = {
    name: [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
    for name in ('Bladder', 'Rectum', 'Small_Intestine',
                 'Marrow', 'SpinalCord', 'FemurHead_L', 'FemurHead_R')
}
vx_loss = VxLoss(thr_dict_gy=VX_THR_DICT, dose_max_gy=DOSE_MAX_GY, reduction="mean")


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
            in_channels=IN_CHANNELS_2ND,
            out_channels=1,
            base_num_filters=base_filters_stage2,
            activation=activation_stage2,
        )

    def forward(self, x_2nd: torch.Tensor):
        ct    = x_2nd[:, 0:1]
        masks = x_2nd[:, 1:10]
        strips= x_2nd[:, 10:14]


        x_stage1    = torch.cat([ct, masks], dim=1)
        coarse_pred = self.stage1(x_stage1)


        x_stage2  = torch.cat([ct, masks, coarse_pred, strips], dim=1)
        dose_pred = self.stage2(x_stage2)

        return dose_pred, coarse_pred


def get_state_dict(model: torch.nn.Module):
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()


def train_one_epoch(model, loader, optimizer, scheduler, epoch, epochs,
                    ddp: bool = False, is_main: bool = True):
    model.train()
    loss_sum = 0.0
    train_mae_sum = 0.0
    train_dvh_sum = 0.0
    train_dvh_okbp_sum = 0.0
    train_ssim_sum = 0.0
    n_batches = 0

    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch}/{epochs} 训练中",
        dynamic_ncols=True,
        leave=is_main,
        disable=not is_main,
    )
    global_step = (epoch - 1) * max(1, len(loader))

    for it, (x, y) in enumerate(pbar, 1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        body = extract_body_mask(x)
        ptv, oars = extract_ptv_oars(x)
        strips = extract_strips(x)
        strip_union = strips[0].clone()
        for m in strips[1:]:
            strip_union = strip_union + m
        strip_union = (strip_union > 0.5).float()

        optimizer.zero_grad(set_to_none=True)


        dose_pred, coarse_pred = model(x)
        pred = dose_pred * body


        coarse_in_body = coarse_pred * body
        diff_coarse = (coarse_in_body - y).abs() * body
        loss_stage1_mae = diff_coarse.sum() / body.sum().clamp_min(1.0)


        body_mask = body
        diff_body = (pred - y).abs() * body_mask
        loss_body_mae = diff_body.sum() / body_mask.sum().clamp_min(1.0)


        loss_band_mae = band_mae_loss(pred, y, strips)
        loss_band_ssim = band_ssim_loss(pred, y, strips)


        loss_grad = gradient_l1_loss(pred, y, region_mask=strip_union)


        loss_dmean = dmean_loss(pred, y, oar_masks=oars)
        loss_vx = vx_loss(pred, y, oar_masks=oars)


        if it <= 5:
            print(f"[Debug Loss@Epoch{epoch} Iter{it}] "
                  f"s1_mae={loss_stage1_mae.item():.4f}, "
                  f"body_mae={loss_body_mae.item():.4f}, "
                  f"band_mae={loss_band_mae.item():.4f}, "
                  f"grad={loss_grad.item():.4f}, "
                  f"dmean={loss_dmean.item():.4f}, "
                  f"vx={loss_vx.item():.4f}")


        loss_total = (
                LAMBDA_STAGE1_WEIGHT * loss_stage1_mae
                + LAMBDA_STAGE2_WEIGHT * (LAMBDA_BODY_MAE * loss_body_mae
                                         + LAMBDA_BAND_MAE * loss_band_mae
                                         + LAMBDA_GRAD * loss_grad
                                         + LAMBDA_L_DMEAN * loss_dmean
                                         + LAMBDA_L_VX * loss_vx
                                          )
        )

        loss_total.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step(global_step)
        global_step += 1


        with torch.no_grad():
            tr_mae = mae_metric(pred, y, dose_max=DOSE_MAX_GY, region_mask=body)
            train_mae_sum += float(tr_mae.detach().cpu())

            tr_ssim = ssim3d_metric(pred, y, region_mask=body)
            train_ssim_sum += float(tr_ssim.detach().cpu())

            tr_dvh, _ = dvh_score(
                pred, y, ptv, oars,
                dose_max=DOSE_MAX_GY, rx_gy=45.0, ci_ref_frac=0.95
            )
            train_dvh_sum += float(tr_dvh.detach().cpu())

            tr_dvh_okbp, _ = dvh_score_openkbp(
                pred, y,
                ptv_mask=ptv,
                oar_masks=oars,
                dose_max=DOSE_MAX_GY,
                voxel_volume_mm3=None,
            )
            train_dvh_okbp_sum += float(tr_dvh_okbp.detach().cpu())
            n_batches += 1

        loss_val = float(loss_total.detach().cpu())
        loss_sum += loss_val
        avg_loss = loss_sum / it

        if is_main:
            pbar.set_postfix(
                loss=f"{avg_loss:.6f}",
                trDose=f"{train_mae_sum/max(1,n_batches):.4f}",
                trDVH=f"{train_dvh_sum/max(1,n_batches):.4f}",
                trDVH_OKBP=f"{train_dvh_okbp_sum / max(1, n_batches):.4f}",
                trSSIM=f"{train_ssim_sum/max(1,n_batches):.4f}",
            )


    if ddp and dist.is_initialized():
        vec = torch.tensor(
            [loss_sum, train_ssim_sum, train_mae_sum, train_dvh_sum, train_dvh_okbp_sum, n_batches],
            device=device, dtype=torch.float64,
        )
        dist.all_reduce(vec, op=dist.ReduceOp.SUM)
        loss_sum, train_ssim_sum, train_mae_sum, train_dvh_sum, train_dvh_okbp_sum, n_batches = vec.tolist()


    return (
        loss_sum / max(1, n_batches),
        train_ssim_sum / max(1, n_batches),
        train_mae_sum  / max(1, n_batches),
        train_dvh_sum  / max(1, n_batches),
        train_dvh_okbp_sum / max(1, n_batches),
    )


@torch.no_grad()
def validate_one_epoch(model, loader, epoch, epochs,
                       ddp: bool = False, is_main: bool = True):
    model.eval()
    loss_sum = mae_sum = ssim_sum = dvh_sum = dvh_okbp_sum = 0.0
    n = 0

    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch}/{epochs} 验证中",
        dynamic_ncols=True,
        leave=is_main,
        disable=not is_main,
    )
    for it, (x, y) in enumerate(pbar, 1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        body = extract_body_mask(x)
        ptv, oars = extract_ptv_oars(x)
        strips = extract_strips(x)
        strip_union = strips[0].clone()
        for m in strips[1:]:
            strip_union = strip_union + m
        strip_union = (strip_union > 0.5).float()


        dose_pred, coarse_pred = model(x)
        pred = dose_pred * body


        coarse_in_body = coarse_pred * body
        diff_coarse = (coarse_in_body - y).abs() * body
        loss_stage1_mae = diff_coarse.sum() / body.sum().clamp_min(1.0)


        body_mask = body
        diff_body = (pred - y).abs() * body_mask
        loss_body_mae = diff_body.sum() / body_mask.sum().clamp_min(1.0)


        loss_band_mae = band_mae_loss(pred, y, strips)
        loss_band_ssim = band_ssim_loss(pred, y, strips)


        loss_grad = gradient_l1_loss(pred, y, region_mask=strip_union)


        loss_dmean = dmean_loss(pred, y, oar_masks=oars)
        loss_vx = vx_loss(pred, y, oar_masks=oars)

        loss_total = (
                LAMBDA_STAGE1_WEIGHT * loss_stage1_mae
                + LAMBDA_STAGE2_WEIGHT * (LAMBDA_BODY_MAE * loss_body_mae
                                          + LAMBDA_BAND_MAE * loss_band_mae
                                          + LAMBDA_GRAD * loss_grad
                                          + LAMBDA_L_DMEAN * loss_dmean
                                          + LAMBDA_L_VX * loss_vx
                                          )
        )


        mae  = mae_metric(pred, y, dose_max=DOSE_MAX_GY, region_mask=body)
        ssim = ssim3d_metric(pred, y, region_mask=body)
        dvh, _ = dvh_score(pred, y, ptv, oars, dose_max=DOSE_MAX_GY, rx_gy=45.0, ci_ref_frac=0.95)

        dvh_okbp, _ = dvh_score_openkbp(
            pred, y,
            ptv_mask=ptv,
            oar_masks=oars,
            dose_max=DOSE_MAX_GY,
            voxel_volume_mm3=None,
        )

        loss_sum += float(loss_total.detach().cpu())
        mae_sum  += float(mae.detach().cpu())
        ssim_sum += float(ssim.detach().cpu())
        dvh_sum  += float(dvh.detach().cpu())
        dvh_okbp_sum += float(dvh_okbp.detach().cpu())
        n += 1

        if is_main:
            pbar.set_postfix(
                loss=f"{loss_sum/n:.6f}",
                MAE=f"{mae_sum/n:.4f}",
                SSIM=f"{ssim_sum/n:.4f}",
                DVH=f"{dvh_sum/n:.4f}",
                DVH_OKBP=f"{dvh_okbp_sum / n:.4f}",
            )


    if ddp and dist.is_initialized():
        vec = torch.tensor(
            [loss_sum, mae_sum, ssim_sum, dvh_sum, dvh_okbp_sum, n],
            device=device, dtype=torch.float64,
        )
        dist.all_reduce(vec, op=dist.ReduceOp.SUM)
        loss_sum, mae_sum, ssim_sum, dvh_sum, dvh_okbp_sum, n = vec.tolist()

    return loss_sum/max(1,n), mae_sum/max(1,n), ssim_sum/max(1,n), dvh_sum/max(1,n), dvh_okbp_sum/max(1, n)


if __name__ == "__main__":


    if USE_DDP:

        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    is_main = (rank == 0)


    train_set = FullVolumePTDataset2nd(
        TRAIN_PT,
        augment=False,
        aug_prob=0.5,
        noise_std=0.01,
        add_noise_on=['ct'],
    )
    val_set = FullVolumePTDataset2nd(VAL_PT, augment=False)


    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True) if USE_DDP else None
    val_sampler   = DistributedSampler(val_set,   num_replicas=world_size, rank=rank, shuffle=False) if USE_DDP else None

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=1,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        sampler=val_sampler,
        num_workers=0,
    )


    model = TwoStageDoseNet().to(device)


    if USE_DDP:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)


    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = EPOCHS * steps_per_epoch
    warmup_steps = max(1, int(total_steps * 0.05))
    base_lr = optimizer.param_groups[0]["lr"]
    eta_min = 1e-6

    def lr_lambda(global_step: int):
        if global_step < warmup_steps:
            return 0.1 + 0.9 * (global_step / max(1, warmup_steps))
        t = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return (eta_min/base_lr) + (1 - eta_min/base_lr) * 0.5 * (1 + math.cos(math.pi*t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    writer = SummaryWriter(TB_LOGDIR) if is_main else None
    best = float("inf")
    best_ep = -1
    g0 = time.time()

    for ep in range(1, EPOCHS + 1):
        if USE_DDP and train_sampler is not None:

            train_sampler.set_epoch(ep)

        tr_loss, tr_ssim, tr_mae, tr_dvh, tr_dvh_okbp = train_one_epoch(
            model, train_loader, optimizer, scheduler, ep, EPOCHS,
            ddp=USE_DDP, is_main=is_main,
        )
        va_loss, va_mae, va_ssim, va_dvh, va_dvh_okbp = validate_one_epoch(
            model, val_loader, ep, EPOCHS,
            ddp=USE_DDP, is_main=is_main,
        )

        if is_main:

            writer.add_scalar("Loss/train", tr_loss, ep)
            writer.add_scalar("Loss/val",   va_loss, ep)
            writer.add_scalar("DoseScore/train", tr_mae, ep)
            writer.add_scalar("DoseScore/val",   va_mae, ep)
            writer.add_scalar("DVHScore/train",  tr_dvh, ep)
            writer.add_scalar("DVHScore/val",    va_dvh, ep)
            writer.add_scalar("DVHScore_OpenKBP/train", tr_dvh_okbp, ep)
            writer.add_scalar("DVHScore_OpenKBP/val", va_dvh_okbp, ep)
            writer.add_scalar("SSIM/train", tr_ssim, ep)
            writer.add_scalar("SSIM/val",   va_ssim, ep)
            writer.add_scalar("opt/lr", optimizer.param_groups[0]["lr"], ep)

            print(
                f"[Epoch {ep}/{EPOCHS}] "
                f"train_loss={tr_loss:.6f} | train_DoseScore={tr_mae:.6f} | "
                f"train_DVHScore={tr_dvh:.6f} | train_DVHScore_OpenKBP={tr_dvh_okbp:.6f} | train_SSIM={tr_ssim:.4f} | "
                f"val_loss={va_loss:.6f} | val_DoseScore={va_mae:.6f} | "
                f"val_DVHScore={va_dvh:.6f} | val_DVHScore_OpenKBP={va_dvh_okbp:.6f} | val_SSIM={va_ssim:.4f}"
            )


            state = {
                "epoch": ep,
                "model": get_state_dict(model),
                "opt": optimizer.state_dict(),
                "val_loss": va_loss,
            }

            if va_mae < best - 1e-6:
                best, best_ep = va_mae, ep
                torch.save(state, CKPT_BEST)
                print(f"  [saved best @ {ep}] -> {CKPT_BEST}")


            if ep >= 5:
                ckpt_epoch_path = f"{CKPT_PREFIX}_epoch{ep}.pth"
                torch.save(state, ckpt_epoch_path)
                print(f"  [saved epoch {ep}] -> {ckpt_epoch_path}")


    if is_main:
        print(f"Done. best DoseScore={best:.6f} @ epoch {best_ep}, elapsed={(time.time()-g0)/60:.1f}min")
        if writer is not None:
            writer.close()


    if USE_DDP and dist.is_initialized():
        dist.destroy_process_group()
