# -*- coding: utf-8 -*-
"""Loss functions for 3D dose prediction.

This module provides commonly used loss terms for voxel-wise dose regression and dose-shaping
objectives, including MAE/MSE, gradient-based losses, SSIM-based losses, band/strip weighted
aggregation, DVH-inspired surrogate losses, and a factory helper `get_loss`.

Author: Boda Ning
"""

from __future__ import annotations
import math
from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

                                  
class MAELoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = (pred - target).abs()
        if self.reduction == "sum": return loss.sum()
        if self.reduction == "none": return loss
        return loss.mean()


class MSELoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = (pred - target) ** 2
        if self.reduction == "sum": return loss.sum()
        if self.reduction == "none": return loss
        return loss.mean()

                                           
def gradient_l1_loss(pred: torch.Tensor,
                     target: torch.Tensor,
                     region_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
           
    with torch.cuda.amp.autocast(enabled=False):
        p = pred.float()
        g = target.float()

              
        dz_p = p[:, :, 1:, :, :] - p[:, :, :-1, :, :]
        dy_p = p[:, :, :, 1:, :] - p[:, :, :, :-1, :]
        dx_p = p[:, :, :, :, 1:] - p[:, :, :, :, :-1]

        dz_g = g[:, :, 1:, :, :] - g[:, :, :-1, :, :]
        dy_g = g[:, :, :, 1:, :] - g[:, :, :, :-1, :]
        dx_g = g[:, :, :, :, 1:] - g[:, :, :, :, :-1]

        diff_z = (dz_p - dz_g).abs()
        diff_y = (dy_p - dy_g).abs()
        diff_x = (dx_p - dx_g).abs()

        if region_mask is not None:
            m = (region_mask > 0.5).float()
            mz = m[:, :, 1:, :, :] * m[:, :, :-1, :, :]
            my = m[:, :, :, 1:, :] * m[:, :, :, :-1, :]
            mx = m[:, :, :, :, 1:] * m[:, :, :, :, :-1]

            loss_z = (diff_z * mz).sum() / mz.sum().clamp_min(1.0)
            loss_y = (diff_y * my).sum() / my.sum().clamp_min(1.0)
            loss_x = (diff_x * mx).sum() / mx.sum().clamp_min(1.0)
        else:
            loss_z = diff_z.mean()
            loss_y = diff_y.mean()
            loss_x = diff_x.mean()

        loss = (loss_z + loss_y + loss_x) / 3.0
        return loss.to(pred.dtype)


class GradientL1Loss(nn.Module):
                                         
    def __init__(self):
        super().__init__()
    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                region_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return gradient_l1_loss(pred, target, region_mask=region_mask)


                                          
def _gaussian_kernel_1d(k, s, device, dtype):
    c = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2
    g = torch.exp(-(c ** 2) / (2 * s ** 2)); g = g / g.sum().clamp_min(1e-12); return g
def _gaussian_kernel_3d(k, s, C, device, dtype):
    g1=_gaussian_kernel_1d(k,s,device,dtype); g2=g1.view(-1,1)@g1.view(1,-1)
    g3=g1.view(-1,1,1)*g2.view(1,k,k); g3=g3/g3.sum().clamp_min(1e-12)
    return g3.view(1,1,k,k,k).repeat(C,1,1,1,1)

def _ssim3d_map(x: torch.Tensor, y: torch.Tensor, k=7, s=1.5, K1=0.01, K2=0.03, L=1.0, eps=1e-8):
    B,C,D,H,W = x.shape; device, dtype = x.device, x.dtype
    kernel = _gaussian_kernel_3d(k, s, C, device, dtype); pad = k // 2
    filt = lambda z: F.conv3d(z, kernel, stride=1, padding=pad, groups=C)
    mu_x, mu_y = filt(x), filt(y)
    mu_x2, mu_y2, mu_xy = mu_x*mu_x, mu_y*mu_y, mu_x*mu_y
    sigma_x2, sigma_y2, sigma_xy = filt(x*x)-mu_x2, filt(y*y)-mu_y2, filt(x*y)-mu_xy
    C1, C2 = (K1*L)**2, (K2*L)**2
    num = (2*mu_xy + C1) * (2*sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return (num / (den + eps)).clamp(min=-1.0, max=1.0)

class SSIM3DLoss(nn.Module):
                                                         
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.k, self.s, self.K1, self.K2, self.L = 7, 1.5, 0.01, 0.03, 1.0
        self.reduction = reduction
    def forward(self, pred: torch.Tensor, target: torch.Tensor, region_mask: Optional[torch.Tensor]=None)->torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            x, y = pred.float(), target.float()
            ssim_map = _ssim3d_map(x, y, self.k, self.s, self.K1, self.K2, self.L)
            loss_map = 1.0 - ssim_map
            if self.reduction == "none": return loss_map.to(pred.dtype)
            if self.reduction == "sum":  return loss_map.sum().to(pred.dtype)
            if region_mask is None:      return loss_map.mean().to(pred.dtype)
            m = (region_mask > 0.5).float()
            while m.dim() < loss_map.dim(): m = m.unsqueeze(1)
            w = m.expand_as(loss_map); denom = w.sum().clamp_min(1.0)
            return ((loss_map * w).sum() / denom).to(pred.dtype)


                                              
class BandWeightedMAE(nn.Module):
           
    def __init__(self):
        super().__init__()

    @staticmethod
    def _stack_masks(band_masks, like: torch.Tensor) -> torch.Tensor:
                          
        if isinstance(band_masks, (list, tuple)):
            masks = [m.float() for m in band_masks]
            masks = torch.stack(masks, dim=1)
        else:
            masks = band_masks.float()
            if masks.dim() == 5:                      
                masks = masks.unsqueeze(1)
        return masks.to(device=like.device, dtype=like.dtype)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, band_masks) -> torch.Tensor:
                   
        m = self._stack_masks(band_masks, like=pred)                            
        err = (pred - target).abs().unsqueeze(1).expand_as(m)                   

        mask = (m > 0.5).to(err.dtype)
        B = pred.shape[0]

                                   
        err_flat = (err * mask).view(B, -1)                                    
        vox_flat = mask.view(B, -1).sum(dim=1)                         

                         
        loss_per = err_flat.sum(dim=1) / vox_flat.clamp_min(1.0)       

        valid = vox_flat > 0
        if valid.any():
            return loss_per[valid].mean()
        else:
            return pred.new_tensor(0.0)



class BandWeightedSSIM(nn.Module):
           
    def __init__(self):
        super().__init__()
        self.ssim = SSIM3DLoss(reduction="mean")                     

    @staticmethod
    def _iter_masks(band_masks):
        if isinstance(band_masks, (list, tuple)):
            for m in band_masks:
                yield m
        else:
                                     
            if band_masks.dim() == 6:
                for k in range(band_masks.shape[1]):
                    yield band_masks[:, k:k+1, ...]
            else:
                yield band_masks               

    def forward(self, pred: torch.Tensor, target: torch.Tensor, band_masks) -> torch.Tensor:
        B = pred.shape[0]
        per_patient_losses = []

        for b in range(B):
            num_b = pred.new_tensor(0.0)
            den_b = pred.new_tensor(0.0)

            for m in self._iter_masks(band_masks):
                mb = m[b:b+1]                                          
                vox = (mb > 0.5).sum()
                if vox <= 0:
                    continue
                                      
                l = self.ssim(pred[b:b+1], target[b:b+1], region_mask=mb)
                num_b = num_b + l * vox
                den_b = den_b + vox

            if den_b.item() > 0:
                per_patient_losses.append(num_b / den_b.clamp_min(1.0))

        if len(per_patient_losses) == 0:
            return pred.new_tensor(0.0)
        return torch.stack(per_patient_losses).mean()

                                         
class DVHPenaltyLoss(nn.Module):
    def __init__(self, thr_dict_gy: dict, dose_max_gy: float = 52.0,
                 alpha_per_gy: float = 10.0, beta_per_gy: float = 8.0,
                 reduction: str = "mean"):
        super().__init__()
        self.thr = thr_dict_gy; self.dose_max = float(dose_max_gy)
        self.alpha = float(alpha_per_gy); self.beta = float(beta_per_gy)
        self.reduction = reduction
    def _soft_Vx(self, dose_norm, mask, thr_gy):
        thr_norm = thr_gy / self.dose_max; a = self.alpha * self.dose_max
        v = torch.sigmoid(a * (dose_norm - thr_norm))
        num = (v * mask).sum(); den = mask.sum().clamp_min(1.0); return num / den
    def _soft_Dmax(self, dose_norm, mask):
        b = self.beta * self.dose_max; x = dose_norm[mask.bool()]
        if x.numel() == 0: return dose_norm.new_tensor(0.0)
        lse = torch.logsumexp(b * x, dim=0); return (lse - math.log(float(x.numel()))) / b
    def _Dmean(self, dose_norm, mask):
        x = dose_norm[mask.bool()];  return x.mean() if x.numel() > 0 else dose_norm.new_tensor(0.0)
    def forward(self, pred: torch.Tensor, target: torch.Tensor, oar_masks: Dict[str,torch.Tensor] | None=None):
        assert oar_masks is not None, "DVHPenaltyLoss requires oar_masks"
        B = pred.shape[0]; loss_sum = pred.new_zeros(())
        for organ, mask in oar_masks.items():
            if organ not in self.thr: continue
            m = (mask > 0.5)
            if m.sum() == 0: continue
            dmean_pred = self._Dmean(pred, m) * self.dose_max
            dmean_true = self._Dmean(target, m) * self.dose_max
            loss_sum = loss_sum + (dmean_pred - dmean_true).pow(2)
            if self.thr[organ].get("use_dmax", False):
                dmax_pred = self._soft_Dmax(pred, m) * self.dose_max
                x = target[m.bool()] * self.dose_max
                dmax_true = x.max() if x.numel() > 0 else pred.new_tensor(0.0)
                loss_sum = loss_sum + (dmax_pred - dmax_true).pow(2)
            for vx in self.thr[organ].get("Vx", []):
                v_pred = self._soft_Vx(pred, m, vx)
                v_true = ((target * self.dose_max) >= vx).float()
                v_true = (v_true * m.float()).sum() / m.sum().clamp_min(1.0)
                loss_sum = loss_sum + (v_pred - v_true).pow(2)
        if self.reduction == "sum": return loss_sum
        if self.reduction == "mean": return loss_sum / B
        return loss_sum

class DMeanLoss(nn.Module):
           
    def __init__(self, dose_max_gy: float = 52.0, reduction: str = "mean"):
        super().__init__()
        self.dose_max = float(dose_max_gy)
        self.reduction = reduction

    @staticmethod
    def _dmean_norm(dose_norm: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                   
        m = mask.bool()
        v = dose_norm[m]
        if v.numel() == 0:
            return dose_norm.new_tensor(0.0)
        return v.mean()

    def forward(
        self,
        pred: torch.Tensor,                         
        target: torch.Tensor,                       
        oar_masks: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        assert oar_masks is not None and len(oar_masks) > 0, "DMeanLoss requires oar_masks"
        with torch.cuda.amp.autocast(enabled=False):
            p = pred.float()
            g = target.float()
            B = p.shape[0]

            per_patient_losses = []

            for b in range(B):
                organ_errs = []
                for name, mask in oar_masks.items():
                                   
                    mb = (mask[b:b+1] > 0.5)
                    if mb.sum() == 0:
                        continue

                                            
                    dmean_pred = self._dmean_norm(p[b:b+1], mb) * self.dose_max
                    dmean_true = self._dmean_norm(g[b:b+1], mb) * self.dose_max

                                          
                    organ_errs.append((dmean_pred - dmean_true).abs())

                                                
                if len(organ_errs) > 0:
                    organ_errs_tensor = torch.stack(organ_errs)              
                    per_patient_losses.append(organ_errs_tensor.mean())

                                  
            if len(per_patient_losses) == 0:
                return p.new_tensor(0.0)

            losses = torch.stack(per_patient_losses)                      

            if self.reduction == "sum":
                return losses.sum()
            if self.reduction == "none":
                return losses
                              
            return losses.mean()


class VxLoss(nn.Module):
                                                  
    def __init__(self,
                 thr_dict_gy: Dict[str, list],                                       
                 dose_max_gy: float = 52.0,
                 alpha_per_gy: float = 10.0,
                 reduction: str = "mean"):
        super().__init__()
        self.thr = thr_dict_gy
        self.dose_max = float(dose_max_gy)
        self.alpha = float(alpha_per_gy)
        self.reduction = reduction

    def _soft_Vx(self,
                 dose_norm: torch.Tensor,                      
                 mask: torch.Tensor,                    
                 thr_gy: float) -> torch.Tensor:
                                              
        thr_norm = thr_gy / self.dose_max
        a = self.alpha * self.dose_max
        m = (mask > 0.5).float()
        v = torch.sigmoid(a * (dose_norm - thr_norm))
        num = (v * m).sum()
        den = m.sum().clamp_min(1.0)
        return num / den

    def forward(self,
                pred: torch.Tensor,                         
                target: torch.Tensor,                       
                oar_masks: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert oar_masks is not None and len(oar_masks) > 0, "VxLoss requires oar_masks"
        with torch.cuda.amp.autocast(enabled=False):
            p = pred.float()
            g = target.float()
            B = p.shape[0]
            loss_sum = p.new_zeros(())
            for organ, mask in oar_masks.items():
                if organ not in self.thr:
                    continue
                m = (mask > 0.5)
                if m.sum() == 0:
                    continue
                for vx in self.thr[organ]:
                    v_pred = self._soft_Vx(p, m, vx)
                                 
                    gt_bin = ((g * self.dose_max) >= float(vx)).float()
                    v_true = (gt_bin * m.float()).sum() / m.float().sum().clamp_min(1.0)
                    loss_sum = loss_sum + (v_pred - v_true).pow(2)
            if self.reduction == "sum":
                return loss_sum
            if self.reduction == "mean":
                return loss_sum / max(float(B), 1.0)
            return loss_sum

                                                           
class BaseSSIMLoss(nn.Module):
    def __init__(self, base_loss: nn.Module, w_ssim: float = 0.2):
        super().__init__(); self.base = base_loss; self.ssim = SSIM3DLoss(); self.w_ssim = float(w_ssim)
    def forward(self, pred, target, region_mask=None, **kwargs):
        base = self.base(pred, target)
        ssim_loss = self.ssim(pred, target, region_mask=region_mask)
        return (1.0 - self.w_ssim) * base + self.w_ssim * ssim_loss

class BaseSSIMDVHLoss(nn.Module):
    def __init__(self, base_loss: nn.Module, w_ssim: float, dvh_loss: DVHPenaltyLoss, w_dvh: float):
        super().__init__(); self.base=base_loss; self.ssim=SSIM3DLoss(); self.dvh=dvh_loss
        self.w_ssim=float(w_ssim); self.w_dvh=float(w_dvh)
    def forward(self, pred, target, oar_masks=None, region_mask=None):
        base = self.base(pred, target)
        ssim_loss = self.ssim(pred, target, region_mask=region_mask)
        dvh_loss  = self.dvh(pred, target, oar_masks=oar_masks)
        return (1.0 - self.w_ssim - self.w_dvh) * base + self.w_ssim * ssim_loss + self.w_dvh * dvh_loss

                                        
class ValueDVHLoss(nn.Module):
           
    def __init__(self, dose_max_gy: float = 52.0, reduction: str = "mean", include_empty: bool=False):
        super().__init__()
        self.dose_max = float(dose_max_gy)
        self.reduction = reduction
        self.include_empty = bool(include_empty)

    @staticmethod
    def _sorted_in_mask(x: torch.Tensor, mask: torch.Tensor):
        v = x[mask.bool()]
        if v.numel() == 0:
            return v, 0
        v_sorted, _ = torch.sort(v, descending=True)
        return v_sorted, v_sorted.numel()

    def forward(self,
                pred: torch.Tensor,                         
                target: torch.Tensor,                       
                roi_masks: Dict[str, torch.Tensor]                       
                ) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            p = (pred.float() * self.dose_max)         
            g = (target.float() * self.dose_max)
            B = p.shape[0]

            per_patient_losses = []

            for b in range(B):
                num = p.new_tensor(0.0)
                den = p.new_tensor(0.0)

                for name, m in roi_masks.items():
                    mb = m[b:b+1]               
                    ps, pn = self._sorted_in_mask(p[b:b+1], mb)
                    gs, gn = self._sorted_in_mask(g[b:b+1], mb)

                    if pn == 0 or gn == 0:
                                         
                        continue

                    n = min(pn, gn)
                    num = num + (ps[:n] - gs[:n]).abs().sum()
                    den = den + float(n)

                if den.item() > 0:
                    per_patient_losses.append(num / den.clamp_min(1.0))

            if len(per_patient_losses) == 0:
                return p.new_tensor(0.0)

            losses = torch.stack(per_patient_losses)             
            if self.reduction == "sum":
                return losses.sum()
            if self.reduction == "none":
                return losses
                                
            return losses.mean()


                                                  
class CriteriaDVHLoss(nn.Module):
           
    def __init__(self,
                 dose_max_gy: float = 52.0,
                 ptv_percentiles=(99.0, 95.0, 1.0),
                 use_oar_dmax: bool = True,
                 use_oar_mean: bool = True,
                 reduction: str = "mean"):
        super().__init__()
        self.dose_max = float(dose_max_gy)
        self.ptv_q = tuple(float(q) for q in ptv_percentiles)
        self.use_oar_dmax = bool(use_oar_dmax)
        self.use_oar_mean = bool(use_oar_mean)
        self.reduction = reduction

    @staticmethod
    def _safe_quantile(x: torch.Tensor, q: float):
        if x.numel() == 0:
            return x.new_tensor(0.0)
                                       
        return torch.quantile(x, q)

    def forward(self,
                pred: torch.Tensor,                                 
                target: torch.Tensor,                               
                ptv_mask: torch.Tensor,                      
                oar_masks: Dict[str, torch.Tensor]                       
                ) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            p = (pred.float() * self.dose_max)
            g = (target.float() * self.dose_max)
            B = p.shape[0]

            per_patient_losses = []

            for b in range(B):
                loss_terms = []

                                               
                if ptv_mask is not None:
                    pm = ptv_mask[b:b+1].bool()
                    pv = p[b:b+1][pm]
                    gv = g[b:b+1][pm]
                    if pv.numel() > 0 and gv.numel() > 0:
                        for q in self.ptv_q:
                            q01 = q / 100.0
                            loss_terms.append(
                                (self._safe_quantile(pv, q01) - self._safe_quantile(gv, q01)).abs()
                            )

                                                
                for name, m in oar_masks.items():
                    mb = m[b:b+1].bool()
                    pv = p[b:b+1][mb]
                    gv = g[b:b+1][mb]
                    if pv.numel() == 0 or gv.numel() == 0:
                        continue
                    if self.use_oar_dmax:
                        loss_terms.append((pv.max() - gv.max()).abs())              
                    if self.use_oar_mean:
                        loss_terms.append((pv.mean() - gv.mean()).abs())

                if len(loss_terms) > 0:
                    per_patient_losses.append(torch.stack(loss_terms).sum())

            if len(per_patient_losses) == 0:
                return p.new_tensor(0.0)

            losses = torch.stack(per_patient_losses)             
            if self.reduction == "sum":
                return losses.sum()
            if self.reduction == "none":
                return losses
                               
            return losses.mean()


                                             
class Base_V_C_DVHLoss(nn.Module):
           
    def __init__(self,
                 base_loss: nn.Module,
                 v_loss: ValueDVHLoss,   w_vdvh: float,
                 c_loss: CriteriaDVHLoss, w_cdvh: float):
        super().__init__()
        self.base = base_loss
        self.v_loss, self.w_vdvh = v_loss, float(w_vdvh)
        self.c_loss, self.w_cdvh = c_loss, float(w_cdvh)

    def forward(self, pred, target, *,
                ptv_mask=None,
                oar_masks: Dict[str, torch.Tensor] = None,
                region_mask=None):                            
        base = self.base(pred, target)

        v = 0.0
        c = 0.0
        if oar_masks is not None and len(oar_masks) > 0:
                                  
            roi_masks = dict(oar_masks)
            if ptv_mask is not None:
                roi_masks = {"PTV": ptv_mask, **roi_masks}
            v = self.v_loss(pred, target, roi_masks=roi_masks)

        if (ptv_mask is not None) and (oar_masks is not None) and (len(oar_masks) > 0):
            c = self.c_loss(pred, target, ptv_mask=ptv_mask, oar_masks=oar_masks)

        w0 = max(0.0, 1.0 - self.w_vdvh - self.w_cdvh)
        return w0 * base + self.w_vdvh * v + self.w_cdvh * c


                                            
class BaseSSIM_V_C_DVHLoss(nn.Module):
           
    def __init__(self,
                 base_loss: nn.Module,
                 w_ssim: float,
                 v_loss: ValueDVHLoss,
                 w_vdvh: float,
                 c_loss: CriteriaDVHLoss,
                 w_cdvh: float):
        super().__init__()
        self.base = base_loss
        self.ssim = SSIM3DLoss()
        self.v_loss = v_loss
        self.c_loss = c_loss
        self.w_ssim = float(w_ssim)
        self.w_vdvh = float(w_vdvh)
        self.w_cdvh = float(w_cdvh)

    def forward(self, pred, target, *,
                ptv_mask=None,
                oar_masks: Dict[str, torch.Tensor] = None,
                region_mask=None):
                                            
        base = self.base(pred, target)
        ssim_loss = self.ssim(pred, target, region_mask=region_mask)          

                                           
        v = 0.0
        c = 0.0
        if oar_masks is not None and len(oar_masks) > 0:
                                                   
            roi_masks = dict(oar_masks)
            if ptv_mask is not None:
                roi_masks = {"PTV": ptv_mask, **roi_masks}
            v = self.v_loss(pred, target, roi_masks=roi_masks)

        if (ptv_mask is not None) and (oar_masks is not None) and (len(oar_masks) > 0):
            c = self.c_loss(pred, target, ptv_mask=ptv_mask, oar_masks=oar_masks)

              
        w0 = max(0.0, 1.0 - self.w_ssim - self.w_vdvh - self.w_cdvh)
        return w0 * base + self.w_ssim * ssim_loss + self.w_vdvh * v + self.w_cdvh * c

                                  
def get_loss(name: str, **kwargs) -> nn.Module:
    k = name.strip().lower()
    if k in ("mae","l1"): return MAELoss(**{x:y for x,y in kwargs.items() if x=="reduction"})
    if k in ("mse","l2"): return MSELoss(**{x:y for x,y in kwargs.items() if x=="reduction"})
    if k in ("ssim3d","ssim"): return SSIM3DLoss()

    if k in ("mae+ssim","mse+ssim"):
        base = MSELoss(**{x:y for x,y in kwargs.items() if x=="reduction"}) if k.startswith("mse")             else MAELoss(**{x:y for x,y in kwargs.items() if x=="reduction"})
        return BaseSSIMLoss(base, w_ssim=float(kwargs.get("w_ssim",0.2)))

    if k in ("mae+ssim+dvh","mse+ssim+dvh"):
        base = MSELoss(**{x:y for x,y in kwargs.items() if x=="reduction"}) if k.startswith("mse")             else MAELoss(**{x:y for x,y in kwargs.items() if x=="reduction"})
        thr = kwargs.get("thr_dict_gy", {
            "Bladder":{"Vx":[30.0,40.0,45.0]},
            "Rectum":{"Vx":[30.0,40.0,45.0]},
            "Small_Intestine":{"Vx":[30.0], "use_dmax": True},
            "FemurHead_L":{"Vx":[30.0,40.0]},
            "FemurHead_R":{"Vx":[30.0,40.0]},
            "SpinalCord":{"Vx":[], "use_dmax": True},
            "Marrow":{"Vx":[30.0]},
        })
        dvh = DVHPenaltyLoss(thr_dict_gy=thr,
                             dose_max_gy=float(kwargs.get("dose_max_gy",52.0)),
                             alpha_per_gy=float(kwargs.get("alpha_per_gy",10.0)),
                             beta_per_gy=float(kwargs.get("beta_per_gy",8.0)),
                             reduction="mean")
        return BaseSSIMDVHLoss(base, w_ssim=float(kwargs.get("w_ssim",0.2)),
                               dvh_loss=dvh, w_dvh=float(kwargs.get("w_dvh",0.3)))

    if k in ("mae+vdvh+cdvh", "mse+vdvh+cdvh"):
        base = MSELoss(**{x: y for x, y in kwargs.items() if x == "reduction"}) if k.startswith("mse")            else MAELoss(**{x: y for x, y in kwargs.items() if x == "reduction"})
        dose_max_gy = float(kwargs.get("dose_max_gy", 52.0))
        v_loss = ValueDVHLoss(dose_max_gy=dose_max_gy, reduction="mean")
        c_loss = CriteriaDVHLoss(
            dose_max_gy=dose_max_gy,
            ptv_percentiles=kwargs.get("ptv_percentiles", (99.0, 95.0, 1.0)),
            use_oar_dmax=kwargs.get("use_oar_dmax", True),
            use_oar_mean=kwargs.get("use_oar_mean", True),
            reduction="sum"
        )
        return Base_V_C_DVHLoss(
            base_loss=base,
            v_loss=v_loss, w_vdvh=float(kwargs.get("w_vdvh", 0.20)),
            c_loss=c_loss, w_cdvh=float(kwargs.get("w_cdvh", 0.20)),
        )
    if k in ("mae+ssim+vdvh+cdvh", "mse+ssim+vdvh+cdvh"):
        base = MSELoss(**{x: y for x, y in kwargs.items() if x == "reduction"}) if k.startswith("mse")            else MAELoss(**{x: y for x, y in kwargs.items() if x == "reduction"})
        dose_max_gy = float(kwargs.get("dose_max_gy", 52.0))
                   
        v_loss = ValueDVHLoss(dose_max_gy=dose_max_gy, reduction="mean")
        c_loss = CriteriaDVHLoss(
            dose_max_gy=dose_max_gy,
            ptv_percentiles=kwargs.get("ptv_percentiles", (99.0, 95.0, 1.0)),
            use_oar_dmax=kwargs.get("use_oar_dmax", True),
            use_oar_mean=kwargs.get("use_oar_mean", True),
            reduction="sum"               
        )
        return BaseSSIM_V_C_DVHLoss(
            base_loss=base,
            w_ssim=float(kwargs.get("w_ssim", 0.2)),
            v_loss=v_loss, w_vdvh=float(kwargs.get("w_vdvh", 0.15)),
            c_loss=c_loss, w_cdvh=float(kwargs.get("w_cdvh", 0.15)),
        )
    raise ValueError(f"Unknown loss name: {name}")
