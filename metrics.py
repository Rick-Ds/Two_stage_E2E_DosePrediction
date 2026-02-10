"""
metrics.py

Evaluation metrics for 3D dose prediction:
- MAE in Gy (optionally rescales normalized dose with dose_max)
- 3D SSIM (returns SSIM, not loss)
- DVH-based score from a fixed set of PTV and OAR endpoints (returns a scalar score and per-ROI details)

PTV endpoints: Dmean, D98, D2, V95, V100, Paddick CI (at ci_ref_frac × Rx), and ICRU-83 HI.
Selected OAR endpoints: Dmean and V30/V40 for bladder/rectum/femoral heads/marrow; Dmax for small intestine and spinal cord.

Author: Boda Ning
"""


from __future__ import annotations
from typing import Dict, Tuple, Optional
import torch
import torch.nn.functional as F
from loss.Loss_Func import SSIM3DLoss

from typing import List, Any, DefaultDict
from collections import defaultdict

rx = 45.0


@torch.no_grad()
def mae_metric(pred: torch.Tensor, target: torch.Tensor,
               dose_max: Optional[float]=None,
               region_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
    with torch.cuda.amp.autocast(enabled=False):
        pred = pred.float(); target = target.float()
        if dose_max is not None:
            pred = pred * float(dose_max); target = target * float(dose_max)
        if region_mask is None:
            return (pred - target).abs().mean()
        m = (_ensure_b1dhw(region_mask) > 0.5).float().to(pred.device)
        num = ((pred - target).abs() * m).sum()
        den = m.sum().clamp_min(1.0)
        return num / den

@torch.no_grad()
def ssim3d_metric(pred: torch.Tensor, target: torch.Tensor,
                  region_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
    with torch.cuda.amp.autocast(enabled=False):
        pred = pred.float(); target = target.float()

        loss = SSIM3DLoss(reduction="mean")(pred, target, region_mask=region_mask)
        return 1.0 - loss


def _ensure_b1dhw(t: torch.Tensor) -> torch.Tensor:
    if t.dim()==5: return t
    if t.dim()==4: return t.unsqueeze(1)
    if t.dim()==3: return t.unsqueeze(0).unsqueeze(0)
    raise ValueError(f"expect (B,1,D,H,W)-like, got {tuple(t.shape)}")


def _masked_values(dose: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    dose = _ensure_b1dhw(dose)
    mask = _ensure_b1dhw(mask).to(dtype=torch.bool, device=dose.device)
    vals = dose[mask].view(-1)
    return vals


def _vals_in_gy(dose: torch.Tensor, mask: Optional[torch.Tensor], dose_max: Optional[float]) -> torch.Tensor:
    if mask is None:
        vals = dose.view(-1)
    else:
        vals = _masked_values(dose, mask)
    if dose_max is not None:
        vals = vals * float(dose_max)
    return vals


def _scalar_from_channel(rx_channel: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    rx = rx_channel
    if rx.dim() == 5:
        pass
    elif rx.dim() == 4:
        rx = rx.unsqueeze(1)
    elif rx.dim() == 3:
        rx = rx.unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError(f"rx_channel shape unsupported: {tuple(rx.shape)}")
    rx = rx.to(dtype=torch.float32)

    if mask is not None:
        m = _ensure_b1dhw(mask).to(dtype=torch.bool, device=rx.device)
        if m.any():
            return rx[m].mean()
    return rx.mean()


@torch.no_grad()
def _D_percent(dose: torch.Tensor, mask: torch.Tensor, p: float, dose_max: Optional[float]) -> torch.Tensor:
    vals = _vals_in_gy(dose, mask, dose_max)
    if vals.numel() == 0:
        return torch.tensor(float("nan"), device=dose.device, dtype=dose.dtype)
    q = min(max(1.0 - float(p), 0.0), 1.0)
    return torch.quantile(vals, q)


@torch.no_grad()
def _D_mean(dose: torch.Tensor, mask: torch.Tensor, dose_max: Optional[float]) -> torch.Tensor:
    vals = _vals_in_gy(dose, mask, dose_max)
    if vals.numel() == 0:
        return torch.tensor(float("nan"), device=dose.device, dtype=dose.dtype)
    return vals.mean()


@torch.no_grad()
def _D_max(dose: torch.Tensor, mask: torch.Tensor, dose_max: Optional[float]) -> torch.Tensor:
    vals = _vals_in_gy(dose, mask, dose_max)
    if vals.numel() == 0:
        return torch.tensor(float("nan"), device=dose.device, dtype=dose.dtype)
    return vals.max()


@torch.no_grad()
def _V_greater_equal(dose: torch.Tensor, mask: torch.Tensor, thr_gy: float, dose_max: Optional[float]) -> torch.Tensor:
    vals = _vals_in_gy(dose, mask, dose_max)
    if vals.numel() == 0:
        return torch.tensor(float("nan"), device=dose.device, dtype=dose.dtype)
    return (vals >= float(thr_gy)).float().mean()


@torch.no_grad()
def _paddick_CI(dose: torch.Tensor, ptv_mask: torch.Tensor, ref_thr_gy: float, dose_max: Optional[float]) -> torch.Tensor:
    dose = _ensure_b1dhw(dose)
    ptv = _ensure_b1dhw(ptv_mask).to(dtype=torch.bool, device=dose.device)


    dose_vals = dose * float(dose_max) if dose_max is not None else dose
    piv = (dose_vals >= float(ref_thr_gy))

    tv = ptv.sum().float()
    pv = piv.sum().float()
    tv_pv = (ptv & piv).sum().float()

    denom = (tv * pv).clamp_min(1e-8)
    ci = (tv_pv * tv_pv) / denom
    ci = torch.where((tv <= 0) | (pv <= 0), torch.tensor(float("nan"), device=dose.device), ci)
    return ci


@torch.no_grad()
def _icru83_HI(dose: torch.Tensor, ptv_mask: torch.Tensor, dose_max: Optional[float]) -> torch.Tensor:
    d2  = _D_percent(dose, ptv_mask, 0.02, dose_max)
    d98 = _D_percent(dose, ptv_mask, 0.98, dose_max)
    d50 = _D_percent(dose, ptv_mask, 0.50, dose_max)
    denom = d50.abs().clamp_min(1e-8)
    hi = (d2 - d98) / denom
    return hi


@torch.no_grad()
def dvh_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    ptv_mask: torch.Tensor,
    oar_masks: Dict[str, torch.Tensor],
    dose_max: Optional[float] = None,
    rx_channel: Optional[torch.Tensor] = None,
    rx_gy: Optional[float] = None,
    ci_ref_frac: float = 0.95,
    skip_nan: bool = True,
) -> Tuple[torch.Tensor, Dict[str, Dict[str, Dict[str, float]]]]:
    with torch.cuda.amp.autocast(enabled=False):

        pred = pred.float()
        target = target.float()
        ptv_mask = ptv_mask.float()
        oar_masks = {k: v.float() for k, v in (oar_masks or {}).items()}
        if rx_channel is not None:
            rx_channel = rx_channel.float()

        device, dtype = pred.device, pred.dtype
        diffs = []
        details: Dict[str, Dict[str, Dict[str, float]]] = {}


        rx_val = None
        if rx_channel is not None:
            try:
                rx_val = _scalar_from_channel(rx_channel, mask=ptv_mask)
            except Exception:
                rx_val = _scalar_from_channel(rx_channel, mask=None)
        if rx_val is None and rx_gy is not None:
            rx_val = torch.tensor(float(rx_gy), device=device, dtype=torch.float32)
        if rx_val is None and dose_max is not None:
            rx_val = torch.tensor(float(dose_max), device=device, dtype=torch.float32)


        def _rec(entry, key, pv_t, gv_t):
            if (torch.isnan(pv_t) or torch.isnan(gv_t)) and skip_nan:
                return

            abs_diff = (pv_t - gv_t).abs()


            if key.startswith("D"):

                denom = float(rx) if rx is not None else (float(dose_max) if dose_max is not None else 1.0)
                norm_diff = abs_diff / denom
            elif key.startswith("V"):
                norm_diff = abs_diff * 100.0
            elif key in ["CI", "HI"]:
                norm_diff = abs_diff
            else:
                norm_diff = abs_diff

            if torch.isnan(norm_diff) and skip_nan:
                return

            diffs.append(norm_diff)

            entry[key] = {
                "pred": float(pv_t.detach().cpu().item()) if not torch.isnan(pv_t) else float("nan"),
                "gt": float(gv_t.detach().cpu().item()) if not torch.isnan(gv_t) else float("nan"),
                "abs_diff": float(abs_diff.detach().cpu().item()) if not torch.isnan(abs_diff) else float("nan"),
                "norm_diff": float(norm_diff.detach().cpu().item()) if not torch.isnan(norm_diff) else float("nan"),
            }


        name = "PTV"
        entry = {}


        for key, func in {
            "Dmean": lambda d, m: _D_mean(d, m, dose_max),
            "D98":   lambda d, m: _D_percent(d, m, 0.98, dose_max),
            "D2":    lambda d, m: _D_percent(d, m, 0.02, dose_max),
        }.items():
            pv = func(pred, ptv_mask); gv = func(target, ptv_mask); _rec(entry, key, pv, gv)


        if rx_val is not None:
            thr95 = float(ci_ref_frac) * float(rx_val)
            thr100 = 1.0 * float(rx_val)
            v95p = _V_greater_equal(pred, ptv_mask, thr95, dose_max)
            v95g = _V_greater_equal(target, ptv_mask, thr95, dose_max)
            _rec(entry, "V95", v95p, v95g)

            v100p = _V_greater_equal(pred, ptv_mask, thr100, dose_max)
            v100g = _V_greater_equal(target, ptv_mask, thr100, dose_max)
            _rec(entry, "V100", v100p, v100g)
        else:
            nan = torch.tensor(float("nan"), device=device, dtype=dtype)
            _rec(entry, "V95", nan, nan)
            _rec(entry, "V100", nan, nan)


        if rx_val is not None:
            ci_p = _paddick_CI(pred, ptv_mask, ref_thr_gy=float(ci_ref_frac) * float(rx_val), dose_max=dose_max)
            ci_g = _paddick_CI(target, ptv_mask, ref_thr_gy=float(ci_ref_frac) * float(rx_val), dose_max=dose_max)
            _rec(entry, "CI", ci_p, ci_g)
        else:
            nan = torch.tensor(float("nan"), device=device, dtype=dtype)
            _rec(entry, "CI", nan, nan)

        hi_p = _icru83_HI(pred, ptv_mask, dose_max)
        hi_g = _icru83_HI(target, ptv_mask, dose_max)
        _rec(entry, "HI", hi_p, hi_g)

        details[name] = entry


        group_mean_v = {"Bladder", "Rectum", "FemurHead_L", "FemurHead_R", "Marrow"}
        group_dmax   = {"Small_Intestine", "SpinalCord"}

        for oname, omask in (oar_masks or {}).items():
            entry_o = {}
            if oname in group_mean_v:
                dmp = _D_mean(pred, omask, dose_max); dmg = _D_mean(target, omask, dose_max)
                _rec(entry_o, "Dmean", dmp, dmg)

                v30p = _V_greater_equal(pred, omask, 30.0, dose_max)
                v30g = _V_greater_equal(target, omask, 30.0, dose_max)
                _rec(entry_o, "V30", v30p, v30g)

                v40p = _V_greater_equal(pred, omask, 40.0, dose_max)
                v40g = _V_greater_equal(target, omask, 40.0, dose_max)
                _rec(entry_o, "V40", v40p, v40g)

            elif oname in group_dmax:
                dmp = _D_max(pred, omask, dose_max); dmg = _D_max(target, omask, dose_max)
                _rec(entry_o, "Dmax", dmp, dmg)

            if entry_o:
                details[oname] = entry_o


        if len(diffs) == 0:
            score = torch.tensor(float("nan"), device=device, dtype=dtype)
        else:
            score = torch.stack(diffs).mean()

        return score, details

@torch.no_grad()
def dvh_score_openkbp(
    pred: torch.Tensor,
    target: torch.Tensor,
    ptv_mask: torch.Tensor,
    oar_masks: Dict[str, torch.Tensor],
    dose_max: Optional[float] = None,
    voxel_volume_mm3: Optional[float] = None,
    skip_nan: bool = True,
) -> Tuple[torch.Tensor, Dict[int, Dict[str, Dict[str, Dict[str, float]]]]]:
    with torch.cuda.amp.autocast(enabled=False):
        pred   = pred.float()
        target = target.float()
        ptv_mask = ptv_mask.float()
        oar_masks = {k: v.float() for k, v in (oar_masks or {}).items()}
        device = pred.device
        dtype  = pred.dtype

        B = pred.shape[0]


        if dose_max is not None:
            scale = float(dose_max)
            pred   = pred * scale
            target = target * scale

        all_diffs: List[torch.Tensor] = []
        details: Dict[int, Dict[str, Dict[str, Dict[str, float]]]] = {}

        def _safe_quantile(vals: torch.Tensor, q: float) -> torch.Tensor:
            if vals.numel() == 0:
                return torch.tensor(float("nan"), device=device, dtype=dtype)
            q = min(max(q, 0.0), 1.0)
            return torch.quantile(vals, q)

        for b in range(B):
            case_detail: Dict[str, Dict[str, Dict[str, float]]] = {}

            dose_p = pred[b:b+1]
            dose_g = target[b:b+1]


            m_ptv = (ptv_mask[b:b+1] > 0.5)
            vals_p = dose_p[m_ptv].view(-1)
            vals_g = dose_g[m_ptv].view(-1)

            roi_entry: Dict[str, Dict[str, float]] = {}
            if vals_p.numel() > 0 and vals_g.numel() > 0:

                d99_p = _safe_quantile(vals_p, 0.01)
                d99_g = _safe_quantile(vals_g, 0.01)
                diff = (d99_p - d99_g).abs()
                if not torch.isnan(diff) or not skip_nan:
                    all_diffs.append(diff)
                roi_entry["D_99"] = {
                    "pred": float(d99_p.detach().cpu().item()) if not torch.isnan(d99_p) else float("nan"),
                    "gt":   float(d99_g.detach().cpu().item()) if not torch.isnan(d99_g) else float("nan"),
                    "abs_diff": float(diff.detach().cpu().item()) if not torch.isnan(diff) else float("nan"),
                }


                d95_p = _safe_quantile(vals_p, 0.05)
                d95_g = _safe_quantile(vals_g, 0.05)
                diff = (d95_p - d95_g).abs()
                if not torch.isnan(diff) or not skip_nan:
                    all_diffs.append(diff)
                roi_entry["D_95"] = {
                    "pred": float(d95_p.detach().cpu().item()) if not torch.isnan(d95_p) else float("nan"),
                    "gt":   float(d95_g.detach().cpu().item()) if not torch.isnan(d95_g) else float("nan"),
                    "abs_diff": float(diff.detach().cpu().item()) if not torch.isnan(diff) else float("nan"),
                }


                d1_p = _safe_quantile(vals_p, 0.99)
                d1_g = _safe_quantile(vals_g, 0.99)
                diff = (d1_p - d1_g).abs()
                if not torch.isnan(diff) or not skip_nan:
                    all_diffs.append(diff)
                roi_entry["D_1"] = {
                    "pred": float(d1_p.detach().cpu().item()) if not torch.isnan(d1_p) else float("nan"),
                    "gt":   float(d1_g.detach().cpu().item()) if not torch.isnan(d1_g) else float("nan"),
                    "abs_diff": float(diff.detach().cpu().item()) if not torch.isnan(diff) else float("nan"),
                }

            if roi_entry:
                case_detail["PTV"] = roi_entry


            for oname, omask in oar_masks.items():
                m_oar = (omask[b:b+1] > 0.5)
                vals_p = dose_p[m_oar].view(-1)
                vals_g = dose_g[m_oar].view(-1)
                if vals_p.numel() == 0 or vals_g.numel() == 0:
                    continue

                entry_o: Dict[str, Dict[str, float]] = {}


                if voxel_volume_mm3 is not None and voxel_volume_mm3 > 0:

                    voxels_in_0_1cc = max(1, int(round(100.0 / float(voxel_volume_mm3))))
                    frac_vol = 100.0 - voxels_in_0_1cc / float(vals_p.numel()) * 100.0
                    q = frac_vol / 100.0
                    d01_p = _safe_quantile(vals_p, q)
                    d01_g = _safe_quantile(vals_g, q)
                else:

                    d01_p = vals_p.max()
                    d01_g = vals_g.max()

                diff = (d01_p - d01_g).abs()
                if not torch.isnan(diff) or not skip_nan:
                    all_diffs.append(diff)
                entry_o["D_0.1_cc"] = {
                    "pred": float(d01_p.detach().cpu().item()) if not torch.isnan(d01_p) else float("nan"),
                    "gt":   float(d01_g.detach().cpu().item()) if not torch.isnan(d01_g) else float("nan"),
                    "abs_diff": float(diff.detach().cpu().item()) if not torch.isnan(diff) else float("nan"),
                }


                mean_p = vals_p.mean()
                mean_g = vals_g.mean()
                diff = (mean_p - mean_g).abs()
                if not torch.isnan(diff) or not skip_nan:
                    all_diffs.append(diff)
                entry_o["mean"] = {
                    "pred": float(mean_p.detach().cpu().item()) if not torch.isnan(mean_p) else float("nan"),
                    "gt":   float(mean_g.detach().cpu().item()) if not torch.isnan(mean_g) else float("nan"),
                    "abs_diff": float(diff.detach().cpu().item()) if not torch.isnan(diff) else float("nan"),
                }

                case_detail[oname] = entry_o

            details[b] = case_detail

        if len(all_diffs) == 0:
            score = torch.tensor(float("nan"), device=device, dtype=dtype)
        else:
            score = torch.stack(all_diffs).mean()

        return score, details
