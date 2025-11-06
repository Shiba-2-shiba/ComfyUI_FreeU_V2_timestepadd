# [ファイル名: utils.py]
# (前回提案のV4から変更なし)

import torch
import logging
import math

def Fourier_filter_gauss(x, radius_ratio: float, scale: float, hf_boost: float = 1.0):
    """
    指定された半径比（radius_ratio）とスケールに基づき、
    ガウシアンフィルタを周波数領域で適用します。
    LFは'scale'倍、HFは'hf_boost'倍します。
    """
    x_f = torch.fft.fftn(x.float(), dim=(-2, -1))
    x_f = torch.fft.fftshift(x_f, dim=(-2, -1))
    B, C, H, W = x_f.shape
    
    R = max(1, int(min(H, W) * radius_ratio))
    
    yy, xx = torch.meshgrid(
        torch.arange(H, device=x.device, dtype=torch.float32) - H // 2,
        torch.arange(W, device=x.device, dtype=torch.float32) - W // 2,
        indexing='ij'
    )
    dist2 = (yy**2 + xx**2)
    
    sigma_f = (R**2) / 2.0
    if sigma_f < 1e-6:
        sigma_f = 1e-6
        
    center = torch.exp(-dist2 / sigma_f)
    
    mask2d = (scale * center) + (hf_boost * (1.0 - center))
    mask = mask2d.view(1, 1, H, W)
    
    x_f_filtered = x_f * mask
    x_f_filtered = torch.fft.ifftshift(x_f_filtered, dim=(-2, -1))
    x_filtered = torch.fft.ifftn(x_f_filtered, dim=(-2, -1)).real
    
    return x_filtered.to(x.dtype)

def get_band_energy_stats(x, R: int):
    """
    指定された半径（R）内の低周波（LF）と高周波（HF）の
    平均エネルギー（振幅の2乗）を計算します。
    """
    xf = torch.fft.fftn(x.float(), dim=(-2, -1))
    xf = torch.fft.fftshift(xf, dim=(-2, -1))
    B, C, H, W = xf.shape
    
    yy, xx = torch.meshgrid(
        torch.arange(H, device=x.device, dtype=torch.float32) - H // 2,
        torch.arange(W, device=x.device, dtype=torch.float32) - W // 2,
        indexing='ij'
    )
    dist2 = (yy**2 + xx**2)
    
    lf_mask = (dist2 <= (R*R))
    hf_mask = ~lf_mask
    
    mag2 = (xf.real**2 + xf.imag**2)
    
    lf_e = mag2[lf_mask.expand_as(mag2)].mean().item() if lf_mask.any() else 0.0
    hf_e = mag2[hf_mask.expand_as(mag2)].mean().item() if hf_mask.any() else 0.0
    
    lf_pixels = lf_mask.sum().item()
    total_pixels = H * W
    cover_rate = (lf_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0
    
    return lf_e, hf_e, cover_rate