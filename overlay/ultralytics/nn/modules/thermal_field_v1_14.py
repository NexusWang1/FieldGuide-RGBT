"""
================================================================================
Phy-Bridge-YOLO26 v1.14 热场模块：HeatField + TFAMBoost
================================================================================

v1.14 结构修订（2026-08-18 与用户确认，B 方案）：
  - fire 主通路 = stage3 原样（IR→PIIP→CMEE/DCMA→检测头），唯一特征层改动是
    TFAMBoost：T-FAM 掩码形态的 IR 侧增强，用 F 场放大热区 IR 特征（α 零初始化，
    开局严格恒等）。
  - v1.11/v1.12 的 FieldGuide（调制融合流）与 v1.13 的 TFGuide（调制 RGB 支路）
    全部退役：前者把热背景放大成 fire 误报（fire P 崩到 0.19），后者让 fire
    失去热区放大而崩塌（fire mAP50 0.0045）。教训：F 只能增强 IR 自己，且只能
    在融合前。
  - SmokeCalibrator / SmokeField 一并退役（calib 负贡献已证实；smoke 召回改由
    监督层 FireAnchored-STAL 承担，见 loss_rgbt_v1_14.py）。
================================================================================
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ultralytics.nn.modules import Conv
except ImportError:
    class Conv(nn.Module):
        default_act = nn.SiLU()

        def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
            super().__init__()
            p = k // 2 if p is None else p
            self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False)
            self.bn = nn.BatchNorm2d(c2)
            self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        def forward(self, x):
            return self.act(self.bn(self.conv(x)))


def _gaussian_spread(t: torch.Tensor, sigma_norm: torch.Tensor, kmax: float = 0.40) -> torch.Tensor:
    """用可学习的高斯核（归一化 sigma）对 1 通道响应图做可分离扩散。

    t:          (B, 1, H, W) 响应图，值域 [0, 1]
    sigma_norm: 标量张量，归一化 sigma（相对特征图高度 H），梯度可回传
    kmax:       核支撑半径（归一化），固定不动，只决定 kernel 长度上限
    """
    B, _, H, W = t.shape
    sig_px = (sigma_norm * H).clamp(min=1.0)
    K = max(1, int(kmax * H))
    x = torch.arange(-K, K + 1, device=t.device, dtype=torch.float32)
    g = torch.exp(-(x ** 2) / (2 * sig_px ** 2))
    g = g / g.sum()
    tf = t.float()
    tf = F.conv2d(tf, g.view(1, 1, 1, -1), padding=(0, K))
    tf = F.conv2d(tf, g.view(1, 1, -1, 1), padding=(K, 0))
    return tf.to(t.dtype)


# ==============================================================================
# 1. 热影响场生成器（物理锚定版，与 v1.12/v1.13 相同）：IR 强度先验 × 可学习门控
# ==============================================================================
class HeatField(nn.Module):
    """热响应核 t 由物理先验锚定，不依赖随机初始化卷积从零学习。

        t0 = relu(IR - θ) / (1 - θ)     # 绝对强度阈值化：只有显著热区才有响应
        t  = gate(i_feat) ⊙ t0          # 可学习门控只负责抑制伪热区/放大真热区
        F  = GaussianSpread(t, σ)       # σ 可学习，初始化 0.12（关系统计）

    - θ 可学习，clamp [0.3, 0.9]，初始化 0.6（IR 输入归一化 [0,1]；letterbox
      灰边 ≈0.45 < 0.6 不会误触发；批内不做 min-max，避免无火图被强行造出热点）。
    - v1.14 起由 aux BCE（fire GT 栅格化目标）全程监督，见 tasks_rgbt_v1_14.loss。
    """

    def __init__(self, c: int, sigma_init: float = 0.12, theta_init: float = 0.6):
        super().__init__()
        self.gate = nn.Sequential(Conv(c, c // 4, 3), nn.Conv2d(c // 4, 1, 1), nn.Sigmoid())
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(float(sigma_init))))
        self.theta = nn.Parameter(torch.tensor(float(theta_init)))

    @property
    def sigma(self) -> torch.Tensor:
        return self.log_sigma.exp().clamp(0.02, 0.5)

    def forward(self, i_feat: torch.Tensor, ir_raw: torch.Tensor):
        # ir_raw: 原始 IR 输入 (B,3,H0,W0) 三通道复制灰度，值域 [0,1]；取单通道
        th = self.theta.clamp(0.3, 0.9)
        t0 = (ir_raw[:, :1].float() - th).clamp(min=0.0) / (1.0 - th)
        t0 = F.interpolate(t0, size=i_feat.shape[-2:], mode="bilinear", align_corners=False)
        # 防爆①（v1.14 ep16 崩溃事故）：i_feat 来自 AMP(fp16) 的 PIIP 输出，
        # 训练中后期个别 batch 会溢出 inf -> gate 的 BN/卷积产出 NaN -> t/field NaN，
        # 再经 TFAM 乘回主流 + BN running stats 前向期污染 -> 数轮后 assigner 硬崩。
        # 入口清洗 + 出口双侧清洗，保证 field 恒有限 ∈[0,1]（下游乘法就永远安全）。
        x = i_feat.float()
        if not torch.isfinite(x).all():
            x = x.nan_to_num(nan=0.0, posinf=1e2, neginf=-1e2)
        t = self.gate(x).float() * t0  # (B,1,H,W)，数学上 ∈[0,1]
        if not torch.isfinite(t).all():
            t = t.nan_to_num(nan=0.0, posinf=1.0, neginf=0.0)
        t = t.clamp(0.0, 1.0).to(i_feat.dtype)
        spread = _gaussian_spread(t, self.sigma)
        # 峰值保持重标定：质归一化高斯会把点源峰值稀释 ~S² 倍（S=核和），
        # 按"场强峰值 = 源强峰值"重标定（比例系数 detach，梯度只走 spread 主干）。
        with torch.no_grad():
            scale = t.amax(dim=(-2, -1), keepdim=True) / spread.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        field = spread * scale
        if not torch.isfinite(field).all():
            field = field.nan_to_num(nan=0.0, posinf=1.0, neginf=0.0)
        field = field.clamp(0.0, 1.0)
        return field, t


# ==============================================================================
# 2. TFAMBoost：T-FAM 掩码形态的 IR 侧热增强（v1.13 TFGuide 改名 + 职责变更）
# ==============================================================================
class TFAMBoost(nn.Module):
    """y = x * (1 + alpha * sigmoid(beta * (F - 0.5)))

    - 作用对象从 v1.13 的"纯 RGB 支路"改为【PIIP 之后的 IR 特征 i2p】，融合前使用：
      fire 信号 99% 在 IR（模态消融 RGB-only fire=0.0005），增强必须加在 IR 自己身上。
    - sigmoid 软掩码有界平滑，alpha 零初始化 => 开局严格恒等（=stage3），
      增益完全由数据驱动学出。
    - 热背景风险由两处兜底：F 源有 aux 监督（只在真火 GT 处响应），
      且 alpha 若学成负值则退化为热区抑制，模型自选方向。
    """

    def __init__(self, beta: float = 6.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = float(beta)

    def forward(self, feat: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
        fmap = F.interpolate(field, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        mask = torch.sigmoid(self.beta * (fmap - 0.5))
        # field 可能来自 fp32 的 heatfield（AMP 防爆而强制 fp32），掩码回投 feat 精度
        return feat * (1 + self.alpha * mask.to(feat.dtype))
