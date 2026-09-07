"""
================================================================================
Phy-Bridge-YOLO26 组合模块：热影响场（Thermal Field）
================================================================================

设计原则（2026-08-17 与用户确认）：
  1. 物理先验只决定结构形态与初始化值，所有尺度参数可学习；
  2. 全部增益以 gamma 零初始化残差接入——最坏情况模块退化为恒等映射；
  3. 三件功能独立开关，可单独消融：
     - guide : 非对称引导（IR 热场 F 调制 P2/P3 特征，治 fire 漏检）
     - calib : F 外 smoke 软降权（训练期 logit 惩罚，治 smoke 误报）
     - boost : 烟邻域 fire 增强（RGB 烟显著场扩散后增强特征，版本 B）
  4. 关系统计（audit_smoke_fire_relation, RGBT-3M train+val）只用于初始化：
     - smoke->fire 中心距离中位数 0.20 / 75 分位 0.31  -> F 场 sigma_init=0.12
     - fire->smoke 中位数 0.14 / 0.3 内覆盖 99.8%     -> boost sigma_init=0.10
     - 12.4%(val)~32.3%(train) 的烟无伴生火 -> calib 只做软惩罚，禁硬 veto
================================================================================
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_LN2 = 0.6931471805599453

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
# 1. 热影响场生成器（v1.2 物理锚定版）：IR 强度先验 × 可学习门控 -> 高斯扩散 -> F
# ==============================================================================
class HeatField(nn.Module):
    """v1.2：热响应核 t 由物理先验锚定，不再依赖随机初始化的卷积从零学习。

        t0 = relu(IR - θ) / (1 - θ)     # 绝对强度阈值化：只有显著热区才有响应
        t  = gate(i_feat) ⊙ t0          # 可学习门控只负责抑制伪热区/放大真热区
        F  = GaussianSpread(t, σ)       # σ 可学习，初始化 0.12（关系统计）

    - θ 可学习，clamp [0.3, 0.9]，初始化 0.6（IR 输入归一化 [0,1]；letterbox
      灰边 ≈0.45 < 0.6 不会误触发；批内不做 min-max，避免无火图被强行造出热点）。
    - gate 为 sigmoid 卷积，初始均值 ≈0.5 -> 开局 t ≈ 0.5·t0，物理上即合理。
    - 论文表述：物理先验（火焰 = IR 显著高温区）锚定场源，数据驱动学习门控与尺度。
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
        t = (self.gate(i_feat).float() * t0).to(i_feat.dtype)  # (B,1,H,W)
        spread = _gaussian_spread(t, self.sigma)
        # 峰值保持重标定：质归一化高斯会把点源峰值稀释 ~S² 倍（S=核和），
        # 按"场强峰值 = 源强峰值"重标定（比例系数 detach，梯度只走 spread 主干）。
        with torch.no_grad():
            scale = t.amax(dim=(-2, -1), keepdim=True) / spread.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        field = (spread * scale).clamp(0.0, 1.0)
        return field, t


# ==============================================================================
# 2. 非对称引导：F 调制 neck 特征（每层级一个零初始化增益门）
# ==============================================================================
class FieldGuide(nn.Module):
    """feat * (1 + gamma * F_upsampled)。gamma=0 时严格恒等。"""

    def __init__(self):
        super().__init__()
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, feat: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
        fmap = F.interpolate(field, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        return feat * (1 + self.g * fmap)


# ==============================================================================
# 3. F 外 smoke 软降权（训练期 scores 惩罚；推理期无操作——约束已学进权重）
# ==============================================================================
class SmokeCalibrator(nn.Module):
    """YOLO26 end2end 检测头训练态输出结构（debug_head_out.py 实测）：

        dict{one2many, one2one} -> 各含
            boxes  (B, 4, 34000)   已解码，不碰
            scores (B, nc, 34000)  每锚点类别分数 <- 唯一作用目标
            feats  [4 层特征图]     不碰

    34000 = 160^2+80^2+40^2+20^2，锚点按 P2->P5 层级顺序展平。
    将热场 F 插值到各层级尺寸、展平、拼接成 (B, 34000) 惩罚向量，
    只作用于 scores 的 smoke 通道。

    两种模式（首次调用按 scores 值域自动判定，各打印一次）：
      - 概率模式（scores ∈ [0,1]）：scores_smoke *= clamp(1 - p, floor, 1)
      - logit 模式（存在负值）：scores_smoke -= p
    p = (softplus(g) - ln2) * scale * (1 - F)，g 零初始化 -> p = 0，严格恒等。

    只在 training 模式生效：训练期约束让模型学会"F 外不报烟"，
    eval/val 输出（end2end top-300 解码）不携带空间坐标，无法也无需再施加。
    """

    def __init__(self, nc: int, smoke_id: int = 0, scale: float = 2.0, floor: float = 0.3):
        super().__init__()
        self.nc, self.sid = int(nc), int(smoke_id)
        self.scale, self.floor = float(scale), float(floor)
        self.g = nn.Parameter(torch.zeros(1))
        self._mode = None   # None=未判定, 'prob' 或 'logit'
        self._warned = False

    def _penalty_vec(self, field: torch.Tensor, n_anchors: int, ref: torch.Tensor) -> torch.Tensor:
        """F (B,1,H,W) -> (B, n_anchors)，按 P2..P5 层级展平拼接。"""
        H = field.shape[-2]
        sizes = [H, H // 2, H // 4, H // 8]
        if sum(h * h for h in sizes) != n_anchors:
            if not self._warned:
                print(f"[SmokeCalibrator][WARN] anchor layout mismatch: "
                      f"{sizes} vs n={n_anchors}，本次跳过惩罚")
                self._warned = True
            return None
        parts = [F.interpolate(field.float(), size=(h, h), mode="bilinear",
                               align_corners=False).flatten(1) for h in sizes]
        fcat = torch.cat(parts, 1)                                   # (B, 34000)
        p = (F.softplus(self.g) - _LN2) * self.scale * (1 - fcat)    # g=0 -> 0
        return p.to(ref.dtype)

    def _cal_scores(self, scores: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
        p = self._penalty_vec(field, scores.shape[-1], scores)
        if p is None:
            return scores
        if self._mode is None:
            with torch.no_grad():
                smin, smax = float(scores.min()), float(scores.max())
            self._mode = "prob" if (smin >= -1e-4 and smax <= 1.2) else "logit"
            print(f"[SmokeCalibrator] mode={self._mode} (scores range "
                  f"[{smin:.3f}, {smax:.3f}]), shape {tuple(scores.shape)}, smoke_id={self.sid}")
        if self._mode == "prob":
            w = (1 - p).clamp(min=self.floor, max=1.0).unsqueeze(1)  # (B,1,N)
            idx = self.sid
            smoke = scores[:, idx:idx + 1] * w
            return torch.cat([scores[:, :idx], smoke, scores[:, idx + 1:]], 1)
        else:
            idx = self.sid
            smoke = scores[:, idx:idx + 1] - p.unsqueeze(1)
            return torch.cat([scores[:, :idx], smoke, scores[:, idx + 1:]], 1)

    def _calibrate(self, out, field):
        if isinstance(out, dict):
            return {k: (self._cal_scores(v, field)
                        if k == "scores" and torch.is_tensor(v) and v.ndim == 3
                        else self._calibrate(v, field))
                    for k, v in out.items()}
        # list / tuple / feats / boxes 一律原样返回，绝不递归进特征图
        return out

    def forward(self, out, field):
        if not self.training:
            return out
        return self._calibrate(out, field)


# ==============================================================================
# 4. 烟邻域增强场（版本 B）：RGB 融合特征 -> 烟显著图 -> 可学习扩散 -> 增强门
# ==============================================================================
class SmokeField(nn.Module):
    """从 P3 融合特征预测烟雾显著图并扩散（sigma 可学习，初始化 0.10，
    对应关系统计 fire->smoke 中位距离 0.14 的七成）。输出 (B,1,H,W) ∈ [0,1]，
    由模型级零初始化门控接入 feat * (1 + g * S)，初始恒等。"""

    def __init__(self, c: int, sigma_init: float = 0.10):
        super().__init__()
        self.det = nn.Sequential(Conv(c, c // 4, 3), nn.Conv2d(c // 4, 1, 1), nn.Sigmoid())
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(float(sigma_init))))

    @property
    def sigma(self) -> torch.Tensor:
        return self.log_sigma.exp().clamp(0.02, 0.5)

    def forward(self, f_feat: torch.Tensor) -> torch.Tensor:
        s = self.det(f_feat)
        return _gaussian_spread(s, self.sigma).clamp(0.0, 1.0)
