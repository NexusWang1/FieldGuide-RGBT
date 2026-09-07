# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""GDK —— 高斯形变卷积核（v2.0 核心模块）

设计（v2.0_design.md）：
    weight = base + A · (G − mean(G))
    G(u,v) = exp( −(u'²/2σx² + v'²/2σy²) ),  [u',v'] = R(θ)·[u−μx, v−μy]

    每核 6 个可学参数：a(幅度), θ(旋转), sx_raw/sy_raw(两轴展宽), μx/μy(中心偏移)。
    A = tanh(a)·(2·|base|max+ε)   —— 有界幅度
    σ = 0.6 + 1.2·sigmoid(s_raw)  —— 限定 [0.6,1.8]px（3×3 网格物理区间）
    μ = tanh(μ_raw)               —— 限定 [-1,1]px
    a 零初始化 ⇒ 开局与 base 卷积严格恒等；形变沿"各向异性高斯流形"低维展开。

    base 保持可训练 ⇒ v2.0 ⊃ v1.19（严格单变量：唯一新增即形变项）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianDeformConv2d(nn.Module):
    """3×3 卷积：可训练基础核 + 有界定向高斯形变项。"""

    SIGMA_MIN, SIGMA_SPAN = 0.6, 1.2  # σ ∈ [0.6, 1.8] px

    def __init__(self, base: nn.Conv2d):
        super().__init__()
        assert isinstance(base, nn.Conv2d) and base.kernel_size == (3, 3)
        assert base.stride == (1, 1) and base.groups == 1 and base.dilation == (1, 1)
        self.base = base  # 可训练，与 v1.19 基线一致
        co, ci = base.weight.shape[:2]
        self.a = nn.Parameter(torch.zeros(co, ci))
        self.theta = nn.Parameter(torch.zeros(co, ci))
        self.s_raw = nn.Parameter(torch.zeros(co, ci, 2))    # σx=σy=1.2px 起步
        self.mu_raw = nn.Parameter(torch.zeros(co, ci, 2))   # 中心居中起步
        amp0 = 2.0 * base.weight.detach().abs().amax(dim=(2, 3)) + 1e-3
        self.register_buffer("amp_max", amp0)                # (co,ci) 形变幅度上限
        g = torch.tensor([-1.0, 0.0, 1.0])
        uu, vv = torch.meshgrid(g, g, indexing="xy")         # uu=x 坐标, vv=y 坐标
        self.register_buffer("uu", uu)
        self.register_buffer("vv", vv)

    def deformed_weight(self) -> torch.Tensor:
        w = self.base.weight                                  # (co,ci,3,3)
        theta = self.theta[..., None, None]
        s = self.SIGMA_MIN + self.SIGMA_SPAN * torch.sigmoid(self.s_raw)
        sx = s[..., 0, None, None]
        sy = s[..., 1, None, None]
        mu = torch.tanh(self.mu_raw)
        mux = mu[..., 0, None, None]
        muy = mu[..., 1, None, None]
        u = self.uu - mux
        v = self.vv - muy
        ct, st = torch.cos(theta), torch.sin(theta)
        up = ct * u + st * v
        vp = -st * u + ct * v
        G = torch.exp(-(up ** 2 / (2 * sx ** 2) + vp ** 2 / (2 * sy ** 2)))
        G = G - G.mean(dim=(2, 3), keepdim=True)              # 零均值化（保直流响应不变）
        A = torch.tanh(self.a)[..., None, None] * self.amp_max[..., None, None]
        return w + A * G

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.deformed_weight(), self.base.bias, (1, 1), self.base.padding)

    @torch.no_grad()
    def deform_report(self) -> dict:
        """收官判读用：形变参数离开初值的程度。"""
        s = self.SIGMA_MIN + self.SIGMA_SPAN * torch.sigmoid(self.s_raw)
        return {
            "a_abs_mean": float(self.a.abs().mean()),
            "a_frac_moved_gt0.05": float((self.a.abs() > 0.05).float().mean()),
            "sigma_x_mean": float(s[..., 0].mean()),
            "sigma_y_mean": float(s[..., 1].mean()),
            "sigma_aniso_mean": float((s[..., 0] - s[..., 1]).abs().mean()),
            "theta_std": float(self.theta.std()),
            "mu_abs_mean": float(torch.tanh(self.mu_raw).abs().mean()),
        }


def wrap_conv3x3_gdk(root: nn.Module, Conv) -> int:
    """把 root 内所有 ultralytics Conv 的 3×3 普通卷积换成 GDK 版。返回替换数。"""
    n = 0
    for m in root.modules():
        if isinstance(m, Conv) and isinstance(getattr(m, "conv", None), nn.Conv2d):
            if m.conv.kernel_size == (3, 3) and m.conv.groups == 1 and m.conv.stride == (1, 1):
                m.conv = GaussianDeformConv2d(m.conv)
                n += 1
    return n


class GaussianFixedConv2d(nn.Module):
    """v2.0b —— 固定高斯核版（GFIX）：weight = base + A·(G0 − mean(G0))

    与 GDK 的唯一差异：G0 为固定的各向同性高斯（σ=1.2px，居中，不旋转），
    形状不可学；每核仅 1 个可学参数 a（幅度，零初始化 ⇒ 恒等开局）。
    σ=1.2 取 GDK σ∈[0.6,1.8] 的中点，保证两臂幅度量纲一致、可逐位对照。
    """

    SIGMA0 = 1.2  # 固定 σ（GDK 物理区间中点）

    def __init__(self, base: nn.Conv2d):
        super().__init__()
        assert isinstance(base, nn.Conv2d) and base.kernel_size == (3, 3)
        assert base.stride == (1, 1) and base.groups == 1 and base.dilation == (1, 1)
        self.base = base
        co, ci = base.weight.shape[:2]
        self.a = nn.Parameter(torch.zeros(co, ci))           # 唯一可学形变参数
        amp0 = 2.0 * base.weight.detach().abs().amax(dim=(2, 3)) + 1e-3
        self.register_buffer("amp_max", amp0)
        g = torch.tensor([-1.0, 0.0, 1.0])
        uu, vv = torch.meshgrid(g, g, indexing="xy")
        G0 = torch.exp(-(uu ** 2 + vv ** 2) / (2 * self.SIGMA0 ** 2))
        self.register_buffer("G0", G0 - G0.mean())           # 零均值化（保直流响应）

    def deformed_weight(self) -> torch.Tensor:
        A = torch.tanh(self.a)[..., None, None] * self.amp_max[..., None, None]
        return self.base.weight + A * self.G0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.deformed_weight(), self.base.bias, (1, 1), self.base.padding)

    @torch.no_grad()
    def deform_report(self) -> dict:
        return {
            "a_abs_mean": float(self.a.abs().mean()),
            "a_frac_moved_gt0.05": float((self.a.abs() > 0.05).float().mean()),
        }


def wrap_conv3x3_gfixed(root: nn.Module, Conv) -> int:
    """把 root 内所有 ultralytics Conv 的 3×3 普通卷积换成 GFIX 版。返回替换数。"""
    n = 0
    for m in root.modules():
        if isinstance(m, Conv) and isinstance(getattr(m, "conv", None), nn.Conv2d):
            if m.conv.kernel_size == (3, 3) and m.conv.groups == 1 and m.conv.stride == (1, 1):
                m.conv = GaussianFixedConv2d(m.conv)
                n += 1
    return n
