# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""FieldGuide v2.3 —— 无门定强度版 / 开门版双模式

模式由 env PHY_FG_MODE 控制：
  gated（默认）= v2.1 行为：可学门 g_fg，初值 PHY_FG_G0（默认 0）
  fixed        = 无门：f2' = f2·(1+α·S)，α=PHY_FG_ALPHA（默认 0.5）

v2.1 教训：零初始化门 200 轮后 g_fg≈-0.009 从未打开，注入 ~1%，FG 指标贡献为零。
同病：g_cmee 六连负、GDK A=0 鞍点。门的保护理由（防 IR 稀释=断流误诊残案、
恒等开局=永不启用）均已失效；v1.11/12 的 fire 误报教训由羽流移位核+aux BCE 承担。

fixed 模式 = 核心假设"火源场引导找回 smoke"的直接检验：S∈[0,1] 有界，
最坏调制 ×1.5 且形状被 aux 约束，开局即生效，无冻结自由度。
v2.1（g_fg≈0）即事实上的 α=0 对照臂，与 fixed 臂天然成对。
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class FieldGuide(nn.Module):
    """羽流引导图生成 + 门控调制（v2.3 开门版）。"""

    K_SIZE = 7
    SIGMA_X = 1.0
    SIGMA_Y = 2.0
    MU_Y = 1.5

    def __init__(self):
        super().__init__()
        k = self.K_SIZE
        g = torch.arange(k, dtype=torch.float32) - (k - 1) / 2
        uu, vv = torch.meshgrid(g, g, indexing="xy")
        K0 = torch.exp(-(uu ** 2 / (2 * self.SIGMA_X ** 2)
                         + (vv - self.MU_Y) ** 2 / (2 * self.SIGMA_Y ** 2)))
        K0 = K0 / K0.sum()
        self.kernel = nn.Parameter(K0.view(1, 1, k, k))
        self.fg_mode = os.getenv("PHY_FG_MODE", "gated")     # gated=v2.1 | fixed=无门
        self.fg_alpha = float(os.getenv("PHY_FG_ALPHA", "0.5"))
        if self.fg_mode == "gated":
            g0 = float(os.getenv("PHY_FG_G0", "0"))
            self.g_fg = nn.Parameter(torch.full((1,), g0))

    @property
    def strength(self):
        """注入强度：gated=可学门 g_fg；fixed=固定 α。tasks 插入点统一走这里。
        getattr 回退：老 checkpoint 反序列化的实例没有 fg_mode/fg_alpha（__init__ 不重跑），
        一律按 gated + g_fg 处理，与 v2.1 行为逐位一致。"""
        if getattr(self, "fg_mode", "gated") == "gated":
            return self.g_fg
        return getattr(self, "fg_alpha", 0.5)

    def guide_map(self, t: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=t.device.type, enabled=False):
            S = F.conv2d(t.float(), self.kernel.float(), padding=self.K_SIZE // 2)
            if not torch.isfinite(S).all():
                S = S.nan_to_num(nan=0.0, posinf=1.0, neginf=0.0)
            S = S.clamp(0.0, 1.0)
        return S

    def forward(self, feat: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """f2' = f2 · (1 + a·S)；a = g_fg(gated) 或固定 α(fixed)。S∈[0,1] 有界。"""
        S = self.guide_map(t)
        return feat * (1 + self.strength * S.to(feat.dtype))

    @torch.no_grad()
    def guide_report(self) -> dict:
        k2 = self.kernel[0, 0]
        mass_bot = float(k2[self.K_SIZE // 2 + 1:].sum())
        mass_all = float(k2.sum()) + 1e-12
        mode = getattr(self, "fg_mode", "gated")
        rep = {
            "mode": mode,
            "alpha": getattr(self, "fg_alpha", 0.5) if mode == "fixed" else None,
            "kernel_sum": float(k2.sum()),
            "kernel_bot_mass_frac": round(mass_bot / mass_all, 4),
            "kernel_max": float(k2.max()),
        }
        if mode == "gated" and hasattr(self, "g_fg"):
            rep["g_fg"] = float(self.g_fg)
        return rep
