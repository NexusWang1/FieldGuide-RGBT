# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Phy-Bridge YOLO26-P2 RGBT model — v2.0 = v1.19 冠军架构 + GDK 高斯形变卷积核

与 v1.19 的唯一结构差异：
    IR backbone 的 P2 层（ir_backbone[2]，C3k2）内全部 3×3 卷积
    替换为 GaussianDeformConv2d（base 可训练 + 零初始化有界高斯形变项，
    开局与 v1.19 严格恒等 —— 严格单变量实验）。

新增监督：fire 响应 margin loss 作用于 GDK 层输出 i2（P2 IR 特征）：
  z-score 化响应图，fire GT 框内均值应高于框外至少 0.5σ，
  权重 PHY_GDKAUX（默认 0.02），步数线性 warmup PHY_GDKWARM（默认 20000 步）。

环境变量：v1.19 全集 + PHY_GDKAUX / PHY_GDKWARM / PHY_GDK（0=关闭 GDK，纯 v1.19 对照）
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import Conv
from ultralytics.nn.modules.gaussian_kernel import GaussianDeformConv2d, wrap_conv3x3_gdk
from ultralytics.nn.tasks_rgbt_v1_19 import PhyBridgeYOLO26
from ultralytics.utils import LOGGER


class PhyBridgeYOLO26V20(PhyBridgeYOLO26):
    """v1.19 + GDK。通过子类化保持 v1.19 单一事实源，diff 极小可审。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gdk_on = os.getenv("PHY_GDK", "1").strip() not in {"0", "false", "False"}
        self.gdk_aux_w = float(os.getenv("PHY_GDKAUX", "0.02"))
        self.gdk_warm = max(1, int(os.getenv("PHY_GDKWARM", "20000")))
        self._gdk_applied = False
        self._last_i2 = None
        self._gdk_step = 0

    # GDK 装配（必须在预训练权重加载之后：包壳会改 state_dict 键名）
    def apply_gdk(self) -> int:
        if self._gdk_applied or not self.gdk_on:
            return 0
        n = wrap_conv3x3_gdk(self.ir_backbone[2], Conv)
        self._gdk_applied = True
        n_param = sum(p.numel() for m in self.ir_backbone[2].modules()
                      if isinstance(m, GaussianDeformConv2d) for p in m.parameters()
                      if p.requires_grad) if n else 0
        print(f"[GDK] IR P2 C3k2 内 3x3 卷积替换 {n} 个，新增可学形变参数 {n_param} 个（零初始化=恒等开局）")
        return n

    def load_pretrained(self, weights):
        super().load_pretrained(weights)
        self.apply_gdk()

    # forward：捕获 GDK 层输出供 margin aux（其余与 v1.19 逐位一致）
    def _ir_backbone(self, x):
        b = self.ir_backbone
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            i2 = b[2](b[1](b[0](xf)))
            self._last_i2 = i2 if (self.training and self.gdk_aux_w > 0 and self._gdk_applied) else None
            i3 = b[4](b[3](i2))
            i4 = b[6](b[5](i3))
            i5 = b[10](b[9](b[8](b[7](i4))))
        return i2, i3, i4, i5

    # fire 响应 margin aux：GDK 形变参数的显式监督信号
    def _gdk_margin_loss(self, batch):
        i2 = self._last_i2
        if i2 is None:
            return None
        with torch.autocast(device_type=i2.device.type, enabled=False):
            r = i2.float().abs().mean(1, keepdim=True)                 # (B,1,H,W) 响应强度
            tgt = self._fire_target_map(batch, r.shape[-2:], r.device, torch.float32)
            has_fire = tgt.flatten(1).sum(1) > 0
            if not has_fire.any():
                return None
            mu = r.mean(dim=(2, 3), keepdim=True)
            sd = r.std(dim=(2, 3), keepdim=True) + 1e-6
            z = (r - mu) / sd
            inside = (z * tgt).sum(dim=(2, 3)) / (tgt.sum(dim=(2, 3)) + 1e-6)
            outside = (z * (1 - tgt)).sum(dim=(2, 3)) / ((1 - tgt).sum(dim=(2, 3)) + 1e-6)
            margin = F.relu(0.5 - (inside - outside)).squeeze(1)       # (B,)
            ramp = min(1.0, self._gdk_step / self.gdk_warm)
            self._gdk_step += 1
            return (margin[has_fire].mean() * ramp).to(i2.dtype)

    def loss(self, batch, preds=None):
        loss, loss_items = super().loss(batch, preds)
        if self.training and self.gdk_aux_w > 0:
            m = self._gdk_margin_loss(batch)
            if m is not None:
                loss = loss + self.gdk_aux_w * m.to(loss.dtype)
        return loss, loss_items

    @torch.no_grad()
    def gdk_report(self) -> dict:
        rep = {}
        for mi, m in enumerate(self.ir_backbone[2].modules()):
            if isinstance(m, GaussianDeformConv2d):
                for k, v in m.deform_report().items():
                    rep.setdefault(k, []).append(v)
        return {k: round(sum(v) / len(v), 4) for k, v in rep.items()}


def build_model_v20(cfg="yolo26m-p2.yaml", ch=3, nc=3, verbose=True, scale="m"):
    """trainer 装配入口：读 v1.19 同款环境变量 + GDK 开关。"""
    stage = int(os.getenv("PHY_STAGE", "3"))
    model = PhyBridgeYOLO26V20(
        cfg=cfg, ch=ch, nc=nc, verbose=verbose, scale=scale, stage=stage,
        pscmt_res=os.getenv("PHY_PSCMT_RES", "0") == "1",
        tfam_ir=os.getenv("PHY_TFAM_IR", "1") == "1",
        dual_stal=os.getenv("PHY_DUALSTAL", "0") == "1",
        stal_w=float(os.getenv("PHY_STALW", "0.25")),
        smoke_stal=os.getenv("PHY_SMOKESTAL", "0") == "1",
        smoke_w=float(os.getenv("PHY_SMOKEW", "0.25")),
        smoke_dilate=float(os.getenv("PHY_SMOKEDIL", "3.0")),
    )
    LOGGER.info(f"PhyBridgeYOLO26 v2.0 (v1.19 + GDK@IR-P2) stage={stage} "
                f"gdk={model.gdk_on} gdk_aux_w={model.gdk_aux_w} warm={model.gdk_warm}")
    return model
