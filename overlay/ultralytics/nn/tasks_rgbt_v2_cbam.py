# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Phy-Bridge YOLO26-P2 RGBT — CBAM 注意力对照臂。

单变量设计：与基线（PHY_FG=0）逐位一致，唯一改动 =
在 P2/P3/P4/P5 融合特征出口各插一个 CBAM（通道+空间注意力）。
用途：同协议下给注意力路线一个代表行，与 FieldGuide 直注臂对比。

新增环境变量：PHY_CBAM(1/0，默认0)。

实现要点：
  - CBAM 模块懒创建于首次前向（通道数由特征形状决定），并在 __init__
    末尾以 6 通道 dummy 前向完成物化——保证优化器构建与权重加载之前参数已存在。
  - 与 FieldGuide 互斥使用：本臂 PHY_FG=0。
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from ultralytics.nn.modules.cbam import CBAM
from ultralytics.nn.tasks_rgbt_v2_3 import PhyBridgeYOLO26V23
from ultralytics.utils import LOGGER


class PhyBridgeYOLO26CBAM(PhyBridgeYOLO26V23):
    """v2.3 架构 + 颈部融合特征 CBAM（PHY_CBAM=1 启用）。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cbam_on = os.getenv("PHY_CBAM", "0").strip() not in {"0", "false", "False"}
        self.cbams = nn.ModuleDict()
        if self.cbam_on:
            self._materialize_cbam()

    def _cbam(self, tag: str, f: torch.Tensor) -> torch.Tensor:
        if not self.cbam_on:
            return f
        key = f"{tag}_c{f.shape[1]}"
        if key not in self.cbams:
            self.cbams[key] = CBAM(f.shape[1]).to(device=f.device, dtype=f.dtype)
        return self.cbams[key](f)

    @torch.no_grad()
    def _materialize_cbam(self):
        """6 通道 dummy 前向，让四个融合出口的 CBAM 在 __init__ 内物化。"""
        was_training = self.training
        self.eval()
        try:
            self._forward_rgbt(torch.zeros(1, 6, 256, 256))
        finally:
            if was_training:
                self.train()
        n = sum(p.numel() for p in self.cbams.parameters())
        print(f"[CBAM] 融合出口装配 {len(self.cbams)} 个 CBAM，新增参数 {n} 个: "
              f"{list(self.cbams.keys())}")

    def _forward_rgbt(self, x):
        assert x.ndim == 4 and x.shape[1] == 6, f"Expected 6-channel RGBT input, got {tuple(x.shape)}"
        rgb = x[:, :3].contiguous()
        ir = x[:, 3:].contiguous()

        r2, r3, r4, r5 = self._rgb_backbone(rgb)
        i2, i3, i4, i5 = self._ir_backbone(ir)
        i2 = self._guard_finite("i2", i2)
        i3 = self._guard_finite("i3", i3)
        i4 = self._guard_finite("i4", i4)
        i5 = self._guard_finite("i5", i5)

        if self.use_p2:
            with torch.autocast(device_type=x.device.type, enabled=False):
                i2p, _ = self.piip(i2.float())
            i2p = self._guard_finite("i2p", i2p)
        else:
            i2p = i2

        field_map = None
        self._last_t = None
        self._last_s = None
        need_field = ((self.tfam_ir and self.use_p2) or (self.training and self.aux_w > 0)
                      or (self.fg_on and self.use_p2))
        if need_field:
            with torch.autocast(device_type=x.device.type, enabled=False):
                field_map, self._last_t = self.heatfield(i2p.float(), ir.float())

        if self.tfam_ir and field_map is not None and self.use_p2:
            i2p = self._guard_finite("i2p_tfam", self.tfam(i2p, field_map))

        if self.use_p2:
            r2e, i2e = self.cmee(r2, i2p)
            if getattr(self, "_cmee_off", False):
                f2 = self._guard_finite("f2_cmee", r2e)
            elif getattr(self, "cmee_mode", "gated") == "fixed":
                f2 = self._guard_finite("f2_cmee", r2e + i2e)
            else:
                f2 = self._guard_finite("f2_cmee", r2e + self.g_cmee * i2e)
            if self.fg_on and self._last_t is not None:
                S = self.fieldguide.guide_map(self._last_t)
                self._last_s = S if (self.training and self.fg_aux_w > 0) else None
                f2 = self._guard_finite(
                    "f2_fg", f2 * (1 + self.fieldguide.strength * S.to(f2.dtype)))
            f2 = self._cbam("f2", f2)          # CBAM 插入点 1
        else:
            f2 = r2

        if self.use_p3:
            r3a, i3a = self.dcma(r3, i3)
            f3 = self.bmga(r3a, i3a)
            f3 = self._cbam("f3", f3)          # CBAM 插入点 2
        else:
            f3 = r3

        if self.use_p4:
            with torch.autocast(device_type=x.device.type, enabled=False):
                if self.pscmt_res:
                    r4f, i4f = r4.float(), i4.float()
                    f4 = self.cpcf(r4f + self.g_pscmt * self.pscmt(r4f, i4f))
                else:
                    f4 = self.cpcf(self.pscmt(r4.float(), i4.float()))
            f4 = self._cbam("f4", f4)          # CBAM 插入点 3
        else:
            f4 = r4

        if self.use_p5:
            f5 = self.dmfp(r5, l=i5)
            f5 = self._cbam("f5", f5)          # CBAM 插入点 4
        else:
            f5 = r5

        m = self.model
        p4 = m[13](torch.cat([f4, self._up(f5, f4, m[11])], 1))
        p3 = m[16](torch.cat([f3, self._up(p4, f3, m[14])], 1))
        p2 = m[19](torch.cat([f2, self._up(p3, f2, m[17])], 1))
        o3 = m[22](torch.cat([p3, self._up(m[20](p2), p3)], 1))
        o4 = m[25](torch.cat([p4, self._up(m[23](o3), p4)], 1))
        o5 = m[28](torch.cat([f5, self._up(m[26](o4), f5)], 1))
        return m[-1]([p2, o3, o4, o5])


def build_model_cbam(cfg="yolo26m-p2.yaml", ch=3, nc=3, verbose=True, scale="m"):
    """trainer 装配入口：v2.3 环境变量全集 + PHY_CBAM。"""
    stage = int(os.getenv("PHY_STAGE", "3"))
    model = PhyBridgeYOLO26CBAM(
        cfg=cfg, ch=ch, nc=nc, verbose=verbose, scale=scale, stage=stage,
        pscmt_res=os.getenv("PHY_PSCMT_RES", "0") == "1",
        tfam_ir=os.getenv("PHY_TFAM_IR", "1") == "1",
        dual_stal=os.getenv("PHY_DUALSTAL", "0") == "1",
        stal_w=float(os.getenv("PHY_STALW", "0.25")),
        smoke_stal=os.getenv("PHY_SMOKESTAL", "0") == "1",
        smoke_w=float(os.getenv("PHY_SMOKEW", "0.25")),
        smoke_dilate=float(os.getenv("PHY_SMOKEDIL", "3.0")),
    )
    LOGGER.info(f"PhyBridgeYOLO26 CBAM-arm cbam={model.cbam_on} fg={model.fg_on} "
                f"KMODE={os.getenv('PHY_KMODE', 'none')}")
    return model
