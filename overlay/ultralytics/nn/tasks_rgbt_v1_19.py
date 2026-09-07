# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Phy-Bridge YOLO26-P2 RGBT model — v1.19 门控 CMEE 注入 + fp32 存档版。

v1.19 = v1.17b 结构 + 唯一结构改动：CMEE IR 支路改为零初始化门控注入

    f2 = r2e + g_cmee · i2e，  g_cmee = nn.Parameter(zeros(1))

    设计：g 零初始化 => 开局 = v1.15 冠军功能点（IR 注入关闭），
          g 的学习曲线本身即"网络是否需要 IR 注入"的直接读数。
          注意 g=0 时 i2e 支路梯度为 0（cmee.ir/guide、TFAM 断流），
          g 先动、支路后通——与 g_pscmt 同款零初始化门课程。
    工程沿用 v1.17b 两处补丁：f2 出口 _guard_finite("f2_cmee") 围堵+计数；
          save_model/strip_optimizer fp32 存档（trainer 侧）。

stage 控制与 v1.12~v1.14 相同：stage=1 PIIP+CMEE@P2；2 +DCMA+BMGA@P3；
3 +PSCMT+CPCF@P4；4 +DMFP@P5。pscmt_res 残差化开关保留。
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import C2PSA, C3k2, SPPF, Conv
from ultralytics.nn.modules.phy_bridge import BMGA, CMEE, CPCF, DCMA, DMFP, PIIP, PSCMT
from ultralytics.nn.modules.thermal_field_v1_14 import HeatField, TFAMBoost
from ultralytics.nn.tasks import DetectionModel

SCALES = {
    "n": [0.50, 0.25, 1024],
    "s": [0.50, 0.50, 1024],
    "m": [0.50, 1.00, 512],
    "l": [1.00, 1.00, 512],
    "x": [1.00, 1.50, 512],
}


def dv(x, d=8):
    return max(round(x / d) * d, d)


class PhyBridgeYOLO26(DetectionModel):
    def __init__(self, cfg="yolo26m-p2.yaml", ch=3, nc=None, verbose=True, scale="m", stage=1,
                 pscmt_res=False, tfam_ir=True,
                 dual_stal=False, stal_w=0.25,
                 smoke_stal=False, smoke_w=0.25, smoke_dilate=3.0):
        self._rgbt_ready = False
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

        d, w, mc = SCALES[scale]
        c = lambda x: dv(min(int(x * w), mc), 8)
        n = lambda x: max(round(x * d), 1)

        # IR backbone mirrors YOLO26 backbone layers 0-10, from replicated IR gray(3ch).
        self.ir_backbone = nn.ModuleList(
            [
                Conv(3, c(64), 3, 2),                    # 0 P1/2
                Conv(c(64), c(128), 3, 2),               # 1 P2/4
                C3k2(c(128), c(256), n=n(2), c3k=True, e=0.25),  # 2 P2
                Conv(c(256), c(256), 3, 2),              # 3 P3/8
                C3k2(c(256), c(512), n=n(2), c3k=True, e=0.25),  # 4 P3
                Conv(c(512), c(512), 3, 2),              # 5 P4/16
                C3k2(c(512), c(512), n=n(2), c3k=True),  # 6 P4
                Conv(c(512), c(1024), 3, 2),             # 7 P5/32
                C3k2(c(1024), c(1024), n=n(2), c3k=True),# 8
                SPPF(c(1024), c(1024)),                  # 9
                C2PSA(c(1024), c(1024), n=n(2)),         # 10
            ]
        )

        self.piip = PIIP(c(256))
        self.cmee = CMEE(c(256))
        self.dcma = DCMA(c(512))
        self.bmga = BMGA(c(512))
        self.pscmt = PSCMT(c(512))
        self.cpcf = CPCF(c(512))
        self.dmfp = DMFP(c(1024))

        self.pscmt_res = bool(pscmt_res)
        self.g_pscmt = nn.Parameter(torch.zeros(1))           # 残差化 PSCMT 门
        # v1.19：CMEE IR 注入门。零初始化 => 开局 f2 = r2e（恰等于冠军 v1.15
        # 的功能点），网络自己学 IR 注入剂量。
        self.g_cmee = nn.Parameter(torch.zeros(1))            # CMEE IR 注入门

        # --- v1.14 热场：F 源 + IR 侧 T-FAM 增强（全部零初始化 => 开局恒等） ---
        self.heatfield = HeatField(c(256), sigma_init=0.12)   # 源: PIIP 增强 IR @P2
        self.tfam = TFAMBoost()                               # i2p 热区增强
        self.tfam_ir = bool(tfam_ir)

        # F 源 aux 监督（fire GT 栅格化 BCE 全程约束 t；0=关闭）
        self.aux_w = float(os.getenv("PHY_AUXW", "0.0"))
        self.fire_id = 1
        self.smoke_id = 0
        self._last_t = None
        self._aux_step = 0

        # --- v1.14 监督层：Dual-STAL（fire）+ FireAnchored-STAL（smoke） ---
        self.dual_stal = bool(dual_stal)
        self.stal_w = float(stal_w)
        self.smoke_stal = bool(smoke_stal)
        self.smoke_w = float(smoke_w)
        self.smoke_dilate = float(smoke_dilate)

        self.set_stage(stage)
        self._rgbt_ready = True

    def set_stage(self, stage: int):
        self.stage = int(stage)
        self.use_p2 = self.stage >= 1
        self.use_p3 = self.stage >= 2
        self.use_p4 = self.stage >= 3
        self.use_p5 = self.stage >= 4

    @staticmethod
    def _up(x, ref, layer=None):
        if layer is not None:
            x = layer(x)
        return F.interpolate(x, size=ref.shape[-2:], mode="nearest") if x.shape[-2:] != ref.shape[-2:] else x

    def _rgb_backbone(self, x):
        m = self.model
        r2 = m[2](m[1](m[0](x)))       # P2/4
        r3 = m[4](m[3](r2))            # P3/8
        r4 = m[6](m[5](r3))            # P4/16
        r5 = m[10](m[9](m[8](m[7](r4))))  # P5/32
        return r2, r3, r4, r5

    def _ir_backbone(self, x):
        # fp32 防溢出（v1.14 两次 ep16-17 崩溃事故的根因修复）：fp16 下 IR 侧激活
        # 训练中后期稳步爬升越过 65504 → 单点 inf → BN 批次统计量把它放大成整通道
        # NaN，级联冲进检测头。IR 是 fire 信号主载体，整体关 autocast 用 fp32。
        b = self.ir_backbone
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            i2 = b[2](b[1](b[0](xf)))
            i3 = b[4](b[3](i2))
            i4 = b[6](b[5](i3))
            i5 = b[10](b[9](b[8](b[7](i4))))
        return i2, i3, i4, i5

    def _guard_finite(self, name, t):
        """分层 NaN/Inf 围堵：任何一层失守只退化当前 batch，不再整跑报废。"""
        if torch.isfinite(t).all():
            return t
        stats = getattr(self, "_guard_stats", None)
        if stats is None:
            stats = self._guard_stats = {}
        hits = stats[name] = stats.get(name, 0) + 1
        if hits <= 3 or hits % 500 == 0:
            n_bad = int((~torch.isfinite(t)).sum())
            print(f"[guard][WARN] {name} 含 {n_bad} 个 NaN/Inf (该层第 {hits} 次)，已 nan_to_num 兜底")
        return t.nan_to_num(nan=0.0, posinf=1e2, neginf=-1e2)

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

        # PIIP（同样 fp32 防溢出）：F 源（heatfield）和 P2 融合都需要 i2p
        if self.use_p2:
            with torch.autocast(device_type=x.device.type, enabled=False):
                i2p, _ = self.piip(i2.float())
            i2p = self._guard_finite("i2p", i2p)
        else:
            i2p = i2

        # F 场（融合前计算）：TFAM 增强或 aux 监督时需要。
        field_map = None
        self._last_t = None
        need_field = (self.tfam_ir and self.use_p2) or (self.training and self.aux_w > 0)
        if need_field:
            with torch.autocast(device_type=x.device.type, enabled=False):
                field_map, self._last_t = self.heatfield(i2p.float(), ir.float())  # (B,1,H2,W2)

        # v1.14 唯一特征层改动：T-FAM 掩码增强 IR 主通路（融合前，fire 专用）
        if self.tfam_ir and field_map is not None and self.use_p2:
            i2p = self._guard_finite("i2p_tfam", self.tfam(i2p, field_map))

        if self.use_p2:
            # ===== v1.19：CMEE 双路 + 零初始化注入门 =====
            # f2 = r2e + g_cmee·i2e，g 零初始化 => 开局=冠军 v1.15 功能点。
            r2e, i2e = self.cmee(r2, i2p)
            f2 = self._guard_finite("f2_cmee", r2e + self.g_cmee * i2e)
        else:
            f2 = r2

        if self.use_p3:
            r3a, i3a = self.dcma(r3, i3)
            f3 = self.bmga(r3a, i3a)
        else:
            f3 = r3

        if self.use_p4:
            # PSCMT/CPCF 含注意力矩阵乘，AMP(fp16) 下激活易溢出产生 NaN，fp32 计算。
            with torch.autocast(device_type=x.device.type, enabled=False):
                if self.pscmt_res:
                    r4f, i4f = r4.float(), i4.float()
                    f4 = self.cpcf(r4f + self.g_pscmt * self.pscmt(r4f, i4f))
                else:
                    f4 = self.cpcf(self.pscmt(r4.float(), i4.float()))
        else:
            f4 = r4

        if self.use_p5:
            f5 = self.dmfp(r5, l=i5)
        else:
            f5 = r5

        # YOLO26-P2 neck, layers 11-28 in cfg/models/26/yolo26-p2.yaml.
        m = self.model
        p4 = m[13](torch.cat([f4, self._up(f5, f4, m[11])], 1))
        p3 = m[16](torch.cat([f3, self._up(p4, f3, m[14])], 1))
        p2 = m[19](torch.cat([f2, self._up(p3, f2, m[17])], 1))
        o3 = m[22](torch.cat([p3, self._up(m[20](p2), p3)], 1))
        o4 = m[25](torch.cat([p4, self._up(m[23](o3), p4)], 1))
        o5 = m[28](torch.cat([f5, self._up(m[26](o4), f5)], 1))
        return m[-1]([p2, o3, o4, o5])

    def forward(self, x, *args, **kwargs):
        if isinstance(x, dict):
            return self.loss(x, *args, **kwargs)
        if not getattr(self, "_rgbt_ready", False):
            return super().forward(x, *args, **kwargs)
        return self._forward_rgbt(x)

    # ------------------------------------------------------------------
    # F 源辅助监督（fire GT -> P2 目标图 -> BCE(t, target)）
    # ------------------------------------------------------------------
    def _fire_target_map(self, batch, hw, device, dtype):
        """把 batch 里的 fire 框栅格化为 (B,1,H,W) 0/1 目标图（框内为 1，最小 2px）。"""
        H, W = hw
        B = batch["img"].shape[0]  # 按图像数开：末尾样本无 GT 时 batch_idx.max() 会偏小
        tgt = torch.zeros((B, 1, H, W), device=device, dtype=torch.float32)
        cls = batch["cls"].view(-1)
        bidx = batch["batch_idx"].view(-1).long()
        bbx = batch["bboxes"]  # (M,4) 归一化 xywh
        fire = (cls == self.fire_id).nonzero(as_tuple=True)[0]
        for j in fire.tolist():
            cx, cy, w, h = bbx[j].tolist()
            x0 = int(max(0, (cx - w / 2) * W)); x1 = int(min(W, (cx + w / 2) * W) + 0.5)
            y0 = int(max(0, (cy - h / 2) * H)); y1 = int(min(H, (cy + h / 2) * H) + 0.5)
            x1 = max(x1, x0 + 2); y1 = max(y1, y0 + 2)
            tgt[bidx[j], 0, y0:min(y1, H), x0:min(x1, W)] = 1.0
        return tgt.to(dtype)

    def loss(self, batch, preds=None):
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        preds = self.forward(batch["img"]) if preds is None else preds
        loss, loss_items = self.criterion(preds, batch)
        if self.training and self.aux_w > 0 and self._last_t is not None:
            t = self._last_t
            tgt = self._fire_target_map(batch, t.shape[-2:], t.device, torch.float32)
            # AMP 下 BCE 不安全（autocast 禁用项）：关 autocast 并用 float32 计算
            with torch.autocast(device_type=t.device.type, enabled=False):
                tc = t.float()
                if not torch.isfinite(tc).all():
                    print(f"[aux][WARN] t 含 NaN/Inf @step={self._aux_step}，已 nan_to_num 兜底")
                    tc = tc.nan_to_num(nan=0.0, posinf=1.0, neginf=0.0)
                aux = F.binary_cross_entropy(tc.clamp(1e-6, 1 - 1e-6), tgt.float())
            self._aux_step += 1
            loss = loss + self.aux_w * aux.to(loss.dtype)
        return loss, loss_items

    # ------------------------------------------------------------------
    # v1.14 监督层：Dual-STAL(fire) + FireAnchored-STAL(smoke)
    # ------------------------------------------------------------------
    def init_criterion(self):
        """Build the stock criterion, then swap in fire-anchored dual assigners."""
        from ultralytics.utils.loss import v8DetectionLoss
        from ultralytics.utils.loss_rgbt_v1_14 import FireAnchoredV8Loss

        crit = super().init_criterion()
        if not (self.dual_stal or self.smoke_stal):
            return crit

        def _mk(topk, topk2=None):
            return FireAnchoredV8Loss(
                self, tal_topk=topk, tal_topk2=topk2,
                fire_id=self.fire_id,
                thermal_w=self.stal_w if self.dual_stal else 0.0,
                smoke_id=self.smoke_id,
                smoke_w=self.smoke_w if self.smoke_stal else 0.0,
                smoke_dilate=self.smoke_dilate,
            )

        if hasattr(crit, "one2many") and hasattr(crit, "one2one"):
            # YOLO26 end2end：one2many/one2one 双分支都必须换，否则 one2one 头吃不到红利
            o2o = crit.one2one
            crit.one2many = _mk(topk=10)
            crit.one2one = _mk(topk=7, topk2=1)
            del o2o
        elif isinstance(crit, v8DetectionLoss):
            crit = _mk(topk=10)
        else:
            raise TypeError(f"FireAnchored-Dual-STAL 不支持的 criterion 类型: {type(crit)}")
        print(f"[fireanchored-stal] active: fire w={self.stal_w if self.dual_stal else 0.0} "
              f"Q=(1-w)*IoU+w*C_T | smoke w={self.smoke_w if self.smoke_stal else 0.0} "
              f"dilate={self.smoke_dilate} Q=(1-ws)*IoU+ws*G on criterion={type(crit).__name__}")
        return crit

    def load_pretrained(self, weights):
        # Accept either a .pt path or an already-loaded DetectionModel.
        from pathlib import Path

        if isinstance(weights, dict) and "model" in weights:
            src_model = weights["model"]
            self.load(weights)
        elif isinstance(weights, (str, Path)):
            weights = str(weights)
            self.load(weights)
            from ultralytics import YOLO

            src_model = YOLO(weights).model
        else:
            src_model = weights
            self.load(src_model)

        # Initialize IR backbone from the original 3-channel RGB backbone weights.
        ir_state = src_model.model[:11].state_dict()
        missing, unexpected = self.ir_backbone.load_state_dict(ir_state, strict=False)
        print(f"[OK] YOLO26 pretrained loaded; IR backbone missing={len(missing)} unexpected={len(unexpected)}")
