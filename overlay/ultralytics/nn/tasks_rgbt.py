# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Phy-Bridge YOLO26-P2 RGBT model.

Design constraints:
- keep DetectionModel.model intact so Ultralytics trainer/validator/loss/export still work;
- input tensor is 6 channels: RGB(3) + IR gray replicated to 3 channels;
- stage controls how many Phy-Bridge blocks are enabled after real IR is available:
    stage=1: PIIP + CMEE at P2 only (safe baseline)
    stage=2: + DCMA + BMGA at P3
    stage=3: + PSCMT + CPCF at P4
    stage=4: + DMFP at P5
- thermal-field add-ons (independent of stage, each separately ablatable):
    field={"guide"}:  HeatField from PIIP-enhanced IR; F gates P2/P3 features (fire recall)
    field={"calib"}:  soft logit penalty on smoke outside F (train-time only, anti-FP)
    field={"boost"}:  SmokeField from fused P3; dilated smoke saliency gates features (v-B)
    pscmt_res=True:   residual PSCMT fix  f4 = cpcf(r4 + g*pscmt(r4,i4)), g zero-init
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import C2PSA, C3k2, SPPF, Conv
from ultralytics.nn.modules.phy_bridge import BMGA, CMEE, CPCF, DCMA, DMFP, PIIP, PSCMT
from ultralytics.nn.modules.thermal_field import FieldGuide, HeatField, SmokeCalibrator, SmokeField
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
                 field=(), pscmt_res=False):
        self._rgbt_ready = False
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

        d, w, mc = SCALES[scale]
        c = lambda x: dv(min(int(x * w), mc), 8)
        n = lambda x: max(round(x * d), 1)

        # IR backbone mirrors YOLO26 backbone layers 0-10, but starts from replicated IR gray(3ch).
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

        # --- thermal-field add-ons: all gains zero-init => identity at start ---
        self.field = frozenset(field)
        self.pscmt_res = bool(pscmt_res)
        self.g_pscmt = nn.Parameter(torch.zeros(1))           # residual PSCMT gate
        self.heatfield = HeatField(c(256), sigma_init=0.12)   # source: PIIP-enhanced IR @P2
        self.guide2, self.guide3 = FieldGuide(), FieldGuide()
        self.smokefield = SmokeField(c(512), sigma_init=0.10)  # source: fused P3 features
        self.g_boost2 = nn.Parameter(torch.zeros(1))
        self.g_boost3 = nn.Parameter(torch.zeros(1))
        self.calib = SmokeCalibrator(nc if nc else 3, smoke_id=0)

        # v1.3: F 源辅助监督权重（0=关闭）。用 fire GT 框在 P2 分辨率上栅格化目标，
        # 对热响应核 t 做 BCE 监督——给 F 源独立目标函数，与检测 loss 解耦。
        self.aux_w = float(os.getenv("PHY_AUXW", "0.0"))
        self.fire_id = 1
        self._last_t = None
        self._aux_step = 0

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
        b = self.ir_backbone
        i2 = b[2](b[1](b[0](x)))
        i3 = b[4](b[3](i2))
        i4 = b[6](b[5](i3))
        i5 = b[10](b[9](b[8](b[7](i4))))
        return i2, i3, i4, i5

    def _forward_rgbt(self, x):
        assert x.ndim == 4 and x.shape[1] == 6, f"Expected 6-channel RGBT input, got {tuple(x.shape)}"
        rgb = x[:, :3].contiguous()
        ir = x[:, 3:].contiguous()

        r2, r3, r4, r5 = self._rgb_backbone(rgb)
        i2, i3, i4, i5 = self._ir_backbone(ir)

        if self.use_p2:
            i2p, _ = self.piip(i2)
            f2, _ = self.cmee(r2, i2p)
        else:
            i2p = i2
            f2 = r2

        if self.use_p3:
            r3a, i3a = self.dcma(r3, i3)
            f3 = self.bmga(r3a, i3a)
        else:
            f3 = r3

        # --- thermal field add-ons (guide / boost) ---
        field_map = None
        self._last_t = None
        if self.field:
            field_map, self._last_t = self.heatfield(i2p, ir)  # 物理锚定 F：P2 分辨率 (B,1,H2,W2)
            if "guide" in self.field:
                f2 = self.guide2(f2, field_map)
                f3 = self.guide3(f3, field_map)
            if "boost" in self.field:
                s = self.smokefield(f3)  # dilated smoke saliency at P3 resolution
                f3 = f3 * (1 + self.g_boost3 * s)
                f2 = f2 * (1 + self.g_boost2 * F.interpolate(
                    s, size=f2.shape[-2:], mode="bilinear", align_corners=False))

        if self.use_p4:
            # PSCMT/CPCF 含注意力矩阵乘，AMP(fp16) 下激活易溢出产生 NaN
            # （stage3 ep8/ep9 连续崩溃）。对这两个模块单独关闭 autocast，
            # 用 fp32 计算，其余部分保留 AMP 加速。
            with torch.autocast(device_type=x.device.type, enabled=False):
                if self.pscmt_res:
                    # 残差化修复（stage3b）：保留原始 RGB 特征，避免无热烟区被稀释
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
        out = m[-1]([p2, o3, o4, o5])
        if "calib" in self.field and field_map is not None:
            out = self.calib(out, field_map)  # train-time soft penalty on smoke logits outside F
        return out

    def forward(self, x, *args, **kwargs):
        if isinstance(x, dict):
            return self.loss(x, *args, **kwargs)
        if not getattr(self, "_rgbt_ready", False):
            return super().forward(x, *args, **kwargs)
        return self._forward_rgbt(x)

    # ------------------------------------------------------------------
    # v1.3: F 源辅助监督（fire GT -> P2 目标图 -> BCE(t, target)）
    # ------------------------------------------------------------------
    def _fire_target_map(self, batch, hw, device, dtype):
        """把 batch 里的 fire 框栅格化为 (B,1,H,W) 0/1 目标图（框内为 1，最小 2px）。"""
        H, W = hw
        B = int(batch["batch_idx"].max().item()) + 1 if batch["batch_idx"].numel() else batch["img"].shape[0]
        tgt = torch.zeros((B, 1, H, W), device=device, dtype=torch.float32)
        cls = batch["cls"].view(-1)
        bidx = batch["batch_idx"].view(-1).long()
        bbx = batch["bboxes"]  # (M,4) 归一化 xywh，与 batch['img'] 对齐
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
            tgt = self._fire_target_map(batch, t.shape[-2:], t.device, t.dtype)
            aux = F.binary_cross_entropy(t.float().clamp(1e-6, 1 - 1e-6), tgt.float())
            self._aux_step += 1
            if self._aux_step % 200 == 1:
                print(f"[aux] step={self._aux_step} field_bce={float(aux):.4f} "
                      f"w={self.aux_w} theta={float(self.heatfield.theta):.3f} "
                      f"sigma={float(self.heatfield.sigma):.4f}")
            loss = loss + self.aux_w * aux.to(loss.dtype)
        return loss, loss_items

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
