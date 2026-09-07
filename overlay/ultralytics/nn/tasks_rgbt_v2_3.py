# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Phy-Bridge YOLO26-P2 RGBT model — v2.3 = v2.x 胜出版本 + FieldGuide 羽流引导

v2.3 = v1.19 冠军底座 + 高斯核臂（PHY_KMODE 选 gdk/gfix/none）
+ FieldGuide（火源场 t → 羽流引导图 S → 调制 P2 融合特征 f2，smoke 寻回）。

新增环境变量：PHY_FG(1/0) / PHY_FGAUX(默认0.02) / PHY_FGWARM(默认20000步)
              PHY_KMODE(gdk|gfix|none，默认 none=v1.19 纯底座)
              PHY_CMEE_MODE(gated|fixed，默认 gated)
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import Conv
from ultralytics.nn.modules.field_guide_v2_3 import FieldGuide
from ultralytics.nn.modules.gaussian_kernel import (
    GaussianDeformConv2d,
    GaussianFixedConv2d,
    wrap_conv3x3_gdk,
    wrap_conv3x3_gfixed,
)
from ultralytics.nn.tasks_rgbt_v2_0 import PhyBridgeYOLO26V20
from ultralytics.utils import LOGGER


class PhyBridgeYOLO26V23(PhyBridgeYOLO26V20):
    """v2.x 底座 + FieldGuide。PHY_KMODE 决定 IR P2 卷积形态（含 none=不包壳）。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fg_on = os.getenv("PHY_FG", "1").strip() not in {"0", "false", "False"}
        self.fg_aux_w = float(os.getenv("PHY_FGAUX", "0.02"))
        self.fg_warm = max(1, int(os.getenv("PHY_FGWARM", "20000")))
        self.fieldguide = FieldGuide()
        self._last_s = None
        self._fg_step = 0
        # v2.4：CMEE 融合模式 gated(默认, g_cmee 可学门) | fixed(纯加法 r2e+i2e)
        self.cmee_mode = os.getenv("PHY_CMEE_MODE", "gated").strip().lower()

    # 高斯核臂选择：PHY_KMODE = gdk / gfix / none（默认 none = v1.19 纯底座）
    def apply_gdk(self) -> int:
        if self._gdk_applied or not self.gdk_on:
            return 0
        mode = os.getenv("PHY_KMODE", "none").strip().lower()
        if mode == "none":
            self._gdk_applied = True
            print("[KMODE] none：IR P2 保持原生 3x3 卷积（v1.19 底座）")
            return 0
        wrap = wrap_conv3x3_gdk if mode == "gdk" else wrap_conv3x3_gfixed
        cls = GaussianDeformConv2d if mode == "gdk" else GaussianFixedConv2d
        n = wrap(self.ir_backbone[2], Conv)
        self._gdk_applied = True
        n_param = sum(p.numel() for m in self.ir_backbone[2].modules()
                      if isinstance(m, cls) for p in m.parameters() if p.requires_grad) if n else 0
        print(f"[KMODE={mode}] IR P2 C3k2 内 3x3 卷积替换 {n} 个，新增参数 {n_param} 个")
        return n

    @torch.no_grad()
    def gdk_report(self) -> dict:
        rep = {}
        for m in self.ir_backbone[2].modules():
            if isinstance(m, (GaussianDeformConv2d, GaussianFixedConv2d)):
                for k, v in m.deform_report().items():
                    rep.setdefault(k, []).append(v)
        return {k: round(sum(v) / len(v), 4) for k, v in rep.items()}

    @torch.no_grad()
    def fg_report(self) -> dict:
        return self.fieldguide.guide_report()

    # forward：v1.19 主干原样，唯一插入点 = f2 融合出口加羽流引导
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

        # F 场（FieldGuide 也需要 t，need_field 增加 fg_on 条件）
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
                # v2.4：无门纯加法融合 f2 = r2e + i2e
                f2 = self._guard_finite("f2_cmee", r2e + i2e)
            else:
                f2 = self._guard_finite("f2_cmee", r2e + self.g_cmee * i2e)
            # ===== 插入点：羽流引导调制 P2 融合特征 =====
            # f2' = f2·(1 + strength·S)；strength=g_fg(gated) 或固定 α(fixed, v2.3)
            if self.fg_on and self._last_t is not None:
                S = self.fieldguide.guide_map(self._last_t)
                self._last_s = S if (self.training and self.fg_aux_w > 0) else None
                f2 = self._guard_finite(
                    "f2_fg", f2 * (1 + self.fieldguide.strength * S.to(f2.dtype)))
        else:
            f2 = r2

        if self.use_p3:
            r3a, i3a = self.dcma(r3, i3)
            f3 = self.bmga(r3a, i3a)
        else:
            f3 = r3

        if self.use_p4:
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

        m = self.model
        p4 = m[13](torch.cat([f4, self._up(f5, f4, m[11])], 1))
        p3 = m[16](torch.cat([f3, self._up(p4, f3, m[14])], 1))
        p2 = m[19](torch.cat([f2, self._up(p3, f2, m[17])], 1))
        o3 = m[22](torch.cat([p3, self._up(m[20](p2), p3)], 1))
        o4 = m[25](torch.cat([p4, self._up(m[23](o3), p4)], 1))
        o5 = m[28](torch.cat([f5, self._up(m[26](o4), f5)], 1))
        return m[-1]([p2, o3, o4, o5])

    # smoke GT 栅格化（与 _fire_target_map 同构，类 id 换成 smoke）
    def _smoke_target_map(self, batch, hw, device, dtype):
        H, W = hw
        B = batch["img"].shape[0]
        tgt = torch.zeros((B, 1, H, W), device=device, dtype=torch.float32)
        cls = batch["cls"].view(-1)
        bidx = batch["batch_idx"].view(-1).long()
        bbx = batch["bboxes"]
        smoke = (cls == self.smoke_id).nonzero(as_tuple=True)[0]
        for j in smoke.tolist():
            cx, cy, w, h = bbx[j].tolist()
            x0 = int(max(0, (cx - w / 2) * W)); x1 = int(min(W, (cx + w / 2) * W) + 0.5)
            y0 = int(max(0, (cy - h / 2) * H)); y1 = int(min(H, (cy + h / 2) * H) + 0.5)
            x1 = max(x1, x0 + 2); y1 = max(y1, y0 + 2)
            tgt[bidx[j], 0, y0:min(y1, H), x0:min(x1, W)] = 1.0
        return tgt.to(dtype)

    # loss：GDK margin aux + 羽流 aux（BCE(S, smoke GT)，warmup）
    def loss(self, batch, preds=None):
        loss, loss_items = super().loss(batch, preds)
        if self.training and self.fg_aux_w > 0 and self._last_s is not None:
            S = self._last_s
            tgt = self._smoke_target_map(batch, S.shape[-2:], S.device, torch.float32)
            with torch.autocast(device_type=S.device.type, enabled=False):
                Sc = S.float()
                if not torch.isfinite(Sc).all():
                    print(f"[fg-aux][WARN] S 含 NaN/Inf @step={self._fg_step}，已 nan_to_num 兜底")
                    Sc = Sc.nan_to_num(nan=0.0, posinf=1.0, neginf=0.0)
                aux = F.binary_cross_entropy(Sc.clamp(1e-6, 1 - 1e-6), tgt.float())
            ramp = min(1.0, self._fg_step / self.fg_warm)
            self._fg_step += 1
            loss = loss + (self.fg_aux_w * ramp * aux).to(loss.dtype)
        return loss, loss_items


def build_model_v21(cfg="yolo26m-p2.yaml", ch=3, nc=3, verbose=True, scale="m"):
    """trainer 装配入口：v1.19 环境变量 + GDK/FG 两组开关。"""
    stage = int(os.getenv("PHY_STAGE", "3"))
    model = PhyBridgeYOLO26V23(
        cfg=cfg, ch=ch, nc=nc, verbose=verbose, scale=scale, stage=stage,
        pscmt_res=os.getenv("PHY_PSCMT_RES", "0") == "1",
        tfam_ir=os.getenv("PHY_TFAM_IR", "1") == "1",
        dual_stal=os.getenv("PHY_DUALSTAL", "0") == "1",
        stal_w=float(os.getenv("PHY_STALW", "0.25")),
        smoke_stal=os.getenv("PHY_SMOKESTAL", "0") == "1",
        smoke_w=float(os.getenv("PHY_SMOKEW", "0.25")),
        smoke_dilate=float(os.getenv("PHY_SMOKEDIL", "3.0")),
    )
    LOGGER.info(f"PhyBridgeYOLO26 v2.3 (FieldGuide@f2 + KMODE={os.getenv('PHY_KMODE', 'none')}) "
                f"stage={stage} fg={model.fg_on} fg_aux_w={model.fg_aux_w} warm={model.fg_warm}")
    return model
