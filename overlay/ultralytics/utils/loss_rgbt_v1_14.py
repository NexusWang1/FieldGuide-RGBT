# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""v1.14 FireAnchored-Dual-STAL：fire 热一致 + smoke 火锚扩展的双向标签分配。

机制（两条项都只改 assigner 排序，特征/解码路径不动）：

1. fire 类（Dual-STAL）：
       Q = (1 - w) * IoU + w * C_T
   C_T(anchor) = 预测框内 IR 均值 / 全图 IR 最大值。小火点预测框罩住真实热核但
   IoU 低时被原版 TAL 当负例丢掉，C_T 把它捞回来。

2. smoke 类（FireAnchored-STAL）：
       Q = (1 - w_s) * IoU + w_s * G
   G(anchor) = 预测框对【膨胀 fire GT 掩码】的覆盖率 ∈ [0,1]。
   F 只从已确认标签的火点出发，按 dilate 倍膨胀覆盖物理上飘离火点的烟；
   无伴生火的烟 GT 的 G≈0，自动回退纯 IoU 排序。

3. 非 fire/smoke 类（person）与原版 TAL 逐位一致。
"""

from __future__ import annotations

import torch

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import TaskAlignedAssigner


def build_dilated_fire_mask(batch, hw, fire_id: int = 1, dilate: float = 3.0) -> torch.Tensor:
    """把 batch 里的 fire GT 框按 dilate 倍放大后栅格化为 (B,H,W) 0/1 掩码。"""
    H, W = hw
    img = batch["img"]
    B = img.shape[0]
    mask = torch.zeros((B, H, W), device=img.device, dtype=torch.float32)
    cls = batch["cls"].view(-1)
    bidx = batch["batch_idx"].view(-1).long()
    bbx = batch["bboxes"]  # (M,4) 归一化 xywh
    idx = (cls == fire_id).nonzero(as_tuple=True)[0]
    for j in idx.tolist():
        cx, cy, w, h = (float(v) for v in bbx[j])
        w2, h2 = w * dilate, h * dilate
        x0 = int(max(0.0, (cx - w2 / 2) * W)); x1 = int(min(W, (cx + w2 / 2) * W) + 0.5)
        y0 = int(max(0.0, (cy - h2 / 2) * H)); y1 = int(min(H, (cy + h2 / 2) * H) + 0.5)
        x1 = max(x1, x0 + 2); y1 = max(y1, y0 + 2)
        mask[bidx[j], y0:min(y1, H), x0:min(x1, W)] = 1.0
    return mask


class FireAnchoredDualAssigner(TaskAlignedAssigner):
    """fire 用 C_T 热一致项、smoke 用火锚覆盖项的 TAL 分配器。"""

    def __init__(self, *args, fire_id: int = 1, thermal_w: float = 0.25,
                 smoke_id: int = 0, smoke_w: float = 0.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.fire_id = int(fire_id)
        self.thermal_w = float(thermal_w)
        self.smoke_id = int(smoke_id)
        self.smoke_w = float(smoke_w)
        self.ir_img = None     # (B, H, W) float [0,1]，loss 每次调用前设置
        self.fire_mask = None  # (B, H, W) 0/1 膨胀 fire GT 掩码，loss 每次调用前设置

    @staticmethod
    def _box_mean_field(field: torch.Tensor, pd_bboxes: torch.Tensor) -> torch.Tensor:
        """积分图求每个预测框内 field 的均值。field (B,H,W) -> 返回 (B,A)。"""
        # NaN 框防腐：先 nan_to_num 再取索引
        pd_bboxes = pd_bboxes.nan_to_num(nan=0.0, posinf=1e4, neginf=0.0)
        B, A = pd_bboxes.shape[0], pd_bboxes.shape[1]
        H, W = field.shape[-2:]
        ii = torch.zeros((B, H + 1, W + 1), device=field.device, dtype=torch.float32)
        ii[:, 1:, 1:] = field.float().cumsum(dim=1).cumsum(dim=2)

        x1 = pd_bboxes[..., 0].clamp(0, W - 1).floor().long()
        y1 = pd_bboxes[..., 1].clamp(0, H - 1).floor().long()
        x2 = pd_bboxes[..., 2].clamp(1, W).ceil().long()   # exclusive
        y2 = pd_bboxes[..., 3].clamp(1, H).ceil().long()   # exclusive
        x2 = torch.maximum(x2, x1 + 1)
        y2 = torch.maximum(y2, y1 + 1)

        bi = torch.arange(B, device=field.device).unsqueeze(1).expand(B, A)
        s = ii[bi, y2, x2] - ii[bi, y1, x2] - ii[bi, y2, x1] + ii[bi, y1, x1]
        area = (x2 - x1) * (y2 - y1)
        return s / area.clamp(min=1).float()

    @torch.no_grad()
    def _thermal_conf(self, pd_bboxes: torch.Tensor) -> torch.Tensor:
        """C_T：框内 IR 均值 / 全图 IR 最大值，(B,A) ∈ [0,1]。"""
        ir = self.ir_img
        mean_ir = self._box_mean_field(ir, pd_bboxes)
        max_ir = ir.float().amax(dim=(1, 2)).clamp(min=1e-3)  # (B,)
        ct = (mean_ir / max_ir.unsqueeze(1)).clamp(0.0, 1.0)
        return ct.to(pd_bboxes.dtype)

    @torch.no_grad()
    def _fire_field_conf(self, pd_bboxes: torch.Tensor) -> torch.Tensor:
        """G：框内膨胀 fire 掩码覆盖率，(B,A) ∈ [0,1]。"""
        return self._box_mean_field(self.fire_mask, pd_bboxes).clamp(0.0, 1.0).to(pd_bboxes.dtype)

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        use_fire = self.ir_img is not None and self.thermal_w > 0
        use_smoke = self.fire_mask is not None and self.smoke_w > 0
        if not (use_fire or use_smoke):
            return super().get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt)

        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        batch_ind = torch.arange(self.bs, device=pd_scores.device)[:, None]
        bbox_scores[mask_gt] = pd_scores[batch_ind, :, gt_labels.squeeze(-1).long()][mask_gt]

        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        cls = gt_labels.squeeze(-1)                       # (B, M)
        q = overlaps
        if use_fire:
            ct = self._thermal_conf(pd_bboxes).unsqueeze(1)          # (B,1,A)
            is_fire = (cls == self.fire_id).unsqueeze(-1)            # (B,M,1)
            q = torch.where(is_fire, (1.0 - self.thermal_w) * overlaps + self.thermal_w * ct, q)
        if use_smoke:
            g = self._fire_field_conf(pd_bboxes).unsqueeze(1)        # (B,1,A)
            is_smoke = (cls == self.smoke_id).unsqueeze(-1)
            q = torch.where(is_smoke, (1.0 - self.smoke_w) * overlaps + self.smoke_w * g, q)

        align_metric = bbox_scores.pow(self.alpha) * q.clamp(min=0).pow(self.beta)
        # overlaps 原样返回：一锚点一 GT 消解与 target 归一化仍用真 IoU
        return align_metric, overlaps


class FireAnchoredV8Loss(v8DetectionLoss):
    """v8DetectionLoss，assigner 换成 FireAnchoredDualAssigner（fire + smoke 双向）。"""

    def __init__(self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2=None,
                 fire_id: int = 1, thermal_w: float = 0.25,
                 smoke_id: int = 0, smoke_w: float = 0.25, smoke_dilate: float = 3.0):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.fire_id = int(fire_id)
        self.smoke_dilate = float(smoke_dilate)
        self.assigner = FireAnchoredDualAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist(),
            topk2=tal_topk2,
            fire_id=fire_id,
            thermal_w=thermal_w,
            smoke_id=smoke_id,
            smoke_w=smoke_w,
        )

    def loss(self, preds, batch):
        img = batch.get("img", None)
        if img is not None and img.ndim == 4 and img.shape[1] >= 6:
            # 6 通道 RGBT：第 3 通道是复制的 IR 灰度
            self.assigner.ir_img = img[:, 3].detach().float()
            self.assigner.fire_mask = build_dilated_fire_mask(
                batch, img.shape[-2:], fire_id=self.fire_id, dilate=self.smoke_dilate)
        else:
            self.assigner.ir_img = None
            self.assigner.fire_mask = None
        return super().loss(preds, batch)
