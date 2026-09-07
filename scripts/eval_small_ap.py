#!/usr/bin/env python
"""
eval_small_ap.py — RGBT 检测整体 / small / large AP 评估（论文官方口径）

口径：conf=0.001、iou=0.7、rect=True、fp32；small = GT 相对面积 < 0.01。
箱外 GT 被移除、落在移除 GT 上的预测记为 FP，两模型同口径横向对比有效。

    python eval_small_ap.py --weights <run>/weights/best.pt         --data configs/rgbt_full.yaml --mode rgbt --split val --device 0

结果 JSON 写入 <weights目录>/eval_small_ap_<split>_<mode>.json
同帧有火/无火子集分析见 eval_small_ap_cofire.py。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ultralytics.models.yolo.detect.train_rgbt import RGBTDetectionTrainer
from ultralytics.utils.metrics import ap_per_class, box_iou

try:
    from ultralytics.utils.nms import non_max_suppression
except ImportError:
    from ultralytics.utils.ops import non_max_suppression

NAMES = ["smoke", "fire", "person"]
IOU_THRS = np.linspace(0.5, 0.95, 10)
SMALL_AREA = 0.01
SMOKE, FIRE = 0, 1

BINS = ("all", "small", "large")


def build_loader(args, model):
    overrides = dict(
        model="yolo26m-p2.yaml",
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        cache=False,
        rect=True,
        mode="val",
    )
    trainer = RGBTDetectionTrainer(overrides=overrides)
    trainer.model = model
    val_path = str(Path(trainer.data["path"]) / trainer.data[args.split])
    loader = trainer.get_dataloader(val_path, batch_size=args.batch, rank=-1, mode="val")
    return loader


def load_model(weights: str, device: str):
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    model = (ckpt.get("model") if ckpt.get("model") is not None else ckpt["ema"]).float().eval().to(device)
    names = getattr(model, "names", None) or {i: n for i, n in enumerate(NAMES)}
    return model, names


def model_in_channels(model) -> int:
    try:
        first = model.model[0]
        conv = getattr(first, "conv", first)
        return int(conv.in_channels)
    except Exception:
        return -1


def is_stock_model(model) -> bool:
    return type(model).__name__ == "DetectionModel"


def match_predictions_local(pred_cls, true_cls, iou):
    n_pred = pred_cls.shape[0]
    correct = np.zeros((n_pred, len(IOU_THRS)), dtype=bool)
    if true_cls.shape[0] == 0 or n_pred == 0:
        return correct
    iou_np = iou.cpu().numpy() if torch.is_tensor(iou) else np.asarray(iou)
    pcls = pred_cls.cpu().numpy() if torch.is_tensor(pred_cls) else np.asarray(pred_cls)
    tcls = true_cls.cpu().numpy() if torch.is_tensor(true_cls) else np.asarray(true_cls)
    for ti, thr in enumerate(IOU_THRS):
        used = np.zeros(tcls.shape[0], dtype=bool)
        for pi in range(n_pred):
            cand = np.where((tcls == pcls[pi]) & (~used) & (iou_np[:, pi] >= thr))[0]
            if cand.size:
                gi = cand[np.argmax(iou_np[cand, pi])]
                used[gi] = True
                correct[pi, ti] = True
    return correct


def collect_image_stats(pred, gt_cls, gt_box_norm, img_hw, stats_bins):
    """与原版一致，另按"同帧是否有 fire GT"对 smoke GT 拆两个子集。"""
    h, w = img_hw
    if pred is None or len(pred) == 0:
        pred_xy = np.zeros((0, 4))
        conf = np.zeros((0,))
        pcls = np.zeros((0,))
    else:
        pred = pred[np.argsort(-pred[:, 4])]
        scale = np.array([w, h, w, h], dtype=np.float32)
        pred_xy = np.clip(pred[:, :4] / scale, 0, 1)
        conf = pred[:, 4]
        pcls = pred[:, 5]

    areas = (gt_box_norm[:, 2] - gt_box_norm[:, 0]) * (gt_box_norm[:, 3] - gt_box_norm[:, 1])

    bins = {
        "all": np.ones(len(gt_cls), dtype=bool),
        "small": areas < SMALL_AREA,
        "large": areas >= SMALL_AREA,
    }
    for bname, bmask in bins.items():
        gcls_b = gt_cls[bmask]
        gbox_b = gt_box_norm[bmask]
        if len(gcls_b):
            iou = box_iou(torch.from_numpy(gbox_b), torch.from_numpy(pred_xy)) if len(pred_xy) else torch.zeros((len(gcls_b), 0))
            correct = match_predictions_local(
                torch.from_numpy(pcls), torch.from_numpy(gcls_b), iou
            ) if len(pred_xy) else np.zeros((0, len(IOU_THRS)), dtype=bool)
        else:
            correct = np.zeros((len(pcls), len(IOU_THRS)), dtype=bool)
        s = stats_bins[bname]
        s["tp"].append(correct)
        s["conf"].append(conf)
        s["pcls"].append(pcls)
        s["tcls"].append(gcls_b)


def evaluate_bin(s, names):
    tp = np.concatenate(s["tp"]) if s["tp"] else np.zeros((0, len(IOU_THRS)), dtype=bool)
    conf = np.concatenate(s["conf"]) if s["conf"] else np.zeros((0,))
    pcls = np.concatenate(s["pcls"]) if s["pcls"] else np.zeros((0,))
    tcls = np.concatenate(s["tcls"]) if s["tcls"] else np.zeros((0,))
    if tp.shape[0] == 0 and tcls.shape[0] == 0:
        return {}
    out = ap_per_class(tp, conf, pcls, tcls, names=names)
    p, r, ap, uc = out[2], out[3], out[5], out[6]
    res = {}
    for ci, c in enumerate(np.atleast_1d(uc)):
        c = int(c)
        res[names.get(c, str(c))] = {
            "precision": float(p[ci]),
            "recall": float(r[ci]),
            "mAP50": float(ap[ci, 0]),
            "mAP50-95": float(ap[ci].mean()),
            "n_gt": int((tcls == c).sum()),
        }
    if len(uc):
        res["_all"] = {
            "precision": float(p.mean()),
            "recall": float(r.mean()),
            "mAP50": float(ap[:, 0].mean()),
            "mAP50-95": float(ap.mean()),
            "n_gt": int(len(tcls)),
        }
    if s.get("n_img"):
        res["_n_img"] = s["n_img"]
    return res


def fmt_table(res, title):
    lines = [f"\n===== {title} =====",
             f"{'class':<10}{'P':>8}{'R':>8}{'mAP50':>10}{'mAP50-95':>10}{'nGT':>8}"]
    for k, v in res.items():
        if k == "_n_img":
            lines.append(f"# images in subset: {v}")
            continue
        lines.append(f"{k:<10}{v['precision']:>8.4f}{v['recall']:>8.4f}{v['mAP50']:>10.4f}"
                     f"{v['mAP50-95']:>10.4f}{v['n_gt']:>8d}")
    return "\n".join(lines)


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--weights", required=True)
    ap_.add_argument("--data", default="/data/wwc/test/RGBT-3M/rgbt_full.yaml")
    ap_.add_argument("--mode", choices=["rgbt", "rgb", "ir"], default="rgbt")
    ap_.add_argument("--split", choices=["val", "test"], default="val")
    ap_.add_argument("--imgsz", type=int, default=640)
    ap_.add_argument("--batch", type=int, default=4)
    ap_.add_argument("--workers", type=int, default=0)
    ap_.add_argument("--device", default="0")
    ap_.add_argument("--conf", type=float, default=0.001)
    ap_.add_argument("--iou", type=float, default=0.7)
    args = ap_.parse_args()

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    model, names = load_model(args.weights, device)
    end2end = bool(getattr(model, "end2end", False))
    stock = is_stock_model(model)
    in_ch = model_in_channels(model)
    if stock and args.mode == "rgbt":
        raise SystemExit("[eval] 该权重是 stock 单模态模型（DetectionModel），--mode 请用 rgb 或 ir")
    loader = build_loader(args, model)
    print(f"[eval] weights={args.weights}\n[eval] mode={args.mode}  split={args.split}  device={device}"
          f"  end2end={end2end}  stock={stock}  first_conv_in={in_ch}  names={names}")

    stats_bins = {b: {"tp": [], "conf": [], "pcls": [], "tcls": [], "n_img": 0} for b in BINS}

    for bi, batch in enumerate(loader):
        img = batch["img"].to(device).float()
        if img.max() > 1.5:
            img /= 255.0
        if stock:
            img = img[:, :3] if args.mode == "rgb" else img[:, 3:6]
        else:
            if args.mode == "rgb":
                img[:, 3:] = 0.0
            elif args.mode == "ir":
                img[:, :3] = 0.0
        with torch.no_grad():
            preds = model(img)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        preds = non_max_suppression(
            preds, conf_thres=args.conf, iou_thres=args.iou,
            multi_label=True, max_det=300, end2end=end2end,
        )

        gt_cls_all = batch["cls"].squeeze(-1).cpu().numpy()
        gt_box_all = batch["bboxes"].cpu().numpy()
        batch_idx = batch["batch_idx"].cpu().numpy()
        h, w = img.shape[-2:]
        for i in range(img.shape[0]):
            m = batch_idx == i
            gcls = gt_cls_all[m]
            gxywh = gt_box_all[m]
            if len(gxywh):
                cx, cy, bw, bh = gxywh.T
                gxyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
            else:
                gxyxy = np.zeros((0, 4), dtype=np.float32)
            p = preds[i].cpu().numpy() if preds[i] is not None else np.zeros((0, 6))
            collect_image_stats(p, gcls.astype(np.int64), gxyxy.astype(np.float32), (h, w), stats_bins)
        if (bi + 1) % 50 == 0:
            print(f"[eval] {bi + 1}/{len(loader)} batches")

    results = {b: evaluate_bin(stats_bins[b], names) for b in BINS}
    out = {"weights": args.weights, "mode": args.mode, "split": args.split,
           "imgsz": args.imgsz, "stock": stock, "first_conv_in": in_ch,
           "small_area_thr": SMALL_AREA, "results": results}
    out_path = Path(args.weights).parent.parent / f"eval_small_ap_{args.split}_{args.mode}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(fmt_table(results["all"], f"OVERALL (mode={args.mode}, split={args.split})"))
    print(fmt_table(results["small"], f"SMALL area<{SMALL_AREA}"))
    print(fmt_table(results["large"], f"LARGE area>={SMALL_AREA}"))
    print(f"\n[eval] saved -> {out_path}")


if __name__ == "__main__":
    main()
