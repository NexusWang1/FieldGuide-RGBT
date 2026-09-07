# -*- coding: utf-8 -*-
"""原生 YOLO26 RGB 单模态检测可视化 —— 与 FieldGuide 引导图同帧对比用。

用法（服务器 /data/wwc/install/ultralytics 下）：
    # 自动选帧（与 vis_field_compare.py 相同的确定性烟火同框选帧）
    python vis_stockrgb_predict.py \
        --weights runs/phy-rgb/stock26_rgb_v2p1_sgd100/weights/best.pt \
        --data /data/wwc/test/RGBT-3M/rgbt_full.yaml --split val \
        --n 6 --device 0 --out runs/vis_stockrgb

    # 指定帧（与图4现有面板对齐时用）
    python vis_stockrgb_predict.py ... --frames video3_frame_00012 video5_frame_00187

输出：每帧一张 PNG（GT 框 + 原生RGB模型检测框同图），
     外加 stockrgb_summary.txt（每帧 GT/检出数量与最高置信度）。
配色与 fig4 一致：GT smoke=绿、fire=黄；检测框 smoke=红、fire=橙、person=蓝。
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

import yaml

def yaml_load(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

GT_COLOR = {0: (80, 200, 80), 1: (240, 220, 60), 2: (200, 200, 200)}
DT_COLOR = {0: (235, 60, 50), 1: (245, 150, 40), 2: (70, 130, 240)}
CLS_NAME = {0: "smoke", 1: "fire", 2: "person"}


def find_cofire_frames(lbl_dir, n):
    stems = []
    for f in sorted(Path(lbl_dir).glob("*.txt")):
        clss = set()
        for ln in f.read_text().splitlines():
            p = ln.split()
            if len(p) >= 5:
                try:
                    clss.add(int(float(p[0])))
                except ValueError:
                    continue
        if 0 in clss and 1 in clss:
            stems.append(f.stem)
    if not stems:
        raise SystemExit("没有找到烟火同框帧")
    idx = np.linspace(0, len(stems) - 1, min(n, len(stems))).round().astype(int)
    return [stems[i] for i in sorted(set(idx))]


def read_gt(lbl_path):
    rows = []
    for ln in Path(lbl_path).read_text().splitlines():
        p = ln.split()
        if len(p) >= 5:
            try:
                rows.append((int(float(p[0])), [float(v) for v in p[1:5]]))
            except ValueError:
                continue
    return rows


def draw_frame(img, gt, det, size=640):
    """img: uint8 HWC; gt: [(c,[cx,cy,w,h])] 归一化; det: [(c,conf,[x0,y0,x1,y1])] 像素。"""
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    for c, (cx, cy, w, h) in gt:
        x0, y0 = (cx - w / 2) * size, (cy - h / 2) * size
        x1, y1 = (cx + w / 2) * size, (cy + h / 2) * size
        d.rectangle([x0, y0, x1, y1], outline=GT_COLOR.get(c), width=2)
    for c, conf, (x0, y0, x1, y1) in det:
        d.rectangle([x0, y0, x1, y1], outline=DT_COLOR.get(c), width=3)
        d.text((x0 + 2, max(0, y0 - 12)), f"{CLS_NAME.get(c, c)} {conf:.2f}",
               fill=DT_COLOR.get(c))
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="/data/wwc/test/RGBT-3M/rgbt_full.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--frames", nargs="*", default=None, help="帧 stem 列表；缺省自动选")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--miss-only", action="store_true",
                    help="自动筛：含火GT+小烟GT且原生RGB漏掉全部小烟的帧")
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="runs/vis_stockrgb")
    args = ap.parse_args()

    cfg = yaml_load(args.data)
    root = Path(cfg["path"])
    rgb_dir = root / cfg.get(f"{args.split}", f"images/{args.split}")
    if not rgb_dir.is_dir():
        rgb_dir = root / "images" / args.split
    # 与 eval 脚本口径对齐：split=val 的官方评测实际落在 images/test + labels/test
    if args.split == "val" and (root / "images" / "test").is_dir():
        rgb_dir = root / "images" / "test"
    lbl_dir = rgb_dir.parent.parent / "labels" / rgb_dir.name
    print(f"[data] rgb={rgb_dir}\n[data] lbl={lbl_dir}")

    stems = args.frames if args.frames else find_cofire_frames(lbl_dir, args.n)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(args.weights, task="detect")
    size = 640

    if args.miss_only:
        # 自动筛：含火GT + 小烟GT(面积<0.01)，且原生RGB漏掉全部小烟的帧
        cands = []
        for f in sorted(lbl_dir.glob("*.txt")):
            gt = read_gt(f)
            smalls = [b for c, b in gt if c == 0 and b[2] * b[3] < 0.01]
            fires = [b for c, b in gt if c == 1]
            if smalls and fires:
                cands.append((f.stem, smalls))
        print(f"[miss-only] 含火+小烟候选 {len(cands)} 帧，逐帧预测筛选...")
        picked = []
        for i, (stem, smalls) in enumerate(cands):
            rgb_p = next(rgb_dir.glob(stem + ".*"), None)
            if rgb_p is None:
                continue
            r = model.predict(source=str(rgb_p), conf=args.conf, imgsz=size,
                              device=args.device, verbose=False)[0]
            dets = []
            if r.boxes is not None and len(r.boxes):
                W0, H0 = r.boxes.orig_shape[1], r.boxes.orig_shape[0]
                for b in r.boxes:
                    if int(b.cls) == 0:
                        x0, y0, x1, y1 = b.xyxy[0].tolist()
                        dets.append((x0 / W0, y0 / H0, x1 / W0, y1 / H0))
            def hit(sb):  # 小烟GT中心是否落入任一smoke检测框
                cx, cy = sb[0], sb[1]
                return any(x0 <= cx <= x1 and y0 <= cy <= y1
                           for x0, y0, x1, y1 in dets)
            missed = [sb for sb in smalls if not hit(sb)]
            if len(missed) == len(smalls):  # 小烟全漏
                picked.append((stem, len(smalls), min(s[2]*s[3] for s in smalls)))
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(cands)}] 已筛出 {len(picked)} 帧")
        # 排序：小烟越小越靠前（最极端的早期火情画面）
        picked.sort(key=lambda t: t[2])
        stems = [p[0] for p in picked[:args.n]]
        print(f"[miss-only] 小烟全漏帧 {len(picked)} 个，取最小的 {len(stems)} 帧:")
        print("  --frames " + " ".join(stems))

    print(f"[data] 处理 {len(stems)} 帧: {stems}")
    lines = []
    for stem in stems:
        rgb_p = next(rgb_dir.glob(stem + ".*"))
        gt = read_gt(lbl_dir / f"{stem}.txt")
        im0 = Image.open(rgb_p).convert("RGB")
        W0, H0 = im0.size
        img = np.asarray(im0.resize((size, size)), dtype=np.uint8)

        r = model.predict(source=str(rgb_p), conf=args.conf, imgsz=size,
                          device=args.device, verbose=False)[0]
        det = []
        if r.boxes is not None and len(r.boxes):
            sx, sy = size / r.boxes.orig_shape[1], size / r.boxes.orig_shape[0]
            for b in r.boxes:
                x0, y0, x1, y1 = b.xyxy[0].tolist()
                det.append((int(b.cls), float(b.conf),
                            (x0 * sx, y0 * sy, x1 * sx, y1 * sy)))

        panel = draw_frame(img, gt, det, size)
        Image.fromarray(panel).save(out_dir / f"{stem}_stockrgb.png")

        n_gt = {c: sum(1 for g in gt if g[0] == c) for c in (0, 1)}
        n_dt = {c: [d for d in det if d[0] == c] for c in (0, 1)}
        msg = (f"{stem}: GT smoke={n_gt[0]} fire={n_gt[1]} | "
               f"检出 smoke={len(n_dt[0])}"
               f"(max {max((d[1] for d in n_dt[0]), default=0):.2f}) "
               f"fire={len(n_dt[1])}"
               f"(max {max((d[1] for d in n_dt[1]), default=0):.2f})")
        print("[ok]", msg, f"-> {out_dir}/{stem}_stockrgb.png")
        lines.append(msg)

    (out_dir / "stockrgb_summary.txt").write_text("\n".join(lines) + "\n")
    print(f"[done] 输出目录: {out_dir}")


if __name__ == "__main__":
    main()
