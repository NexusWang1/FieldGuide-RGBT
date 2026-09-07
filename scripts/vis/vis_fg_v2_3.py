# -*- coding: utf-8 -*-
"""v2.3 FieldGuide 效果验证：S 图可视化 + smoke 区域定量对比

用法: cd /data/wwc/install/ultralytics && python vis_fg_v2_3.py [--device cuda:1] [--n 12]
输出:
  runs/vis_fg_v23/*.jpg   RGB + S 热区叠加（红=引导图S, 绿框=smoke GT, 红框=fire GT）
  控制台：每张图 smoke区内均S / 区外均S / 倍率；末尾总平均。
  判读：倍率 >> 1（如 >2）→ S 确实压在烟羽上，FG 引导有效；
        倍率 ≈ 1 → S 弥散，引导形同虚设。
"""
import os
os.environ.setdefault("PHY_STAGE", "3")
os.environ.setdefault("PHY_KMODE", "none")
os.environ.setdefault("PHY_GDK", "0")
os.environ.setdefault("PHY_FG", "1")
os.environ.setdefault("PHY_FG_MODE", "fixed")   # 与 v2.3 训练一致：strength=α=0.5
os.environ.setdefault("PHY_FG_ALPHA", "0.5")
os.environ.setdefault("PHY_DUALSTAL", "1")
os.environ.setdefault("PHY_SMOKESTAL", "1")
os.environ.setdefault("PHY_AUXW", "0.1")

import argparse
import numpy as np
import torch
from PIL import Image, ImageDraw

p = argparse.ArgumentParser()
p.add_argument("--weights", default="runs/phy-rgb/v2.3_fgFixed05_official_sgd200/weights/last.pt")
p.add_argument("--yaml", default="/data/wwc/test/RGBT-3M/rgbt_full.yaml")
p.add_argument("--split", default="")   # 留空 = 从 yaml 的 path+val 解析
p.add_argument("--n", type=int, default=12)
p.add_argument("--device", default="cuda:1")
p.add_argument("--out", default="runs/vis_fg_v23")
args = p.parse_args()

from ultralytics.nn.tasks_rgbt_v2_3 import PhyBridgeYOLO26V23

dev = torch.device(args.device if not args.device.isdigit() else f"cuda:{args.device}")
model = PhyBridgeYOLO26V23(cfg="yolo26m-p2.yaml", ch=3, nc=3, verbose=False, scale="m",
                           stage=3, dual_stal=True, smoke_stal=True)
ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
sd = ckpt["ema"].state_dict() if hasattr(ckpt.get("ema"), "state_dict") else ckpt["model"].state_dict()
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"[load] {args.weights}  missing={len(missing)} unexpected={len(unexpected)}")
print(f"[load] fg strength = {model.fieldguide.strength}  "
      f"kernel_sum = {float(model.fieldguide.kernel.sum()):.3f}")
model.eval().float().to(dev)

cap = []
model.heatfield.register_forward_hook(lambda m, i, o: cap.append(o[1]))


def ir_of(rgb_path):
    parts = rgb_path.rsplit("/images/", 1)
    return "/image/".join(parts)


def letterbox(im, size=640, fill=114):
    w, h = im.size
    s = min(size / w, size / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    im2 = im.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new(im.mode, (size, size), fill)
    canvas.paste(im2, ((size - nw) // 2, (size - nh) // 2))
    return canvas, nw, nh, (size - nw) // 2, (size - nh) // 2


# ---- 挑选烟火同框帧，均匀取 n 张 ----
if args.split:
    split_file = args.split
else:
    # 从 yaml 解析 path + val（相对路径基于 yaml 所在目录）
    ydir = os.path.dirname(args.yaml)
    yroot, val_rel = ydir, ""
    for ln in open(args.yaml):
        ln = ln.strip()
        if ln.startswith("path:"):
            yroot = ln.split(":", 1)[1].strip()
            yroot = yroot if os.path.isabs(yroot) else os.path.join(ydir, yroot)
        elif ln.startswith("val:"):
            val_rel = ln.split(":", 1)[1].strip()
    split_file = val_rel if os.path.isabs(val_rel) else os.path.join(yroot, val_rel)
print(f"[data] split = {split_file}")
with open(split_file) as f:
    rgb_list = [ln.strip() for ln in f if ln.strip()]
# split 内可能是相对路径：优先按 yaml 的 path 根解析，其次按 split 文件所在目录
_rgb_bases = [os.path.dirname(os.path.dirname(split_file)), os.path.dirname(split_file)]
def _resolve(p_):
    if os.path.isabs(p_):
        return p_
    for b in _rgb_bases:
        c = os.path.normpath(os.path.join(b, p_))
        if os.path.exists(c):
            return c
    return os.path.normpath(os.path.join(_rgb_bases[0], p_))
rgb_list = [_resolve(p_) for p_ in rgb_list]
print(f"[data] 首条 = {rgb_list[0]}  (存在={os.path.exists(rgb_list[0])})")
cand = []
for rp in rgb_list:
    lp = rp.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    if not os.path.isfile(lp):
        continue
    cls = {ln.split()[0] for ln in open(lp) if ln.strip()}
    if "0" in cls and "1" in cls:
        cand.append((rp, lp))
step = max(1, len(cand) // args.n)
total = len(cand)
cand = cand[::step][:args.n]
print(f"[data] 烟火同框候选 {total} 帧, 均匀抽样 {len(cand)} 张")

os.makedirs(args.out, exist_ok=True)
ratios = []
for rp, lp in cand:
    name = os.path.splitext(os.path.basename(rp))[0]
    rgb_pil = Image.open(rp).convert("RGB")
    ir_pil = Image.open(ir_of(rp)).convert("RGB")
    rgb_l, nw, nh, px, py = letterbox(rgb_pil)
    ir_l, _, _, _, _ = letterbox(ir_pil)
    a = np.asarray(rgb_l, dtype=np.float32) / 255.0
    b = np.asarray(ir_pil, dtype=np.float32) / 255.0
    x = torch.from_numpy(np.concatenate([a, b], -1)).permute(2, 0, 1)[None].to(dev)

    cap.clear()
    with torch.no_grad():
        model(x)
        t = cap[-1]
        S = model.fieldguide.guide_map(t)[0, 0].cpu().numpy()  # (Hs, Ws)

    Hs, Ws = S.shape
    # GT 栅格到 S 分辨率（归一化原图坐标 -> letterbox 像素 -> S 格）
    smoke_m = np.zeros((Hs, Ws), bool)
    fire_m = np.zeros((Hs, Ws), bool)
    boxes = []
    for ln in open(lp):
        c, bx, by, bw, bh = ln.split()[:5]
        bx, by, bw, bh = map(float, (bx, by, bw, bh))
        x0 = int(max(0, ((bx - bw / 2) * nw + px) / 640 * Ws))
        y0 = int(max(0, ((by - bh / 2) * nh + py) / 640 * Hs))
        x1 = int(min(Ws, ((bx + bw / 2) * nw + px) / 640 * Ws + 0.5))
        y1 = int(min(Hs, ((by + bh / 2) * nh + py) / 640 * Hs + 0.5))
        if c == "0":
            smoke_m[y0:y1, x0:x1] = True
        elif c == "1":
            fire_m[y0:y1, x0:x1] = True
        boxes.append((c, bx, by, bw, bh))

    in_s = float(S[smoke_m].mean()) if smoke_m.any() else float("nan")
    out_m = ~(smoke_m | fire_m)
    out_s = float(S[out_m].mean())
    ratio = in_s / max(out_s, 1e-6)
    ratios.append(ratio)
    print(f"  {name}: S(smoke内)={in_s:.3f}  S(区外)={out_s:.3f}  倍率={ratio:.1f}x  Smax={S.max():.2f}")

    # 叠加图：S 热区染红
    vis = np.asarray(rgb_l, dtype=np.float32)
    S_up = np.asarray(Image.fromarray((S * 255).astype(np.uint8)).resize((640, 640), Image.BILINEAR),
                      dtype=np.float32) / 255.0
    alpha = np.clip(S_up / max(S_up.max(), 1e-6), 0, 1) * 0.55
    vis = vis * (1 - alpha[..., None]) + np.array([255, 32, 32], np.float32) * alpha[..., None]
    img = Image.fromarray(vis.astype(np.uint8))
    dr = ImageDraw.Draw(img)
    for c, bx, by, bw, bh in boxes:
        x0 = (bx - bw / 2) * nw + px
        y0 = (by - bh / 2) * nh + py
        x1 = (bx + bw / 2) * nw + px
        y1 = (by + bh / 2) * nh + py
        dr.rectangle([x0, y0, x1, y1], outline=(64, 255, 64) if c == "0" else (255, 255, 64), width=2)
    img.save(os.path.join(args.out, f"{name}.jpg"), quality=90)

print(f"\n[verdict] 平均倍率 = {np.mean(ratios):.2f}x  (>>1 则 S 压在烟羽上, FG 引导有效)")
print(f"[out] 叠加图 -> {args.out}/  绿框=smoke GT 黄框=fire GT 红晕=S")
