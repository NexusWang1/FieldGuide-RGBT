# -*- coding: utf-8 -*-
"""场源对比可视化：baseline (v2off champion) vs ours (v2.3 FG) 的 fire-source field t。

用法（服务器 /data/wwc/install/ultralytics 下）：
    python vis_field_compare.py \
        --base runs/phy-rgb/v2off_champion_gateCMEE_sgd100/weights/best.pt \
        --ours runs/phy-rgb/v2.3_fgFixed05_official_sgd200/weights/best.pt \
        --data /data/wwc/test/RGBT-3M/rgbt_full.yaml --split val \
        --n 6 --device 0 --out runs/vis_field_compare

输出：每帧一张 5 联图 [RGB+GT | IR | baseline t | ours t | ours S]，
     外加 field_compare_stats.txt（火框内/外响应对比 = forward filtering 量化证据）。
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

CLS_NAME = {0: "smoke", 1: "fire", 2: "person"}
CLS_COLOR = {0: (90, 90, 90), 1: (230, 60, 40), 2: (60, 130, 230)}


def load_model(weights, device):
    from ultralytics import YOLO
    m = YOLO(weights, task="detect")
    model = m.model.eval().to(device)
    print(f"[load] {weights}")
    print(f"       has heatfield={hasattr(model, 'heatfield')}  "
          f"has fieldguide={hasattr(model, 'fieldguide')}  "
          f"tfam_ir={getattr(model, 'tfam_ir', '?')}  use_p2={getattr(model, 'use_p2', '?')}")
    return model


def find_cofire_frames(lbl_dir, n):
    """返回同时含 smoke(0) 与 fire(1) GT 的帧 stem 列表，均匀取 n 个。"""
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
    pick = [stems[i] for i in sorted(set(idx))]
    print(f"[data] 烟火同框 {len(stems)} 帧，取 {len(pick)} 帧: {pick}")
    return pick


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


def to_img(path, size=640):
    return np.asarray(Image.open(path).convert("RGB").resize((size, size)),
                      dtype=np.float32) / 255.0


def build_6ch(rgb_path, ir_path, device, size=640):
    rgb = to_img(rgb_path, size)
    ir = to_img(ir_path, size)
    x = np.concatenate([rgb, ir], axis=-1)          # HWC 6ch
    t = torch.from_numpy(x).permute(2, 0, 1)[None].to(device)  # 1,6,H,W
    return t.contiguous()


@torch.no_grad()
def grab_field(model, x):
    """前向一次，用 hook 抓 heatfield 输出 (field_map, t)；没抓到就手动复现前半段。"""
    buf = {}
    h = model.heatfield.register_forward_hook(
        lambda mod, inp, out: buf.setdefault("out", out))
    try:
        model(x)
    finally:
        h.remove()
    if "out" in buf:
        field_map, t = buf["out"]
        return field_map.float(), t.float(), "hook"
    # 手动回退：v2.x 共用基座，属性名一致
    ir = x[:, 3:].contiguous()
    i2, _, _, _ = model._ir_backbone(ir)
    with torch.autocast(device_type=x.device.type, enabled=False):
        i2p, _ = model.piip(i2.float())
        field_map, t = model.heatfield(i2p.float(), ir.float())
    return field_map.float(), t.float(), "manual"


def heat_to_img(t, size=640):
    """(1,1,h,w) -> 640 热力 RGB（无 matplotlib 依赖，手动 jet 近似）。"""
    a = t[0, 0].detach().cpu().numpy()
    a = np.nan_to_num(a)
    lo, hi = float(a.min()), float(a.max())
    n = (a - lo) / (hi - lo + 1e-9)
    n_img = np.asarray(Image.fromarray((n * 255).astype(np.uint8)).resize(
        (size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    r = np.clip(1.5 * n_img - 0.25, 0, 1)
    g = np.clip(1.5 - np.abs(2 * n_img - 1) * 1.5, 0, 1) * (n_img > 0.05)
    b = np.clip(1.25 - 1.5 * n_img, 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def draw_boxes(img_np, gt, size=640):
    im = Image.fromarray(img_np)
    d = ImageDraw.Draw(im)
    for c, (cx, cy, w, h_) in gt:
        x0, y0 = (cx - w / 2) * size, (cy - h_ / 2) * size
        x1, y1 = (cx + w / 2) * size, (cy + h_ / 2) * size
        d.rectangle([x0, y0, x1, y1], outline=CLS_COLOR.get(c, (255, 255, 0)), width=3)
        d.text((x0 + 2, y0 + 2), CLS_NAME.get(c, str(c)), fill=CLS_COLOR.get(c))
    return np.asarray(im)


def fire_mask(gt, hw):
    """火 GT 框的布尔掩码（在 field 分辨率 hw 上）。"""
    H, W = hw
    m = np.zeros((H, W), dtype=bool)
    for c, (cx, cy, w, h_) in gt:
        if c != 1:
            continue
        x0 = int(max(0, (cx - w / 2) * W)); x1 = int(min(W, (cx + w / 2) * W) + 0.5)
        y0 = int(max(0, (cy - h_ / 2) * H)); y1 = int(min(H, (cy + h_ / 2) * H) + 0.5)
        m[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--data", default="/data/wwc/test/RGBT-3M/rgbt_full.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--frames", nargs="*", default=None,
                    help="帧 stem 列表（与 stockrgb 脚本 --miss-only 输出对齐）；缺省自动选")
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="runs/vis_field_compare")
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    cfg = yaml_load(args.data)
    root = Path(cfg["path"])
    rgb_dir = root / cfg.get(f"{args.split}", "images/val")
    if not rgb_dir.is_dir():  # yaml 的 val 可能是相对 images 的写法
        rgb_dir = root / "images" / args.split
    # 与 eval 脚本口径对齐：split=val 的官方评测实际落在 images/test + labels/test
    if args.split == "val" and (root / "images" / "test").is_dir():
        rgb_dir = root / "images" / "test"
    leaf = rgb_dir.name
    lbl_dir = rgb_dir.parent.parent / "labels" / leaf
    # IR 配对：先在 yaml ir_images 根下找，找不到就在数据集并列目录里搜同帧
    ir_root = root / cfg.get("ir_images", "")
    ir_dirs = [ir_root] if ir_root.is_dir() else []
    ir_dirs += [p for p in sorted(root.parent.glob(f"*/images/{leaf}"))
                if p != rgb_dir and p.is_dir()]
    print(f"[data] rgb={rgb_dir}\n[data] lbl={lbl_dir}")
    print(f"[data] IR 候选目录: {[str(p) for p in ir_dirs]}")
    assert lbl_dir.is_dir(), f"标签目录不存在: {lbl_dir}"

    def find_ir(stem):
        for d in ir_dirs:
            p = next(d.glob(stem + ".*"), None)
            if p is not None:
                return p
        return None

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    mb = load_model(args.base, device)
    mo = load_model(args.ours, device)

    stems = args.frames if args.frames else find_cofire_frames(lbl_dir, args.n)
    stats = []
    for stem in stems:
        rgb_p = next(rgb_dir.glob(stem + ".*"))
        ir_p = find_ir(stem)
        if ir_p is None:
            print(f"[skip] {stem} 无 IR 配对")
            continue
        gt = read_gt(lbl_dir / f"{stem}.txt")
        x = build_6ch(rgb_p, ir_p, device)

        _, tb, how_b = grab_field(mb, x)
        _, to_, how_o = grab_field(mo, x)
        S = mo.fieldguide.guide_map(to_) if hasattr(mo, "fieldguide") else None

        # 统计：火框内均值 / 框外均值
        for name, t in [("base", tb), ("ours", to_)]:
            a = t[0, 0].cpu().numpy()
            fm = fire_mask(gt, a.shape)
            inside = float(a[fm].mean()) if fm.any() else float("nan")
            outside = float(a[~fm].mean())
            stats.append((stem, name, inside, outside, inside / (outside + 1e-9)))

        # 拼图
        tiles = [draw_boxes((to_img(rgb_p) * 255).astype(np.uint8), gt),
                 to_img(ir_p).astype(np.float32)]
        tiles[1] = (tiles[1] * 255).astype(np.uint8)
        tiles += [heat_to_img(tb), heat_to_img(to_)]
        if S is not None:
            tiles.append(heat_to_img(S))
        row = np.concatenate(tiles, axis=1)
        Image.fromarray(row).save(out_dir / f"{stem}_field.png")
        print(f"[ok] {stem}  ({how_b}/{how_o}) -> {out_dir}/{stem}_field.png")

    with open(out_dir / "field_compare_stats.txt", "w") as f:
        f.write("frame\tmodel\tfire_inside_mean\toutside_mean\tcontrast\n")
        for r in stats:
            f.write("%s\t%s\t%.4f\t%.4f\t%.2f\n" % r)
        for name in ["base", "ours"]:
            v = [s[4] for s in stats if s[1] == name and np.isfinite(s[4])]
            if v:
                line = f"{name}: 平均火框对比度 = {np.mean(v):.2f}x (n={len(v)})"
                print("[stat]", line)
                f.write(line + "\n")
    print(f"[done] 输出目录: {out_dir}")


if __name__ == "__main__":
    main()
