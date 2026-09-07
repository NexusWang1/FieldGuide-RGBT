# make_degraded_flame2.py —— 模态退化压力测试数据集生成器 v2（免训练）
# 服务器放置: /data/wwc/install/ultralytics/make_degraded_flame2.py
#
# v2 修正（2026-09-04）：数据集是双树结构，不是左右拼接！
#   images/val/<name>.jpg  = RGB（主列表，splits/val.txt 指这里）
#   image/val/<name>.*     = IR（按主名配对，后缀不限）
# v1 错误地把单张 RGB 切成左右两半分别退化，且没生成 image/ 树，导致配对全灭。
#
# 用法:
#   python make_degraded_flame2.py --dry-run   # 出 4 张预览到 preview_deg/ 人工核对
#   python make_degraded_flame2.py             # 正式生成全部退化档
# 逐档 eval:
#   for d in rgb_blur_k5 rgb_blur_k9 rgb_blur_k15 rgb_dark_060 rgb_dark_040 rgb_dark_025 \
#            ir_noise_s10 ir_noise_s20 ir_noise_s35; do
#     python eval_small_ap.py --weights runs/phy-rgb/v2.3_fgFixed05_official_sgd200/weights/best.pt \
#       --data /data/wwc/test/FLAME2_dt_v3_deg/$d/flame2dt_deg.yaml --mode rgbt --split val --device 2
#   done

import argparse
import glob
import os
import cv2
import numpy as np

DEGRADATIONS = {
    # RGB 侧失效（整图）
    "rgb_blur_k5":  dict(side="rgb", kind="blur",  k=5),
    "rgb_blur_k9":  dict(side="rgb", kind="blur",  k=9),
    "rgb_blur_k15": dict(side="rgb", kind="blur",  k=15),
    "rgb_dark_060": dict(side="rgb", kind="dark",  f=0.60),
    "rgb_dark_040": dict(side="rgb", kind="dark",  f=0.40),
    "rgb_dark_025": dict(side="rgb", kind="dark",  f=0.25),
    # IR 侧失效（整图）
    "ir_noise_s10": dict(side="ir",  kind="noise", s=10.0),
    "ir_noise_s20": dict(side="ir",  kind="noise", s=20.0),
    "ir_noise_s35": dict(side="ir",  kind="noise", s=35.0),
}

SRC = "/data/wwc/test/FLAME2_dt_v3"
DST = "/data/wwc/test/FLAME2_dt_v3_deg"


def find_ir(rgb_path):
    """与 rgbt_dataset._rgb_to_ir_path 同规则：images -> image，同主名任意后缀。"""
    stem = os.path.splitext(os.path.basename(rgb_path))[0]
    cands = sorted(glob.glob(os.path.join(SRC, "image", "val", stem + ".*")))
    assert cands, f"找不到 IR 配对: {rgb_path}"
    return cands[0]


def degrade(img, spec, rng):
    if spec["kind"] == "blur":
        return cv2.GaussianBlur(img, (spec["k"], spec["k"]), 0)
    if spec["kind"] == "dark":
        return np.clip(img.astype(np.float32) * spec["f"], 0, 255).astype(img.dtype)
    if spec["kind"] == "noise":
        n = rng.normal(0, spec["s"], img.shape).astype(np.float32)
        return np.clip(img.astype(np.float32) + n, 0, 255).astype(img.dtype)
    raise ValueError(spec["kind"])


def load(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    assert img is not None, f"读不到 {path}"
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(os.path.join(SRC, "splits", "val.txt")) as f:
        val = [ln.strip() for ln in f if ln.strip()]
    assert os.path.exists(val[0]), val[0]
    ir0 = find_ir(val[0])
    print(f"[data] val 帧数 = {len(val)}")
    print(f"[data] RGB 首条 = {val[0]}")
    print(f"[data] IR  首条 = {ir0}")

    if args.dry_run:
        os.makedirs("preview_deg", exist_ok=True)
        rng = np.random.default_rng(0)
        cv2.imwrite("preview_deg/rgb_dark_040.jpg",
                    degrade(load(val[0]), DEGRADATIONS["rgb_dark_040"], rng))
        cv2.imwrite("preview_deg/rgb_orig.jpg", load(val[0]))
        cv2.imwrite("preview_deg/ir_noise_s20.jpg",
                    degrade(load(ir0), DEGRADATIONS["ir_noise_s20"], rng))
        cv2.imwrite("preview_deg/ir_orig.jpg", load(ir0))
        print("[dry-run] 预览 -> preview_deg/  核对: rgb_dark 应整图变暗, ir_noise 应整图加噪")
        return

    for name, spec in DEGRADATIONS.items():
        droot = os.path.join(DST, name)
        rgbdir = os.path.join(droot, "images", "val")
        irdir = os.path.join(droot, "image", "val")
        os.makedirs(rgbdir, exist_ok=True)
        os.makedirs(irdir, exist_ok=True)
        if not os.path.exists(os.path.join(droot, "labels")):
            os.symlink(os.path.join(SRC, "labels"), os.path.join(droot, "labels"))
        os.makedirs(os.path.join(droot, "splits"), exist_ok=True)

        rng = np.random.default_rng(42)
        new_list = []
        for rgb_path in val:
            ir_path = find_ir(rgb_path)
            stem_jpg = os.path.basename(rgb_path)
            rgb_out = os.path.join(rgbdir, stem_jpg)
            ir_out = os.path.join(irdir, os.path.basename(ir_path))

            if spec["side"] == "rgb":
                if not os.path.exists(rgb_out):
                    cv2.imwrite(rgb_out, degrade(load(rgb_path), spec, rng))
                if not os.path.exists(ir_out):
                    os.symlink(ir_path, ir_out)          # IR 原样软链
            else:
                if not os.path.exists(rgb_out):
                    os.symlink(rgb_path, rgb_out)        # RGB 原样软链
                if not os.path.exists(ir_out):
                    cv2.imwrite(ir_out, degrade(load(ir_path), spec, rng))
            new_list.append(rgb_out)

        with open(os.path.join(droot, "splits", "val.txt"), "w") as f:
            f.write("\n".join(new_list) + "\n")
        with open(os.path.join(SRC, "flame2dt_v3.yaml")) as f:
            y = f.read()
        y = y.replace(SRC, droot)
        with open(os.path.join(droot, "flame2dt_deg.yaml"), "w") as f:
            f.write(y)
        print(f"[OK] {name}: {len(new_list)} 帧 -> {droot}")

    print("[done] 逐档 eval 命令见脚本 docstring")


if __name__ == "__main__":
    main()
