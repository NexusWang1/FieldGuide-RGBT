# --- CBAM 注意力对照臂 = 基线(v2off champion) 逐位一致 + 颈部 CBAM ---
# 单变量纪律：与基线唯一差异 = PHY_CBAM=1（P2/P3/P4/P5 融合出口各一个 CBAM）。
# 用途：评审 Major② 中策——同协议评注意力路线代表，与 FieldGuide 直注臂(v2.3)对比。
# 服务器放置路径: /data/wwc/install/ultralytics/train_phy_bridge_v2_cbam.py
import os
os.environ["PHY_STAGE"] = "3"
os.environ["PHY_KMODE"] = "none"
os.environ["PHY_TFAM_IR"] = "1"
os.environ["PHY_PSCMT_RES"] = "0"
os.environ["PHY_DUALSTAL"] = "1"
os.environ["PHY_STALW"] = "0.25"
os.environ["PHY_SMOKESTAL"] = "1"
os.environ["PHY_SMOKEW"] = "0.25"
os.environ["PHY_SMOKEDIL"] = "3.0"
os.environ["PHY_AUXW"] = "0.1"
os.environ["PHY_GDK"] = "1"
os.environ["PHY_GDKAUX"] = "0.0"
os.environ["PHY_GDKWARM"] = "20000"
# ===== 与 v2.3 的差异 1/2：FG 关（本臂是"基线+CBAM"，不含 FieldGuide）=====
os.environ["PHY_FG"] = "0"
os.environ["PHY_FGAUX"] = "0.0"
# ===== 与 v2.3 的差异 2/2：CBAM 开 =====
os.environ["PHY_CBAM"] = "1"
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train_rgbt_v2_cbam import RGBTDetectionTrainerCBAM


def _cbam_selfcheck(trainer):
    """on_train_start：CBAM 缺装立刻自爆，不浪费机时。"""
    cb = [n for n, _ in trainer.model.named_modules() if "cbams." in n]
    assert cb, (
        "CBAM 未装配！检查 trainer 类(应为 RGBTDetectionTrainerCBAM)"
        "与 PHY_CBAM 环境变量。"
    )
    print(f"[SELF-CHECK OK] CBAM 已装配 {len(set(n.split('.')[1] for n in cb))} 组: "
          f"{sorted(set(n.split('.')[1] for n in cb))}")


if __name__ == "__main__":
    model = YOLO("yolo26m-p2.yaml")
    model.add_callback("on_train_start", _cbam_selfcheck)

    weights = os.getenv("RGBT_WEIGHTS", "/data/wwc/install/ultralytics/yolo26m.pt").strip()
    pretrained = False if weights.lower() in {"none", "false", "0"} else weights

    model.train(
        trainer=RGBTDetectionTrainerCBAM,
        data="/data/wwc/test/RGBT-3M/rgbt_full.yaml",
        pretrained=pretrained,
        epochs=200,
        imgsz=640,
        batch=4,
        workers=4,
        device="0",                     # 卡 0（baseline/seed1 在卡1/卡2）
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        cos_lr=False,
        cache=False,
        amp=False,
        # ===== 增强与 v2.3 逐位一致 =====
        mosaic=1.0,
        close_mosaic=10,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        erasing=0.0,
        auto_augment=None,
        bgr=0.0,
        project="/data/wwc/install/ultralytics/runs/phy-rgb",
        name="v2cbam_neck_official_sgd200",
    )
