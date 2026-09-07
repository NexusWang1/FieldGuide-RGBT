# --- FLAME2-DT 迁移微调 v3b [α扫描 1.0]：labels_merged3 + v2.3 冠军(含FG直注)初始化 ---
# env/import 与服务器 train_phy_bridge_v2_3.py 逐位一致（已核对 2026-09-03）。
# 单变量 = 数据(FLAME2 v3) + 初始化(v2.3 best.pt) + 轮数(100)。
# 服务器放置路径: /data/wwc/install/ultralytics/train_phy_bridge_flame2ft_v3_fg.py
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
os.environ["PHY_FG"] = "1"
os.environ["PHY_FGAUX"] = "0.02"
os.environ["PHY_FGWARM"] = "20000"
os.environ["PHY_FG_MODE"] = "fixed"   # 无门直注（与 v2.3 主训一致）
os.environ["PHY_FG_ALPHA"] = "1.0"
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train_rgbt_v2_3 import RGBTDetectionTrainerV23


def _fg_selfcheck(trainer):
    """on_train_start：模型已构建、首轮未跑——FG 缺装立刻自爆，不浪费机时。"""
    fg = [n for n, _ in trainer.model.named_modules() if "fieldguide" in n]
    assert fg, (
        "FG 未装配！checkpoint 将不含 fieldguide.kernel。"
        "检查 trainer 类(应为 RGBTDetectionTrainerV23)与 PHY_FG* 环境变量。"
    )
    print(f"[SELF-CHECK OK] fieldguide 已装配: {fg}")


if __name__ == "__main__":
    model = YOLO("yolo26m-p2.yaml")
    model.add_callback("on_train_start", _fg_selfcheck)

    weights = os.getenv(
        "RGBT_WEIGHTS",
        "runs/phy-rgb/v2.3_fgFixed05_official_sgd200/weights/best.pt",
    ).strip()
    pretrained = False if weights.lower() in {"none", "false", "0"} else weights

    model.train(
        trainer=RGBTDetectionTrainerV23,
        data="/data/wwc/test/FLAME2_dt_v3/flame2dt_v3.yaml",
        pretrained=pretrained,
        epochs=100,
        imgsz=640,
        batch=4,
        workers=4,
        device="2",                     # 起跑前按空闲卡改
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        cos_lr=False,
        cache=False,
        amp=False,                      # fp32，与 v2 微调一致
        # ===== 增强配方与 v1/v2 微调逐项一致 =====
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
        name="flame2ft_lblv3_fgA100_sgd100",
    )
