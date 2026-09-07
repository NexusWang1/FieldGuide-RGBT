import os
os.environ["PHY_STAGE"] = "3"
# --- v2.3 = v2.1 唯一改动：去门，固定强度直注 f2'=f2·(1+0.5·S) ---
# 前置证据：v2.1 零初始化门 200ep 后 g_fg≈-0.009（注入 ~1%），FG 指标贡献为零；
#           但 K 已被 aux 学成羽流形（bot_mass 0.641→0.448, kernel_sum ×13.66）。
#           门的两个保护理由已失效：防 IR 稀释=断流误诊残案；恒等开局=永不启用。
# 本 run = 论文核心假设"火源场引导找回 smoke"的首次真实检验：
#   smoke-small 显著 >0.281（0.279+噪声底0.002）且 fire P 不崩 → 假设证实；
#   fire P 前 5 轮崩塌（v1.11/12 覆辙）→ 立即杀；指标不动 → 假设证伪翻篇。
# 其余与 v2.1 逐位一致（单变量纪律），v2.1 即 α≈0 对照臂。
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
os.environ["PHY_FG_MODE"] = "fixed"   # ★ v2.3：无门直注
os.environ["PHY_FG_ALPHA"] = "0.5"    # ★ 固定强度 α=0.5（v2.1 即事实上的 α=0 对照臂）
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train_rgbt_v2_3 import RGBTDetectionTrainerV23

if __name__ == "__main__":
    model = YOLO("yolo26m-p2.yaml")

    weights = os.getenv("RGBT_WEIGHTS", "/data/wwc/install/ultralytics/yolo26m.pt").strip()
    pretrained = False if weights.lower() in {"none", "false", "0"} else weights

    model.train(
        trainer=RGBTDetectionTrainerV23,
        data="/data/wwc/test/RGBT-3M/rgbt_full.yaml",
        pretrained=pretrained,
        epochs=200,
        imgsz=640,
        batch=4,
        workers=4,
        device="2",                     # 卡 2（v2.1 已收官）
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        cos_lr=False,
        cache=False,
        amp=False,
        # ===== 增强与 v2.1 逐位一致 =====
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
        name="v2.3_fgFixed05_official_sgd200",
    )
