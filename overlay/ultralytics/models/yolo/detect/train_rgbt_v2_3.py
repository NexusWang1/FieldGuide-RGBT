# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""DetectionTrainer for YOLO26 RGBT — v2.3 (v2.x 底座 + FieldGuide 羽流引导)。

继承 v1.19 训练器全部内容（同 v2.0），仅 get_model 换成 v2.3 模型。
环境变量：v1.19 全集 + PHY_GDK/PHY_KMODE/PHY_GDKAUX/PHY_GDKWARM
          + PHY_FG/PHY_FGAUX/PHY_FGWARM。
"""

from __future__ import annotations

import os

from ultralytics.models.yolo.detect.train_rgbt_v1_19 import RGBTDetectionTrainer, _env_flag
from ultralytics.nn.tasks_rgbt_v2_3 import PhyBridgeYOLO26V23
from ultralytics.utils import LOGGER, RANK


class RGBTDetectionTrainerV23(RGBTDetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        stage = int(os.getenv("PHY_STAGE", "3"))
        model = self.set_model_names_for_load(
            PhyBridgeYOLO26V23(
                cfg=cfg or "yolo26m-p2.yaml",
                ch=3,
                nc=self.data["nc"],
                verbose=verbose and RANK == -1,
                scale="m",
                stage=stage,
                pscmt_res=_env_flag("PHY_PSCMT_RES"),
                tfam_ir=_env_flag("PHY_TFAM_IR", "1"),
                dual_stal=_env_flag("PHY_DUALSTAL"),
                stal_w=float(os.getenv("PHY_STALW", "0.25")),
                smoke_stal=_env_flag("PHY_SMOKESTAL"),
                smoke_w=float(os.getenv("PHY_SMOKEW", "0.25")),
                smoke_dilate=float(os.getenv("PHY_SMOKEDIL", "3.0")),
            )
        )
        if weights:
            model.load_pretrained(weights)  # 内部按 PHY_KMODE 完成高斯核包壳（或跳过）
        LOGGER.info(f"PhyBridgeYOLO26 v2.3 (FieldGuide + KMODE={os.getenv('PHY_KMODE', 'none')} + fp32 save) "
                    f"stage={stage} fg={model.fg_on} fg_aux_w={model.fg_aux_w} "
                    f"dual_stal={model.dual_stal} smoke_stal={model.smoke_stal} "
                    f"aux_w={os.getenv('PHY_AUXW', '0.0')}")
        return model
