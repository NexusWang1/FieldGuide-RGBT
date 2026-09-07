# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""DetectionTrainer for YOLO26 RGBT — CBAM 注意力对照臂。

继承 v1.19 训练器全部内容（同 v2.3），仅 get_model 换成 CBAM 臂模型。
环境变量：v2.3 全集 + PHY_CBAM。
"""

from __future__ import annotations

import os

from ultralytics.models.yolo.detect.train_rgbt_v1_19 import RGBTDetectionTrainer, _env_flag
from ultralytics.nn.tasks_rgbt_v2_cbam import PhyBridgeYOLO26CBAM
from ultralytics.utils import LOGGER, RANK


class RGBTDetectionTrainerCBAM(RGBTDetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        stage = int(os.getenv("PHY_STAGE", "3"))
        model = self.set_model_names_for_load(
            PhyBridgeYOLO26CBAM(
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
            model.load_pretrained(weights)  # CBAM 已在 __init__ 物化，按名匹配容忍新增参数
        LOGGER.info(f"PhyBridgeYOLO26 CBAM-arm cbam={model.cbam_on} fg={model.fg_on} "
                    f"dual_stal={model.dual_stal} smoke_stal={model.smoke_stal}")
        return model
