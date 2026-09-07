# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""DetectionTrainer for YOLO26 RGBT training."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from ultralytics.data.rgbt_dataset import RGBTYOLODataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks_rgbt import PhyBridgeYOLO26
from ultralytics.utils import LOGGER, RANK, colorstr
from ultralytics.utils.torch_utils import unwrap_model


class RGBTDetectionTrainer(DetectionTrainer):
    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        raw = yaml.safe_load(open(self.args.data, encoding="utf-8"))
        ir_images = raw.get("ir_images", "image")
        ir_root = str(Path(raw.get("path", self.data.get("path", ""))) / ir_images)
        if self.args.cache:
            LOGGER.warning("RGBT patch forces cache=False to avoid stale RGB-only image caches.")
        return RGBTYOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect or mode == "val",
            cache=False,
            single_cls=self.args.single_cls or False,
            stride=gs,
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
            ir_root=ir_root,
            mode=mode,
            ir_channel_mode="rgb3",
            drop_unpaired=True,
        )

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        stage = int(os.getenv("PHY_STAGE", "1"))
        field = set(filter(None, (s.strip() for s in os.getenv("PHY_FIELD", "").split(","))))
        unknown = field - {"guide", "calib", "boost"}
        if unknown:
            raise ValueError(f"PHY_FIELD 含未知开关: {sorted(unknown)}（可选: guide,calib,boost）")
        pscmt_res = os.getenv("PHY_PSCMT_RES", "0").strip() in {"1", "true", "True"}
        model = self.set_model_names_for_load(
            PhyBridgeYOLO26(
                cfg=cfg or "yolo26m-p2.yaml",
                ch=3,
                nc=self.data["nc"],
                verbose=verbose and RANK == -1,
                scale="m",
                stage=stage,
                field=field,
                pscmt_res=pscmt_res,
            )
        )
        if weights:
            model.load_pretrained(weights)
        LOGGER.info(f"PhyBridgeYOLO26 stage={stage} (1=P2 only, 2=+P3, 3=+P4, 4=+P5) "
                    f"field={sorted(field) or 'off'} pscmt_res={pscmt_res}")
        return model
